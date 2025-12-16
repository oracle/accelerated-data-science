#!/usr/bin/env python

import logging
import traceback
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from optuna.trial import TrialState
from sktime.performance_metrics.forecasting import mean_squared_error, \
    mean_absolute_percentage_error
from sktime.split import ExpandingWindowSplitter
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from ads.opctl import logger
from ads.opctl.operator.lowcode.forecast.operator_config import ForecastOperatorConfig
from ads.opctl.operator.lowcode.forecast.utils import (_label_encode_dataframe, smape)
from .base_model import ForecastOperatorBaseModel
from .forecast_datasets import ForecastDatasets, ForecastOutput
from ..const import (
    SupportedModels, ForecastOutputColumns, DEFAULT_TRIALS,
)

logging.getLogger("report_creator").setLevel(logging.WARNING)


def freq_to_sp(freq: str) -> int | None:
    """
    Convert pandas freq string to seasonal period (sp).
    """
    if not freq:
        return None

    freq = freq.upper()

    # Direct mappings
    mapping = {
        "M": 12,
        "Q": 4,
        "A": 1,
        "Y": 1,
        "W": 52,
        "D": 7,
        "H": 24,
        "T": 1440,
        "MIN": 1440,
    }
    if freq in mapping:
        return mapping[freq]

    # Weekly variants (W-MON, W-SUN, etc.)
    if freq.startswith("W"):
        return 52

    # Minute frequencies like "5T" or "15MIN"
    if freq.endswith("T"):
        try:
            return 1440 // int(freq[:-1])
        except ValueError:
            pass

    if freq.endswith("MIN"):  # e.g., "15MIN"
        try:
            return 1440 // int(freq[:-3])
        except ValueError:
            pass

    logger.warning("Unable to infer data frequency and sp")
    return None


class ETSOperatorModel(ForecastOperatorBaseModel):
    """ETS operator model"""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config=config, datasets=datasets)
        self.global_explanation = {}
        self.local_explanation = {}

    def set_kwargs(self):
        """Prepare kwargs for ETS model from spec.
           The operator's 'model_kwargs' is respected.
        """
        model_kwargs = self.spec.model_kwargs
        model_kwargs["alpha"] = self.spec.model_kwargs.get("alpha", 0.9)
        model_kwargs["error"] = self.spec.model_kwargs.get("error", "add")
        model_kwargs["trend"] = self.spec.model_kwargs.get("trend", None)
        model_kwargs["damped_trend"] = self.spec.model_kwargs.get("damped_trend", False)
        model_kwargs["seasonal"] = self.spec.model_kwargs.get("seasonal", None)
        model_kwargs["seasonal_periods"] = self.spec.model_kwargs.get("seasonal_periods", None)
        model_kwargs["initialization_method"] = self.spec.model_kwargs.get("initialization_method", "estimated")

        if self.spec.confidence_interval_width is None:
            self.spec.confidence_interval_width = 1 - 0.90 if model_kwargs["alpha"] is None else model_kwargs["alpha"]

        model_kwargs["interval_width"] = self.spec.confidence_interval_width
        return model_kwargs

    def preprocess(self, data, series_id):
        self.le[series_id], df_encoded = _label_encode_dataframe(
            data,
            no_encode={self.spec.datetime_column.name, self.original_target_column},
        )
        return df_encoded.set_index(self.spec.datetime_column.name)

    def _train_model(self, i, series_id, df: pd.DataFrame, model_kwargs: Dict[str, Any]):
        try:
            self.forecast_output.init_series_output(series_id=series_id, data_at_series=df)
            data = self.preprocess(df, series_id)
            data_i = self.drop_horizon(data)
            freq = pd.infer_freq(data_i.index)
            Y = data_i[self.spec.target_column]
            Y = Y.asfreq(freq)

            inferred_sp = freq_to_sp(freq)
            if inferred_sp is not None:
                if model_kwargs["seasonal"] is None:
                    model_kwargs["seasonal"] = "add"
                if model_kwargs["seasonal_periods"] is None:
                    model_kwargs["seasonal_periods"] = inferred_sp

            if self.loaded_models is not None and series_id in self.loaded_models:
                previous_res = self.loaded_models[series_id].get("model")
                model_kwargs["error"] = previous_res.model.error
                model_kwargs["trend"] = previous_res.model.trend
                model_kwargs["damped_trend"] = previous_res.damped_trend
                model_kwargs["seasonal"] = previous_res.model.seasonal
                model_kwargs["seasonal_periods"] = previous_res.model.seasonal_periods
                model_kwargs["initialization_method"] = previous_res.model.initialization_method
            else:
                if self.perform_tuning:
                    model_kwargs = self.run_tuning(Y, model_kwargs)

            use_seasonal = (model_kwargs["seasonal"] is not None and
                            model_kwargs["seasonal_periods"] is not None and
                            len(Y) >= 2 * model_kwargs["seasonal_periods"]
                            )
            if not use_seasonal:
                model_kwargs["seasonal"] = None
                model_kwargs["seasonal_periods"] = None

            model = ETSModel(Y, error=model_kwargs["error"], trend=model_kwargs["trend"],
                             damped_trend=model_kwargs["damped_trend"], seasonal=model_kwargs["seasonal"],
                             seasonal_periods=model_kwargs["seasonal_periods"],
                             initialization_method=model_kwargs["initialization_method"],
                             initial_level=model_kwargs.get("initial_level", None),
                             initial_trend=model_kwargs.get("initial_trend", None),
                             initial_seasonal=model_kwargs.get("initial_seasonal", None),
                             bounds=model_kwargs.get("bounds", None), )
            fit = model.fit()
            fitted_values = fit.fittedvalues
            forecast_values = fit.forecast(self.spec.horizon)
            f1 = fit.get_prediction(start=len(Y), end=len(Y) + self.spec.horizon - 1)
            forecast_bounds = f1.summary_frame(alpha=0.9)

            forecast = pd.DataFrame(
                pd.concat(
                    [forecast_values, forecast_bounds["pi_lower"], forecast_bounds["pi_upper"]],
                    axis=1,
                ),
                index=forecast_bounds.index,
            )

            forecast.columns = ["yhat", "yhat_lower", "yhat_upper"]

            logger.debug(f"-----------------Model {i}----------------------")
            logger.debug(forecast[["yhat", "yhat_lower", "yhat_upper"]].tail())

            self.forecast_output.populate_series_output(
                series_id=series_id,
                fit_val=fitted_values.values,
                forecast_val=forecast["yhat"].values,
                upper_bound=forecast["yhat_upper"].values,
                lower_bound=forecast["yhat_lower"].values,
            )
            self.outputs[series_id] = forecast
            self.models[series_id] = {}
            self.models[series_id]["model"] = fit
            self.models[series_id]["le"] = self.le[series_id]

            params = vars(model).copy()
            for param in ["arima_res_", "endog_index_"]:
                if param in params:
                    params.pop(param)
            self.model_parameters[series_id] = {
                "framework": SupportedModels.Arima,
                **params,
            }

            logger.debug("===========Done===========")

        except Exception as e:
            self.errors_dict[series_id] = {
                "model_name": self.spec.model,
                "error": str(e),
                "error_trace": traceback.format_exc(),
            }
            logger.error(f"Encountered Error: {e}. Skipping.")
            logger.error(traceback.format_exc())

    def _build_model(self) -> pd.DataFrame:
        """Build models for all series in parallel and return forecast long format."""
        full_data_dict = self.datasets.get_data_by_series()
        self.models = {}
        self.outputs = {}
        self.explanations_info = {}
        model_kwargs = self.set_kwargs()
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width,
            horizon=self.spec.horizon,
            target_column=self.original_target_column,
            dt_column=self.spec.datetime_column.name,
        )

        Parallel(n_jobs=-1, require="sharedmem")(
            delayed(ETSOperatorModel._train_model)(
                self, i, series_id, df, model_kwargs.copy()
            )
            for self, (i, (series_id, df)) in zip(
                [self] * len(full_data_dict), enumerate(full_data_dict.items())
            )
        )

        return self.forecast_output.get_forecast_long()

    def run_tuning(self, y: pd.Series, model_kwargs_i: Dict[str, Any]):
        scoring = {
            "mape": lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred),
            "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            "mse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
            "smape": lambda y_true, y_pred: smape(y_true, y_pred),
        }

        score_fn = scoring.get(
            self.spec.metric.lower(),
            scoring["mape"]
        )

        def objective(trial):

            error = trial.suggest_categorical("error", ["add", "mul"])
            trend = trial.suggest_categorical("trend", ["add", "mul", None])
            damped_trend = trial.suggest_categorical("damped_trend", [True, False])
            initialization_method = trial.suggest_categorical(
                "initialization_method", ["estimated", "heuristic"]
            )

            if (error == "mul" or trend == "mul" or model_kwargs_i["seasonal"] == "mul") and (y <= 0).any():
                raise optuna.exceptions.TrialPruned()

            # Invalid combination
            if trend is None and damped_trend:
                raise optuna.exceptions.TrialPruned()

            cv = ExpandingWindowSplitter(
                initial_window=max(50, self.spec.horizon * 3),
                step_length=self.spec.horizon,
                fh=np.arange(1, self.spec.horizon + 1),
            )

            scores = []

            for train_idx, test_idx in cv.split(y):

                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                try:
                    model = ETSModel(
                        y_train,
                        error=error,
                        trend=trend,
                        damped_trend=damped_trend,
                        seasonal=model_kwargs_i.get("seasonal"),
                        seasonal_periods=model_kwargs_i.get("seasonal_periods"),
                        initialization_method=initialization_method,
                        initial_level=model_kwargs_i.get("initial_level"),
                        initial_trend=model_kwargs_i.get("initial_trend"),
                        initial_seasonal=model_kwargs_i.get("initial_seasonal"),
                        bounds=model_kwargs_i.get("bounds"),
                    )

                    fit = model.fit()
                    y_pred = fit.forecast(len(y_test))

                    score = score_fn(y_test.values, y_pred.values)
                    scores.append(score)

                except Exception:
                    continue

            if len(scores) == 0:
                raise optuna.exceptions.TrialPruned()

            return float(np.mean(scores))

        study = optuna.create_study(direction="minimize")

        n_trials = (
            DEFAULT_TRIALS
            if self.spec.tuning.n_trials is None
            else self.spec.tuning.n_trials
        )

        study.optimize(objective, n_trials=n_trials)

        completed_trials = [
            t for t in study.trials
            if t.state == TrialState.COMPLETE
        ]

        if not completed_trials:
            return model_kwargs_i

        model_kwargs_i.update({
            "error": study.best_params["error"],
            "trend": study.best_params["trend"],
            "damped_trend": study.best_params["damped_trend"],
            "initialization_method": study.best_params["initialization_method"],
        })

        return model_kwargs_i

    def _generate_report(self):
        import report_creator as rc
        """The method that needs to be implemented on the particular model level."""
        all_sections = []

        if len(self.models) > 0:
            sec5_text = rc.Heading("ETS Model Parameters", level=2)
            blocks = [
                rc.Html(
                    m["model"].summary().as_html(),
                    label=s_id if self.target_cat_col else None,
                )
                for i, (s_id, m) in enumerate(self.models.items())
            ]
            sec5 = rc.Select(blocks=blocks) if len(blocks) > 1 else blocks[0]
            all_sections = [sec5_text, sec5]

        if self.spec.generate_explanations:
            try:
                # If the key is present, call the "explain_model" method
                self.explain_model()

                # Convert the global explanation data to a DataFrame
                global_explanation_df = pd.DataFrame(self.global_explanation)

                self.formatted_global_explanation = (
                        global_explanation_df / global_explanation_df.sum(axis=0) * 100
                )
                self.formatted_global_explanation = (
                    self.formatted_global_explanation.rename(
                        {self.spec.datetime_column.name: ForecastOutputColumns.DATE},
                        axis=1,
                    )
                )
                aggregate_local_explanations = pd.DataFrame()
                for s_id, local_ex_df in self.local_explanation.items():
                    local_ex_df_copy = local_ex_df.copy()
                    local_ex_df_copy["Series"] = s_id
                    aggregate_local_explanations = pd.concat(
                        [aggregate_local_explanations, local_ex_df_copy], axis=0
                    )
                self.formatted_local_explanation = aggregate_local_explanations

                if not self.target_cat_col:
                    self.formatted_global_explanation = (
                        self.formatted_global_explanation.rename(
                            {"Series 1": self.original_target_column},
                            axis=1,
                        )
                    )
                    self.formatted_local_explanation.drop(
                        "Series", axis=1, inplace=True
                    )

                # Create a markdown section for the global explainability
                global_explanation_section = rc.Block(
                    rc.Heading("Global Explanation of Models", level=2),
                    rc.Text(
                        "The following tables provide the feature attribution for the global explainability."
                    ),
                    rc.DataTable(self.formatted_global_explanation, index=True),
                )

                blocks = [
                    rc.DataTable(
                        local_ex_df.div(local_ex_df.abs().sum(axis=1), axis=0) * 100,
                        label=s_id if self.target_cat_col else None,
                        index=True,
                    )
                    for s_id, local_ex_df in self.local_explanation.items()
                ]
                local_explanation_section = rc.Block(
                    rc.Heading("Local Explanation of Models", level=2),
                    rc.Select(blocks=blocks) if len(blocks) > 1 else blocks[0],
                )

                # Append the global explanation text and section to the "all_sections" list
                all_sections = all_sections + [
                    global_explanation_section,
                    local_explanation_section,
                ]
            except Exception as e:
                logger.warning(f"Failed to generate Explanations with error: {e}.")
                logger.debug(f"Full Traceback: {traceback.format_exc()}")

        model_description = rc.Text(
            "ETS stands for Error, Trend, Seasonal. An ETS forecaster is a classical time-series forecasting model "
            "that explains a series using these three components and extrapolates them into the future."
        )
        other_sections = all_sections

        return (
            model_description,
            other_sections,
        )

    def get_explain_predict_fn(self, series_id):
        def _custom_predict(
                data,
                model=self.models[series_id]["model"],
        ):
            """
            data: ForecastDatasets.get_data_at_series(s_id)
            """
            return model.predict(start=0, end=len(data) - 1)

        return _custom_predict
