#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import traceback
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd
from pandas.tseries.frequencies import to_offset
from joblib import Parallel, delayed
from optuna.trial import TrialState
from scipy.stats import linregress
from sktime.split import ExpandingWindowSplitter
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from ads.opctl import logger
from ads.opctl.operator.lowcode.common.utils import find_seasonal_period_from_dataset
from ads.opctl.operator.lowcode.forecast.operator_config import ForecastOperatorConfig
from ads.opctl.operator.lowcode.forecast.utils import (_label_encode_dataframe, _build_metrics_df)
from .forecast_datasets import ForecastDatasets, ForecastOutput
from .univariate_model import UnivariateForecasterOperatorModel
from ..const import (
    SupportedModels, DEFAULT_TRIALS,
)

logging.getLogger("report_creator").setLevel(logging.WARNING)


class ETSOperatorModel(UnivariateForecasterOperatorModel):
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
        model_kwargs["alpha"] = self.spec.model_kwargs.get("alpha", None)
        model_kwargs["error"] = self.spec.model_kwargs.get("error", "add")
        model_kwargs["trend"] = self.spec.model_kwargs.get("trend", None)
        model_kwargs["damped_trend"] = self.spec.model_kwargs.get("damped_trend", False)
        model_kwargs["seasonal"] = self.spec.model_kwargs.get("seasonal", None)
        model_kwargs["seasonal_periods"] = self.spec.model_kwargs.get("seasonal_periods", None)
        model_kwargs["initialization_method"] = self.spec.model_kwargs.get("initialization_method", "estimated")

        if self.spec.confidence_interval_width is None:
            self.spec.confidence_interval_width = 1 - 0.90 if model_kwargs["alpha"] is None else 1 - model_kwargs[
                "alpha"]

        model_kwargs["interval_width"] = self.spec.confidence_interval_width
        return model_kwargs

    def preprocess(self, data, series_id):
        self.le[series_id], df_encoded = _label_encode_dataframe(
            data,
            no_encode={self.spec.datetime_column.name, self.original_target_column},
        )
        return df_encoded.set_index(self.spec.datetime_column.name)

    def _auto_detect_ets_params(self, y: pd.Series, seasonal_period: int) -> Dict[str, Any]:
        """Detect trend, error type, and damping from data."""
        params = {}

        # Detect trend via linear regression significance
        slope, _, r_value, p_value, _ = linregress(range(len(y)), y.values)
        has_trend = p_value < 0.05
        params["trend"] = "add" if has_trend else None
        params["damped_trend"] = (has_trend and abs(r_value) < 0.7)  # weak trend → damp

        # Detect additive vs multiplicative: does variance scale with level?
        if seasonal_period and len(y) >= 2 * seasonal_period:
            segments = [y.iloc[i:i + seasonal_period] for i in range(0, len(y) - seasonal_period, seasonal_period)]
            means = np.array([s.mean() for s in segments])
            stds = np.array([s.std() for s in segments])
            if means.std() > 0:
                corr = np.corrcoef(means, stds)[0, 1]
                params["error"] = "mul" if corr > 0.7 else "add"
                params["seasonal"] = params["error"]  # match seasonal to error type
            else:
                params["error"] = "add"
                params["seasonal"] = "add"
        else:
            params["error"] = "add"
            params["seasonal"] = "add"

        # Multiplicative requires strictly positive data
        if (y <= 0).any():
            params["error"] = "add"
            params["seasonal"] = "add"

        return params

    def _train_model(self, i, series_id, df: pd.DataFrame, model_kwargs: Dict[str, Any]):
        try:
            self.forecast_output.init_series_output(series_id=series_id, data_at_series=df)
            data = self.preprocess(df, series_id)
            data_i = self.drop_horizon(data)
            freq = self.datasets.get_datetime_frequency() if self.datasets.get_datetime_frequency() is not None else pd.infer_freq(
                data_i.index)

            Y = data_i[self.spec.target_column]
            dates = data_i.index.values
            is_daily = False
            if freq is not None:
                try:
                    is_daily = to_offset(freq).name in ["D", "B"]
                except ValueError:
                    pass

            ets_model_args = self.spec.model_kwargs.get("restrict_daily_series_improvement_flow", True)
            sp, probable_sps = find_seasonal_period_from_dataset(Y)
            if model_kwargs["seasonal"] is None:
                model_kwargs["seasonal"] = "add"

            if ets_model_args and is_daily:
                model_kwargs["seasonal_periods"] = sp if sp > 1 else None
            else:
                auto_params = self._auto_detect_ets_params(Y, sp)
                for k, v in auto_params.items():
                    model_kwargs[k] = v

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

            if (model_kwargs["error"] == "mul" or
                model_kwargs["trend"] == "mul" or
                model_kwargs["seasonal"] == "mul") and (Y <= 0).any():

                logger.info("Multiplicative model incompatible with non-positive values. Switching to additive.")
                model_kwargs["error"] = "add"
                if model_kwargs["trend"] == "mul":
                    model_kwargs["trend"] = "add"
                if model_kwargs["seasonal"] == "mul":
                    model_kwargs["seasonal"] = "add"

            use_seasonal = (
                    model_kwargs["seasonal"] is not None and
                    model_kwargs["seasonal_periods"] is not None and
                    model_kwargs["seasonal_periods"] > 1 and
                    len(Y) >= max(2 * model_kwargs["seasonal_periods"], 10)
            )
            if not use_seasonal:
                model_kwargs["seasonal"] = None
                model_kwargs["seasonal_periods"] = None
            model = ETSModel(Y, error=model_kwargs["error"], trend=model_kwargs["trend"],
                             damped_trend=model_kwargs["damped_trend"], seasonal=model_kwargs["seasonal"],
                             seasonal_periods=model_kwargs["seasonal_periods"],
                             dates=dates,
                             freq=freq,
                             initialization_method=model_kwargs["initialization_method"],
                             initial_level=model_kwargs.get("initial_level", None),
                             initial_trend=model_kwargs.get("initial_trend", None),
                             initial_seasonal=model_kwargs.get("initial_seasonal", None), )
            fit = model.fit()
            fitted_values = fit.fittedvalues
            forecast_values = fit.forecast(self.spec.horizon)
            f1 = fit.get_prediction(start=len(Y), end=len(Y) + self.spec.horizon - 1)
            forecast_bounds = f1.summary_frame(alpha=1 - self.spec.confidence_interval_width)

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
                "framework": SupportedModels.ETSForecaster,
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

        tsp, probable_sps = find_seasonal_period_from_dataset(y)

        def objective(trial):

            error = trial.suggest_categorical("error", ["add", "mul"])
            trend = trial.suggest_categorical("trend", ["add", "mul", None])
            damped_trend = trial.suggest_categorical("damped_trend", [True, False])
            sp = trial.suggest_categorical("seasonal_periods", probable_sps)
            initialization_method = trial.suggest_categorical(
                "initialization_method", ["estimated", "heuristic"]
            )
            seasonal = trial.suggest_categorical("seasonal", ["add", "mul", None])

            if (error == "mul" or trend == "mul" or seasonal == "mul") and (y <= 0).any():
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
            dates = y.index.values

            for train_idx, test_idx in cv.split(y):

                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                if (
                        seasonal is not None and sp is not None
                        and len(y_train) < 2 * sp
                ):
                    raise optuna.exceptions.TrialPruned()

                try:
                    model = ETSModel(
                        y_train,
                        error=error,
                        trend=trend,
                        damped_trend=damped_trend,
                        seasonal=seasonal,
                        seasonal_periods=sp,
                        dates=dates,
                        freq=self.datasets.get_datetime_frequency(),
                        initialization_method=initialization_method,
                        initial_level=model_kwargs_i.get("initial_level"),
                        initial_trend=model_kwargs_i.get("initial_trend"),
                        initial_seasonal=model_kwargs_i.get("initial_seasonal"),
                    )

                    fit = model.fit()
                    y_pred = fit.forecast(len(y_test))

                    metrics_df = _build_metrics_df(y_test, y_pred, 0)
                    metrics_dict = {
                        k.lower(): v
                        for k, v in metrics_df[0].to_dict().items()
                    }
                    if self.spec.metric.lower() not in metrics_dict:
                        scores.append(metrics_dict["mape"])
                    else:
                        scores.append(metrics_dict[self.spec.metric.lower()])

                except Exception:
                    continue
            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        trials = DEFAULT_TRIALS if self.spec.tuning.n_trials is None else self.spec.tuning.n_trials
        study.optimize(objective, n_trials=trials)
        completed_trials = [
            t for t in study.trials
            if t.state == TrialState.COMPLETE
        ]

        if not completed_trials:
            logger.debug(
                "ETS tuning produced no completed trials. "
                "Falling back to default parameters."
            )
            return model_kwargs_i

        model_kwargs_i.update({
            "error": study.best_params["error"],
            "trend": study.best_params["trend"],
            "damped_trend": study.best_params["damped_trend"],
            "seasonal": study.best_params["seasonal"],
            "seasonal_periods": study.best_params["seasonal_periods"],
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

                global_explanation_section, local_explanation_section = self.generate_explanation_report_from_data()

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
