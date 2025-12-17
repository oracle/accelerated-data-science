#!/usr/bin/env python

import logging
import traceback
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from optuna.trial import TrialState
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import mean_squared_error, \
    mean_absolute_percentage_error
from sktime.split import ExpandingWindowSplitter

from ads.opctl import logger
from ads.opctl.operator.lowcode.forecast.operator_config import ForecastOperatorConfig
from ads.opctl.operator.lowcode.forecast.utils import (_label_encode_dataframe, smape)
from .base_model import ForecastOperatorBaseModel
from .forecast_datasets import ForecastDatasets, ForecastOutput
from ..const import (
    SupportedModels, ForecastOutputColumns, DEFAULT_TRIALS,
)
from ads.opctl.operator.lowcode.common.utils import find_seasonal_period_from_dataset

logging.getLogger("report_creator").setLevel(logging.WARNING)


class ThetaOperatorModel(ForecastOperatorBaseModel):
    """Theta operator model"""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config=config, datasets=datasets)
        self.global_explanation = {}
        self.local_explanation = {}

    def set_kwargs(self):
        """Prepare kwargs for Theta model from spec.
           The operator's 'model_kwargs' is respected.
        """
        model_kwargs = self.spec.model_kwargs
        model_kwargs["alpha"] = self.spec.model_kwargs.get("alpha", None)
        model_kwargs["initial_level"] = self.spec.model_kwargs.get("initial_level", None)
        model_kwargs["deseasonalize"] = self.spec.model_kwargs.get("deseasonalize", True)
        model_kwargs["deseasonalize_model"] = self.spec.model_kwargs.get("deseasonalize_model", "additive")
        model_kwargs["sp"] = self.spec.model_kwargs.get("sp", None)

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
            target = self.spec.target_column

            freq = self.datasets.get_datetime_frequency()
            if freq is not None:
                if freq.startswith("W-"):
                    freq = "W"
                data_i.index = data_i.index.to_period(freq)

            y = data_i[target]
            sp, probable_sps = find_seasonal_period_from_dataset(y)

            model_kwargs["sp"] = model_kwargs.get("sp") or sp

            if not sp or len(y) < 2 * model_kwargs["sp"]:
                model_kwargs["deseasonalize"] = False

            # If model already loaded, extract parameters (best-effort)
            if self.loaded_models is not None and series_id in self.loaded_models:
                previous_res = self.loaded_models[series_id].get("model")
                fitted_params = previous_res.get_fitted_params()
                model_kwargs["deseasonalize_model"] = previous_res.deseasonalize_model
                model_kwargs["sp"] = previous_res.sp
                model_kwargs["deseasonalize"] = previous_res.deseasonalize
                model_kwargs["initial_level"] = fitted_params.get("initial_level", None)
            elif self.perform_tuning:
                model_kwargs = self.run_tuning(y, model_kwargs, probable_sps)

            model = ThetaForecaster(initial_level=model_kwargs["initial_level"],
                                    deseasonalize=model_kwargs["deseasonalize"],
                                    deseasonalize_model=model_kwargs["deseasonalize_model"],
                                    sp=model_kwargs.get("sp", 1), )
            model.fit(y)

            fh = ForecastingHorizon(range(1, self.spec.horizon + 1), is_relative=True)
            fh_in_sample = ForecastingHorizon(range(-len(data_i) + 1, 1))
            fitted_vals = model.predict(fh_in_sample)
            forecast_values = model.predict(fh)
            forecast_range = model.predict_interval(fh=fh)

            lower = forecast_range[(self.original_target_column, 0.9, "lower")].rename("yhat_lower")
            upper = forecast_range[(self.original_target_column, 0.9, "upper")].rename("yhat_upper")
            point = forecast_values.rename("yhat")
            forecast = pd.DataFrame(
                pd.concat([point, lower, upper], axis=1)
            )
            logger.debug(f"-----------------Model {i}----------------------")
            logger.debug(forecast[["yhat", "yhat_lower", "yhat_upper"]].tail())

            self.forecast_output.populate_series_output(
                series_id=series_id,
                fit_val=fitted_vals.values,
                forecast_val=forecast["yhat"].values,
                upper_bound=forecast["yhat_upper"].values,
                lower_bound=forecast["yhat_lower"].values,
            )
            self.outputs[series_id] = forecast
            self.models[series_id] = {}
            self.models[series_id]["model"] = model
            self.models[series_id]["le"] = self.le[series_id]

            params = vars(model).copy()
            self.model_parameters[series_id] = {
                "framework": SupportedModels.Theta,
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
            delayed(ThetaOperatorModel._train_model)(
                self, i, series_id, df, model_kwargs.copy()
            )
            for self, (i, (series_id, df)) in zip(
                [self] * len(full_data_dict), enumerate(full_data_dict.items())
            )
        )

        return self.forecast_output.get_forecast_long()

    def run_tuning(self, y: pd.DataFrame, model_kwargs_i: Dict[str, Any], probable_sps: list[int]):

        scoring = {
            "mape": lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred),
            "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            "mse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
            "smape": lambda y_true, y_pred: smape(y_true, y_pred)
        }
        score_fn = scoring.get(self.spec.metric.lower(), scoring["mape"])

        def objective(trial):
            initial_level = model_kwargs_i["initial_level"]
            sp = trial.suggest_categorical("sp", probable_sps)
            deseasonalize = trial.suggest_categorical("deseasonalize", [True, False])
            deseasonalize_model = trial.suggest_categorical("deseasonalize_model", ["additive", "multiplicative"])
            if deseasonalize_model == "multiplicative" and (y <= 0).any():
                raise optuna.exceptions.TrialPruned()

            model = ThetaForecaster(
                initial_level=initial_level,
                sp=sp,
                deseasonalize_model=deseasonalize_model,
                deseasonalize=deseasonalize,
            )

            cv = ExpandingWindowSplitter(
                initial_window=50,
                step_length=100
            )

            scores = []

            for train, test in cv.split(y):
                t_data = y.iloc[train]
                if t_data.isna().any():
                    continue
                if len(t_data) < 2 * sp:
                    continue

                model.fit(t_data)
                fh = ForecastingHorizon(y.index[test], is_relative=False)
                y_pred = model.predict(fh)
                y_test = y.iloc[test]
                if y_test.isna().any():
                    continue
                scores.append(score_fn(y_test, y_pred))
            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        trials = DEFAULT_TRIALS if self.spec.tuning.n_trials is None else self.spec.tuning.n_trials
        study.optimize(objective, n_trials=trials)
        completed_trials = [
            t for t in study.trials
            if t.state == TrialState.COMPLETE
        ]

        if not completed_trials:
            logger.warning(
                "Theta tuning produced no completed trials. "
                "Falling back to default parameters."
            )
            return model_kwargs_i

        model_kwargs_i["deseasonalize_model"] = study.best_params["deseasonalize_model"]
        model_kwargs_i["deseasonalize"] = study.best_params["deseasonalize"]
        return model_kwargs_i

    def _generate_report(self):
        import report_creator as rc
        """The method that needs to be implemented on the particular model level."""
        all_sections = []
        theta_blocks = []

        for series_id, sm in self.models.items():
            model = sm["model"]

            # ---- Extract details from ThetaModel ----
            fitted_params = model.get_fitted_params()
            initial_level = fitted_params.get("initial_level", None)
            smoothing_level = fitted_params.get("smoothing_level", None)
            sp = model.sp
            deseasonalize_model = model.deseasonalize_model
            desasonalized = model.deseasonalize
            n_obs = len(model._y) if hasattr(model, "_y") else "N/A"

            # Date range
            if hasattr(model, "_y"):
                start_date = model._y.index[0]
                end_date = model._y.index[-1]
            else:
                start_date = ""
                end_date = ""

            # ---- Build the DF ----
            meta_df = pd.DataFrame({
                "Metric": [
                    "Initial Level",
                    "Smoothing Level / Alpha",
                    "No. Observations",
                    "Deseasonalized",
                    "Deseasonalization Method",
                    "Period (sp)",
                    "Sample Start",
                    "Sample End",
                ],
                "Value": [
                    initial_level,
                    smoothing_level,
                    n_obs,
                    str(desasonalized is not None),
                    deseasonalize_model,
                    sp,
                    start_date,
                    end_date,
                ],
            })

            # ---- Create a block (NOT a section directly) ----
            theta_block = rc.Block(
                rc.Heading(f"Theta Model Summary", level=3),
                rc.DataTable(meta_df),
                label=series_id
            )

            # Add with optional label support
            theta_blocks.append(
                theta_block
            )

        # ---- Combine into final section like ARIMA example ----
        theta_title = rc.Heading("Theta Model Parameters", level=2)
        theta_section = []
        if len(theta_blocks) > 1:
            theta_section = rc.Select(blocks=theta_blocks)
        elif len(theta_blocks) == 1:
            theta_section = theta_blocks[0]
        else:
            theta_section = rc.Text("No Theta models were successfully trained.")

        all_sections.extend([theta_title, theta_section])

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
            "A Theta forecaster is a popular and surprisingly effective time series forecasting"
            "method that works by decomposing data into long-term trend and short-term components, forecasting them separately,"
            "and then combining the results, often outperforming complex models by adjusting the original series' local"
            "curvature using a parameter called theta (θ). It's known for its simplicity, speed, and strong performance, "
            "especially in forecasting competitions like the M3, where it served as a strong benchmark, often by using"
            "Simple Exponential Smoothing (SES) with drift on a modified series"
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
            h = len(data)
            fh = ForecastingHorizon(np.arange(1, h + 1), is_relative=True)
            return model.predict(fh)

        return _custom_predict
