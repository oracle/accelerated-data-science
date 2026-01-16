#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

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
from sktime.split import ExpandingWindowSplitter
from sktime.transformations.series.detrend import Deseasonalizer

from ads.opctl import logger
from ads.opctl.operator.lowcode.common.utils import find_seasonal_period_from_dataset, normalize_frequency
from ads.opctl.operator.lowcode.forecast.operator_config import ForecastOperatorConfig
from ads.opctl.operator.lowcode.forecast.utils import (_label_encode_dataframe, _build_metrics_df)
from .base_model import ForecastOperatorBaseModel
from .forecast_datasets import ForecastDatasets, ForecastOutput
from .univariate_model import UnivariateForecasterOperatorModel
from ..const import (
    SupportedModels, DEFAULT_TRIALS,
)

logging.getLogger("report_creator").setLevel(logging.WARNING)


class ThetaOperatorModel(UnivariateForecasterOperatorModel):
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
        model_kwargs["deseasonalize_model"] = "mul"
        model_kwargs["sp"] = self.spec.model_kwargs.get("sp", None)

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

    def _train_model(self, i, series_id, df: pd.DataFrame, model_kwargs: Dict[str, Any]):
        try:
            self.forecast_output.init_series_output(series_id=series_id, data_at_series=df)
            data = self.preprocess(df, series_id)

            data_i = self.drop_horizon(data)
            target = self.spec.target_column

            freq = self.datasets.get_datetime_frequency() if self.datasets.get_datetime_frequency() is not None else pd.infer_freq(
                data_i.index)
            if freq is not None:
                normalized_freq = normalize_frequency(freq)
                data_i.index = data_i.index.to_period(normalized_freq)

            y = data_i[target]
            X_in = data_i.drop(target, axis=1)

            if model_kwargs["deseasonalize"] and model_kwargs["sp"] is None:
                sp, probable_sps = find_seasonal_period_from_dataset(y)
            else:
                sp, probable_sps = 1, [1]

            model_kwargs["sp"] = model_kwargs.get("sp") or sp

            if not sp or len(y) < 2 * model_kwargs["sp"]:
                model_kwargs["deseasonalize"] = False

            # If model already loaded, extract parameters (best-effort)
            if self.loaded_models is not None and series_id in self.loaded_models:
                previous_res = self.loaded_models[series_id].get("model")
                fitted_params = previous_res.get_fitted_params()
                model_kwargs["initial_level"] = fitted_params.get("initial_level", None)
            elif self.perform_tuning:
                model_kwargs = self.run_tuning(y, X_in, model_kwargs, probable_sps)

            # Fit ThetaModel using params
            using_additive_deseasonalization = False
            additive_deseasonalizer = None
            if model_kwargs["deseasonalize"]:
                if (y <= 0).any():
                    logger.warning(
                        "Processing data with additive deseasonalization model as data contains negative or zero values which can't be deseasonalized using multiplicative deseasonalization. And ThetaForecaster by default only supports multiplicative deseasonalization.")
                    model_kwargs["deseasonalize_model"] = "add"
                    using_additive_deseasonalization = True
                    additive_deseasonalizer = Deseasonalizer(
                        sp=model_kwargs["sp"],
                        model="additive",
                    )
                    y_adj = additive_deseasonalizer.fit_transform(y)
                    y = y_adj
                    model_kwargs["deseasonalize"] = False
            else:
                model_kwargs["deseasonalize_model"] = ""

            model = ThetaForecaster(initial_level=model_kwargs["initial_level"],
                                    deseasonalize=model_kwargs["deseasonalize"],
                                    sp=1 if model_kwargs["deseasonalize_model"] == "add" else model_kwargs.get("sp",
                                                                                                               1), )
            model.fit(y, X=X_in)

            fh = ForecastingHorizon(range(1, self.spec.horizon + 1), is_relative=True)
            fh_in_sample = ForecastingHorizon(range(-len(data_i) + 1, 1))
            fitted_vals = model.predict(fh_in_sample)
            forecast_values = model.predict(fh)
            forecast_range = model.predict_interval(fh=fh, coverage=self.spec.confidence_interval_width)

            if using_additive_deseasonalization and additive_deseasonalizer is not None:
                fitted_vals = additive_deseasonalizer.inverse_transform(fitted_vals)
                forecast_values = additive_deseasonalizer.inverse_transform(forecast_values)
                forecast_range_inv = forecast_range.copy()
                for col in forecast_range.columns:
                    forecast_range_inv[col] = additive_deseasonalizer.inverse_transform(
                        forecast_range[[col]]
                    )[col]
                forecast_range = forecast_range_inv

            lower = forecast_range[(self.original_target_column, self.spec.confidence_interval_width, "lower")].rename(
                "yhat_lower")
            upper = forecast_range[(self.original_target_column, self.spec.confidence_interval_width, "upper")].rename(
                "yhat_upper")
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
            self.models[series_id]["model_params"] = model_kwargs
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
            logger.warning(f"Encountered Error: {e}. Skipping.")
            logger.warning(traceback.format_exc())

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

    def run_tuning(self, y: pd.DataFrame, X: pd.DataFrame | None, model_kwargs_i: Dict[str, Any],
                   probable_sps: list[int]):

        def objective(trial):
            y_used = y
            X_used = X

            initial_level = model_kwargs_i["initial_level"]
            sp = trial.suggest_categorical("sp", probable_sps)
            deseasonalize = trial.suggest_categorical("deseasonalize", [True, False])
            deseasonalize_model = trial.suggest_categorical(
                "deseasonalize_model", ["add", "mul"]
            )

            if deseasonalize and deseasonalize_model == "mul" and (y_used <= 0).any():
                raise optuna.exceptions.TrialPruned()
            d_sp, d_deseasonalize = sp, deseasonalize
            if deseasonalize and deseasonalize_model == "add":
                additive_deseasonalizer = Deseasonalizer(
                    sp=sp,
                    model="additive",
                )
                y_used = additive_deseasonalizer.fit_transform(y_used)
                d_sp = 1
                d_deseasonalize = False

            model = ThetaForecaster(
                initial_level=initial_level,
                sp=d_sp,
                deseasonalize=d_deseasonalize,
            )

            cv = ExpandingWindowSplitter(
                initial_window=50,
                step_length=100
            )

            scores = []

            for train, test in cv.split(y_used):
                y_train = y_used.iloc[train]
                y_test = y.iloc[test]
                if y_train.isna().any():
                    continue
                if len(y_train) < 2 * sp:
                    continue

                X_train = None
                X_test = None

                if X_used is not None:
                    X_train = X_used.iloc[train]
                    X_test = X_used.iloc[test]

                model.fit(y_train, X=X_train)
                fh = ForecastingHorizon(y.index[test], is_relative=False)
                y_pred = model.predict(fh, X=X_test)
                if y_test.isna().any():
                    continue
                metrics_df = _build_metrics_df(y_test, y_pred, 0)
                metrics_dict = {
                    k.lower(): v
                    for k, v in metrics_df[0].to_dict().items()
                }
                if self.spec.metric.lower() not in metrics_dict:
                    scores.append(metrics_dict["mape"])
                else:
                    scores.append(metrics_dict[self.spec.metric.lower()])

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
                "Theta tuning produced no completed trials. "
                "Falling back to default parameters."
            )
            return model_kwargs_i

        model_kwargs_i["deseasonalize_model"] = study.best_params["deseasonalize_model"]
        model_kwargs_i["deseasonalize"] = study.best_params["deseasonalize"]
        model_kwargs_i["sp"] = study.best_params["sp"]
        return model_kwargs_i

    def _generate_report(self):
        import report_creator as rc
        """The method that needs to be implemented on the particular model level."""
        all_sections = []
        theta_blocks = []

        for series_id, sm in self.models.items():
            model = sm["model"]
            model_kwargs = sm["model_params"]

            fitted_params = model.get_fitted_params()
            initial_level = fitted_params.get("initial_level", None)
            smoothing_level = fitted_params.get("smoothing_level", None)
            sp = model_kwargs.get("sp", 1)
            deseasonalize_model = model_kwargs.get("deseasonalize_model", "mul")
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
                    "Smoothing Level",
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
                    desasonalized,
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

                global_explanation_section, local_explanation_section = self.generate_explanation_report_from_data()

                # Append the global explanation text and section to the "all_sections" list
                all_sections = all_sections + [
                    global_explanation_section,
                    local_explanation_section,
                ]

            except Exception as e:
                logger.warning(f"Failed to generate Explanations with error: {e}.")
                logger.warning(f"Full Traceback: {traceback.format_exc()}")

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
