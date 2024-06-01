#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import optuna
import pandas as pd
import logging
from joblib import Parallel, delayed
from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.opctl import logger
from ads.opctl.operator.lowcode.forecast.operator_config import ForecastOperatorConfig

from ..const import (
    DEFAULT_TRIALS,
    PROPHET_INTERNAL_DATE_COL,
    ForecastOutputColumns,
    SupportedModels,
)
from ads.opctl.operator.lowcode.forecast.utils import (
    _select_plot_list,
    _label_encode_dataframe,
)
from ads.opctl.operator.lowcode.common.utils import set_log_level
from .base_model import ForecastOperatorBaseModel
from ..operator_config import ForecastOperatorConfig
from .forecast_datasets import ForecastDatasets, ForecastOutput
import traceback
import matplotlib as mpl


try:
    set_log_level("prophet", logger.level)
    set_log_level("cmdstanpy", logger.level)
    mpl.rcParams["figure.max_open_warning"] = 100
except:
    pass


def _add_unit(num, unit):
    return f"{num} {unit}"


def _fit_model(data, params, additional_regressors):
    from prophet import Prophet

    model = Prophet(**params)
    for add_reg in additional_regressors:
        model.add_regressor(add_reg)
    model.fit(data)
    return model


class ProphetOperatorModel(ForecastOperatorBaseModel):
    """Class representing Prophet operator model."""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config=config, datasets=datasets)
        self.global_explanation = {}
        self.local_explanation = {}

    def set_kwargs(self):
        # Extract the Confidence Interval Width and convert to prophet's equivalent - interval_width
        if self.spec.confidence_interval_width is None:
            self.spec.confidence_interval_width = 1 - self.spec.model_kwargs.get(
                "alpha", 0.90
            )
        model_kwargs = self.spec.model_kwargs
        model_kwargs["interval_width"] = self.spec.confidence_interval_width
        return model_kwargs

    def _train_model(self, i, series_id, df, model_kwargs):
        try:
            from prophet import Prophet
            from prophet.diagnostics import cross_validation, performance_metrics

            self.forecast_output.init_series_output(
                series_id=series_id, data_at_series=df
            )

            data = self.preprocess(df, series_id)
            data_i = self.drop_horizon(data)
            if self.loaded_models is not None and series_id in self.loaded_models:
                model = self.loaded_models[series_id]
            else:
                if self.perform_tuning:
                    model_kwargs = self.run_tuning(data_i, model_kwargs)

                model = _fit_model(
                    data=data,
                    params=model_kwargs,
                    additional_regressors=self.additional_regressors,
                )

            # Get future df for prediction
            future = data.drop("y", axis=1)

            # Make Prediction
            forecast = model.predict(future)
            logger.debug(f"-----------------Model {i}----------------------")
            logger.debug(
                forecast[
                    [PROPHET_INTERNAL_DATE_COL, "yhat", "yhat_lower", "yhat_upper"]
                ].tail()
            )

            self.outputs[series_id] = forecast
            self.forecast_output.populate_series_output(
                series_id=series_id,
                fit_val=self.drop_horizon(forecast["yhat"]).values,
                forecast_val=self.get_horizon(forecast["yhat"]).values,
                upper_bound=self.get_horizon(forecast["yhat_upper"]).values,
                lower_bound=self.get_horizon(forecast["yhat_lower"]).values,
            )
            self.models[series_id] = model

            params = vars(model).copy()
            for param in ["history", "history_dates", "stan_fit"]:
                if param in params:
                    params.pop(param)
            self.model_parameters[series_id] = {
                "framework": SupportedModels.Prophet,
                **params,
            }

            logger.debug("===========Done===========")
        except Exception as e:
            self.errors_dict[series_id] = {
                "model_name": self.spec.model,
                "error": str(e),
            }
            logger.debug(f"Encountered Error: {e}. Skipping.")

    def _build_model(self) -> pd.DataFrame:
        full_data_dict = self.datasets.get_data_by_series()
        self.models = dict()
        self.outputs = dict()
        self.additional_regressors = self.datasets.get_additional_data_column_names()
        model_kwargs = self.set_kwargs()
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width,
            horizon=self.spec.horizon,
            target_column=self.original_target_column,
            dt_column=self.spec.datetime_column.name,
        )

        Parallel(n_jobs=-1, require="sharedmem")(
            delayed(ProphetOperatorModel._train_model)(
                self, i, series_id, df, model_kwargs.copy()
            )
            for self, (i, (series_id, df)) in zip(
                [self] * len(full_data_dict), enumerate(full_data_dict.items())
            )
        )

        return self.forecast_output.get_forecast_long()

    def run_tuning(self, data_i, model_kwargs_i):
        from prophet import Prophet
        from prophet.diagnostics import cross_validation, performance_metrics

        def objective(trial):
            params = {
                "seasonality_mode": trial.suggest_categorical(
                    "seasonality_mode", ["additive", "multiplicative"]
                ),
                "changepoint_prior_scale": trial.suggest_float(
                    "changepoint_prior_scale", 0.001, 0.5, log=True
                ),
                "seasonality_prior_scale": trial.suggest_float(
                    "seasonality_prior_scale", 0.01, 10, log=True
                ),
                "holidays_prior_scale": trial.suggest_float(
                    "holidays_prior_scale", 0.01, 10, log=True
                ),
                "changepoint_range": trial.suggest_float(
                    "changepoint_range", 0.8, 0.95
                ),
            }
            params.update(model_kwargs_i)

            model = _fit_model(
                data=data_i,
                params=params,
                additional_regressors=self.additional_regressors,
            )

            # Manual workaround because pandas 1.x dropped support for M and Y
            interval = self.spec.horizon
            freq = self.datasets.get_datetime_frequency()
            unit = freq.split("-")[0] if freq else None
            if unit == "M":
                unit = "D"
                interval = interval * 30.5
            elif unit == "Y":
                unit = "D"
                interval = interval * 365.25
            horizon = _add_unit(int(self.spec.horizon * interval), unit=unit)
            initial = _add_unit((data_i.shape[0] * interval) // 2, unit=unit)
            period = _add_unit((data_i.shape[0] * interval) // 4, unit=unit)

            logger.debug(
                f"using: horizon: {horizon}. initial:{initial}, period: {period}"
            )

            df_cv = cross_validation(
                model,
                horizon=horizon,
                initial=initial,
                period=period,
                parallel="threads",
            )
            df_p = performance_metrics(df_cv)
            try:
                return np.mean(df_p[self.spec.metric])
            except KeyError:
                logger.warn(
                    f"Could not find the metric {self.spec.metric} within "
                    f"the performance metrics: {df_p.columns}. Defaulting to `rmse`"
                )
                return np.mean(df_p["rmse"])

        study = optuna.create_study(direction="minimize")
        m_temp = Prophet()
        study.enqueue_trial(
            {
                "seasonality_mode": m_temp.seasonality_mode,
                "changepoint_prior_scale": m_temp.changepoint_prior_scale,
                "seasonality_prior_scale": m_temp.seasonality_prior_scale,
                "holidays_prior_scale": m_temp.holidays_prior_scale,
                "changepoint_range": m_temp.changepoint_range,
            }
        )
        study.optimize(
            objective,
            n_trials=self.spec.tuning.n_trials if self.spec.tuning else DEFAULT_TRIALS,
            n_jobs=-1,
        )

        study.best_params.update(model_kwargs_i)
        model_kwargs_i = study.best_params
        return model_kwargs_i

    def _generate_report(self):
        import report_creator as rc
        from prophet.plot import add_changepoints_to_plot

        series_ids = self.models.keys()
        all_sections = []
        if len(series_ids) > 0:
            sec1 = _select_plot_list(
                lambda s_id: self.models[s_id].plot(
                    self.outputs[s_id], include_legend=True
                ),
                series_ids=series_ids,
            )
            section_1 = rc.Block(
                rc.Heading("Forecast Overview", level=2),
                rc.Text(
                    "These plots show your forecast in the context of historical data."
                ),
                sec1,
            )

            sec2 = _select_plot_list(
                lambda s_id: self.models[s_id].plot_components(self.outputs[s_id]),
                series_ids=series_ids,
            )
            section_2 = rc.Block(
                rc.Heading("Forecast Broken Down by Trend Component", level=2), sec2
            )

            sec3_figs = {
                s_id: self.models[s_id].plot(self.outputs[s_id]) for s_id in series_ids
            }
            for s_id in series_ids:
                add_changepoints_to_plot(
                    sec3_figs[s_id].gca(), self.models[s_id], self.outputs[s_id]
                )
            sec3 = _select_plot_list(
                lambda s_id: sec3_figs[s_id], series_ids=series_ids
            )
            section_3 = rc.Block(rc.Heading("Forecast Changepoints", level=2), sec3)

            all_sections = [section_1, section_2, section_3]

            sec5_text = rc.Heading("Prophet Model Seasonality Components", level=2)
            model_states = []
            for s_id in series_ids:
                m = self.models[s_id]
                model_states.append(
                    pd.Series(
                        m.seasonalities,
                        index=pd.Index(m.seasonalities.keys(), dtype="object"),
                        name=s_id,
                        dtype="object",
                    )
                )
            all_model_states = pd.concat(model_states, axis=1)
            if not all_model_states.empty:
                sec5 = rc.DataTable(all_model_states, index=True)
                all_sections = all_sections + [sec5_text, sec5]

        if self.spec.generate_explanations:
            try:
                # If the key is present, call the "explain_model" method
                self.explain_model()

                # Convert the global explanation data to a DataFrame
                global_explanation_df = pd.DataFrame(self.global_explanation)

                self.formatted_global_explanation = (
                    global_explanation_df / global_explanation_df.sum(axis=0) * 100
                )

                # Create a markdown section for the global explainability
                global_explanation_section = rc.Block(
                    rc.Heading("Global Explanation of Models", level=2),
                    rc.Text(
                        "The following tables provide the feature attribution for the global explainability."
                    ),
                    rc.DataTable(self.formatted_global_explanation, index=True),
                )

                aggregate_local_explanations = pd.DataFrame()
                for s_id, local_ex_df in self.local_explanation.items():
                    local_ex_df_copy = local_ex_df.copy()
                    local_ex_df_copy[ForecastOutputColumns.SERIES] = s_id
                    aggregate_local_explanations = pd.concat(
                        [aggregate_local_explanations, local_ex_df_copy], axis=0
                    )
                self.formatted_local_explanation = aggregate_local_explanations

                blocks = [
                    rc.DataTable(
                        local_ex_df.div(local_ex_df.abs().sum(axis=1), axis=0) * 100,
                        label=s_id,
                        index=True,
                    )
                    for s_id, local_ex_df in self.local_explanation.items()
                ]
                local_explanation_section = rc.Block(
                    rc.Heading("Local Explanation of Models", level=2),
                    rc.Select(blocks=blocks),
                )

                # Append the global explanation text and section to the "all_sections" list
                all_sections = all_sections + [
                    global_explanation_section,
                    local_explanation_text,
                    local_explanation_section,
                ]
            except Exception as e:
                # Do not fail the whole run due to explanations failure
                logger.warn(f"Failed to generate Explanations with error: {e}.")
                logger.debug(f"Full Traceback: {traceback.format_exc()}")

        model_description = (
            "Prophet is a procedure for forecasting time series data based on an additive "
            "model where non-linear trends are fit with yearly, weekly, and daily seasonality, "
            "plus holiday effects. It works best with time series that have strong seasonal "
            "effects and several seasons of historical data. Prophet is robust to missing "
            "data and shifts in the trend, and typically handles outliers well."
        )
        other_sections = all_sections

        return (
            model_description,
            other_sections,
        )
