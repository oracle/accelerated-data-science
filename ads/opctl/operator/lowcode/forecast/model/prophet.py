#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import optuna
import pandas as pd
from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.opctl import logger
from ads.opctl.operator.lowcode.forecast.operator_config import ForecastOperatorConfig

from ..const import DEFAULT_TRIALS, PROPHET_INTERNAL_DATE_COL, ForecastOutputColumns
from .. import utils
from .base_model import ForecastOperatorBaseModel
from ..operator_config import ForecastOperatorConfig
from .forecast_datasets import ForecastDatasets, ForecastOutput
import traceback
import matplotlib as mpl

mpl.rcParams["figure.max_open_warning"] = 100


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
        self.train_metrics = True
        self.global_explanation = {}
        self.local_explanation = {}

    def _build_model(self) -> pd.DataFrame:
        from prophet import Prophet
        from prophet.diagnostics import cross_validation, performance_metrics

        full_data_dict = self.datasets.full_data_dict
        models = []
        outputs = dict()
        outputs_legacy = []

        # Extract the Confidence Interval Width and convert to prophet's equivalent - interval_width
        if self.spec.confidence_interval_width is None:
            self.spec.confidence_interval_width = 1 - self.spec.model_kwargs.get(
                "alpha", 0.90
            )

        model_kwargs = self.spec.model_kwargs
        model_kwargs["interval_width"] = self.spec.confidence_interval_width

        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width
        )

        for i, (target, df) in enumerate(full_data_dict.items()):
            le, df_encoded = utils._label_encode_dataframe(
                df, no_encode={self.spec.datetime_column.name, target}
            )

            model_kwargs_i = model_kwargs.copy()
            # format the dataframe for this target. Dropping NA on target[df] will remove all future data
            df_clean = self._preprocess(
                df_encoded,
                self.spec.datetime_column.name,
                self.spec.datetime_column.format,
            )
            data_i = df_clean[df_clean[target].notna()]
            data_i.rename({target: "y"}, axis=1, inplace=True)

            # Assume that all columns passed in should be used as additional data
            additional_regressors = set(data_i.columns) - {
                "y",
                PROPHET_INTERNAL_DATE_COL,
            }

            if self.perform_tuning:

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
                        additional_regressors=additional_regressors,
                    )

                    # Manual workaround because pandas 1.x dropped support for M and Y
                    interval = self.spec.horizon.interval
                    unit = self.spec.horizon.interval_unit
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
                    n_trials=self.spec.tuning.n_trials
                    if self.spec.tuning
                    else DEFAULT_TRIALS,
                    n_jobs=-1,
                )

                study.best_params.update(model_kwargs_i)
                model_kwargs_i = study.best_params
            model = _fit_model(
                data=data_i,
                params=model_kwargs_i,
                additional_regressors=additional_regressors,
            )

            # Make future df for prediction
            if len(additional_regressors):
                future = df_clean.drop(target, axis=1)
            else:
                future = model.make_future_dataframe(
                    periods=self.spec.horizon,
                    freq=self.spec.freq,
                )
            # Make Prediction
            forecast = model.predict(future)
            logger.debug(f"-----------------Model {i}----------------------")
            logger.debug(
                forecast[
                    [PROPHET_INTERNAL_DATE_COL, "yhat", "yhat_lower", "yhat_upper"]
                ].tail()
            )

            # Collect Outputs
            models.append(model)
            outputs[target] = forecast
            outputs_legacy.append(forecast)

        self.models = models
        self.outputs = outputs_legacy

        logger.debug("===========Done===========")

        # Merge the outputs from each model into 1 df with all outputs by target and category
        col = self.original_target_column
        output_col = pd.DataFrame()
        yhat_upper_name = ForecastOutputColumns.UPPER_BOUND
        yhat_lower_name = ForecastOutputColumns.LOWER_BOUND
        for cat in self.categories:
            output_i = pd.DataFrame()

            output_i["Date"] = outputs[f"{col}_{cat}"][PROPHET_INTERNAL_DATE_COL]
            output_i["Series"] = cat
            output_i["input_value"] = full_data_dict[f"{col}_{cat}"][f"{col}_{cat}"]

            output_i[f"fitted_value"] = float("nan")
            output_i[f"forecast_value"] = float("nan")
            output_i[yhat_upper_name] = float("nan")
            output_i[yhat_lower_name] = float("nan")

            output_i.iloc[
                : -self.spec.horizon, output_i.columns.get_loc(f"fitted_value")
            ] = (outputs[f"{col}_{cat}"]["yhat"].iloc[: -self.spec.horizon].values)
            output_i.iloc[
                -self.spec.horizon :,
                output_i.columns.get_loc(f"forecast_value"),
            ] = (
                outputs[f"{col}_{cat}"]["yhat"].iloc[-self.spec.horizon :].values
            )
            output_i.iloc[
                -self.spec.horizon :, output_i.columns.get_loc(yhat_upper_name)
            ] = (
                outputs[f"{col}_{cat}"]["yhat_upper"].iloc[-self.spec.horizon :].values
            )
            output_i.iloc[
                -self.spec.horizon :, output_i.columns.get_loc(yhat_lower_name)
            ] = (
                outputs[f"{col}_{cat}"]["yhat_lower"].iloc[-self.spec.horizon :].values
            )
            output_col = pd.concat([output_col, output_i])
            self.forecast_output.add_category(
                category=cat, target_category_column=f"{col}_{cat}", forecast=output_i
            )

        output_col = output_col.reset_index(drop=True)

        return output_col

    def _generate_report(self):
        import datapane as dp
        from prophet.plot import add_changepoints_to_plot

        sec1_text = dp.Text(
            "## Forecast Overview \n"
            "These plots show your forecast in the context of historical data."
        )
        sec1 = utils._select_plot_list(
            lambda idx, *args: self.models[idx].plot(
                self.outputs[idx], include_legend=True
            ),
            target_columns=self.target_columns,
        )

        sec2_text = dp.Text(f"## Forecast Broken Down by Trend Component")
        sec2 = utils._select_plot_list(
            lambda idx, *args: self.models[idx].plot_components(self.outputs[idx]),
            target_columns=self.target_columns,
        )

        sec3_text = dp.Text(f"## Forecast Changepoints")
        sec3_figs = [
            self.models[idx].plot(self.outputs[idx])
            for idx in range(len(self.target_columns))
        ]
        [
            add_changepoints_to_plot(
                sec3_figs[idx].gca(), self.models[idx], self.outputs[idx]
            )
            for idx in range(len(self.target_columns))
        ]
        sec3 = utils._select_plot_list(
            lambda idx, *args: sec3_figs[idx], target_columns=self.target_columns
        )

        all_sections = [sec1_text, sec1, sec2_text, sec2, sec3_text, sec3]

        sec5_text = dp.Text(f"## Prophet Model Seasonality Components")
        model_states = []
        for i, m in enumerate(self.models):
            model_states.append(
                pd.Series(
                    m.seasonalities,
                    index=pd.Index(m.seasonalities.keys(), dtype="object"),
                    name=self.target_columns[i],
                    dtype="object",
                )
            )
        all_model_states = pd.concat(model_states, axis=1)
        if not all_model_states.empty:
            sec5 = dp.DataTable(all_model_states)
            all_sections = all_sections + [sec5_text, sec5]

        if self.spec.generate_explanations:
            try:
                # If the key is present, call the "explain_model" method
                self.explain_model(
                    datetime_col_name=PROPHET_INTERNAL_DATE_COL,
                    explain_predict_fn=self._custom_predict_prophet,
                )

                # Create a markdown text block for the global explanation section
                global_explanation_text = dp.Text(
                    f"## Global Explanation of Models \n "
                    "The following tables provide the feature attribution for the global explainability."
                )

                # Convert the global explanation data to a DataFrame
                global_explanation_df = pd.DataFrame(self.global_explanation)

                self.formatted_global_explanation = (
                    global_explanation_df / global_explanation_df.sum(axis=0) * 100
                )

                # Create a markdown section for the global explainability
                global_explanation_section = dp.Blocks(
                    "### Global Explainability ",
                    dp.DataTable(self.formatted_global_explanation),
                )

                aggregate_local_explanations = pd.DataFrame()
                for s_id, local_ex_df in self.local_explanation.items():
                    local_ex_df_copy = local_ex_df.copy()
                    local_ex_df_copy["Series"] = s_id
                    aggregate_local_explanations = pd.concat(
                        [aggregate_local_explanations, local_ex_df_copy], axis=0
                    )
                self.formatted_local_explanation = aggregate_local_explanations

                local_explanation_text = dp.Text(f"## Local Explanation of Models \n ")
                blocks = [
                    dp.DataTable(
                        local_ex_df.div(local_ex_df.abs().sum(axis=1), axis=0) * 100,
                        label=s_id,
                    )
                    for s_id, local_ex_df in self.local_explanation.items()
                ]
                local_explanation_section = (
                    dp.Select(blocks=blocks) if len(blocks) > 1 else blocks[0]
                )

                # Append the global explanation text and section to the "all_sections" list
                all_sections = all_sections + [
                    global_explanation_text,
                    global_explanation_section,
                    local_explanation_text,
                    local_explanation_section,
                ]
            except Exception as e:
                # Do not fail the whole run due to explanations failure
                logger.warn(f"Failed to generate Explanations with error: {e}.")
                logger.debug(f"Full Traceback: {traceback.format_exc()}")

        model_description = dp.Text(
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

    def _custom_predict_prophet(self, data):
        return self.models[self.target_columns.index(self.series_id)].predict(
            data.reset_index()
        )["yhat"]
