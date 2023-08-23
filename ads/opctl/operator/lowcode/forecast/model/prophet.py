#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import datapane as dp
import numpy as np
import optuna
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot

from ads.opctl import logger

from .. import utils
from .base_model import ForecastOperatorBaseModel


def _add_unit(num, unit):
    return f"{num} {unit}"


def _fit_model(data, params, additional_regressors):
    model = Prophet(**params)
    for add_reg in additional_regressors:
        model.add_regressor(add_reg)
    model.fit(data)
    return model


class ProphetOperatorModel(ForecastOperatorBaseModel):
    """Class representing Prophet operator model."""

    def _build_model(self) -> pd.DataFrame:
        full_data_dict = self.full_data_dict
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
            additional_regressors = set(data_i.columns) - {"y", "ds"}

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
                    horizon = _add_unit(
                        int(self.spec.horizon.periods * interval), unit=unit
                    )
                    initial = _add_unit((data_i.shape[0] * interval) // 2, unit=unit)
                    period = _add_unit((data_i.shape[0] * interval) // 4, unit=unit)

                    logger.info(
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
                    n_trials=self.spec.tuning.n_trials if self.spec.tuning else 10,
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
                # TOOD: this will use the period/range of the additional data
                future = df_clean.drop(target, axis=1)
            else:
                future = model.make_future_dataframe(
                    periods=self.spec.horizon.periods,
                    freq=self.spec.horizon.interval_unit,
                )
            # Make Prediction
            forecast = model.predict(future)
            logger.info(f"-----------------Model {i}----------------------")
            logger.info(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

            # Collect Outputs
            models.append(model)
            outputs[target] = forecast
            outputs_legacy.append(forecast)

        self.models = models
        self.outputs = outputs_legacy

        logger.info("===========Done===========")
        outputs_merged = pd.DataFrame()

        # Merge the outputs from each model into 1 df with all outputs by target and category
        col = self.original_target_column
        output_col = pd.DataFrame()
        yhat_lower_percentage = (100 - model_kwargs["interval_width"] * 100) // 2
        yhat_upper_name = "p" + str(int(100 - yhat_lower_percentage))
        yhat_lower_name = "p" + str(int(yhat_lower_percentage))
        for cat in self.categories:  # Note: add [:2] to restrict
            output_i = pd.DataFrame()

            output_i["Date"] = outputs[f"{col}_{cat}"]["ds"]
            output_i["Series"] = cat
            output_i[f"forecast_value"] = outputs[f"{col}_{cat}"]["yhat"]
            output_i[yhat_upper_name] = outputs[f"{col}_{cat}"]["yhat_upper"]
            output_i[yhat_lower_name] = outputs[f"{col}_{cat}"]["yhat_lower"]
            output_col = pd.concat([output_col, output_i])
        # output_col = output_col.sort_values(self.spec.datetime_column.name).reset_index(drop=True)
        output_col = output_col.reset_index(drop=True)
        outputs_merged = pd.concat([outputs_merged, output_col], axis=1)

        # Re-merge historical data for processing
        data_merged = pd.concat(
            [v[v[k].notna()].set_index("ds") for k, v in full_data_dict.items()], axis=1
        ).reset_index()

        self.data = data_merged
        return outputs_merged

    def _generate_report(self):
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

        model_description = dp.Text(
            "Prophet is a procedure for forecasting time series data based on an additive "
            "model where non-linear trends are fit with yearly, weekly, and daily seasonality, "
            "plus holiday effects. It works best with time series that have strong seasonal "
            "effects and several seasons of historical data. Prophet is robust to missing "
            "data and shifts in the trend, and typically handles outliers well."
        )
        other_sections = all_sections
        forecast_col_name = "yhat"
        train_metrics = True
        ds_column_series = self.data["ds"]
        ds_forecast_col = self.outputs[0]["ds"]
        ci_col_names = ["yhat_lower", "yhat_upper"]

        return (
            model_description,
            other_sections,
            forecast_col_name,
            train_metrics,
            ds_column_series,
            ds_forecast_col,
            ci_col_names,
        )
