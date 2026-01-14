#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import traceback

import pandas as pd

from ads.common.decorator import runtime_dependency
from ads.opctl import logger
from .forecast_datasets import ForecastDatasets
from .ml_forecast import MLForecastBaseModel
from ..const import ForecastOutputColumns, SupportedModels
from ..operator_config import ForecastOperatorConfig


class XGBForecastOperatorModel(MLForecastBaseModel):
    """Class representing MLForecast operator model."""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config=config, datasets=datasets)
        self.model_name = "XGBForecast"
        self.model_description = """XGBoost (XGB) for forecasting is a gradient-boosted decision tree model that predicts 
        future values by learning nonlinear patterns from lagged features and exogenous variables. It excels at 
        capturing complex relationships and interactions but requires careful feature engineering for time-series data."""

    def get_model_kwargs(self):
        """
        Returns the model parameters.
        """
        return self.spec.model_kwargs

    @runtime_dependency(
        module="mlforecast",
        err_msg="MLForecast is not installed, please install it with 'pip install mlforecast'",
    )
    @runtime_dependency(
        module="xgboost",
        err_msg="xgboost is not installed, please install it with 'pip install xgboost'",
    )
    def _train_model(self, data_train, data_test, model_kwargs):
        try:
            xgb_params = {
                "verbosity": model_kwargs.get("verbosity", 0),
                "num_leaves": model_kwargs.get("num_leaves", 512),
            }
            from xgboost import XGBRegressor
            from mlforecast import MLForecast
            from mlforecast.utils import PredictionIntervals
            level = int(self.spec.confidence_interval_width * 100)
            data_freq = self.datasets.get_datetime_frequency()
            additional_data_params = self.set_model_config(data_freq, model_kwargs)
            model = XGBRegressor(**xgb_params)

            prediction_intervals = PredictionIntervals(
                n_windows=5,
                method="conformal_distribution",
            )

            fcst = MLForecast(
                models={"forecast": model},
                freq=data_freq,
                date_features=['year', 'month', 'day', 'dayofweek', 'dayofyear'],
                **additional_data_params,
            )

            num_models = model_kwargs.get("recursive_models", False)

            model_columns = [
                                ForecastOutputColumns.SERIES
                            ] + data_train.select_dtypes(exclude=["object"]).columns.to_list()

            fcst.fit(
                data_train[model_columns],
                prediction_intervals=prediction_intervals,
                id_col=ForecastOutputColumns.SERIES,
                time_col=self.date_col,
                target_col=self.spec.target_column,
                fitted=True,
                static_features=model_kwargs.get("static_features", []),
                max_horizon=None if num_models is False else self.spec.horizon,
            )

            forecast = fcst.predict(
                h=self.spec.horizon,
                X_df=pd.concat(
                    [
                        data_test[model_columns],
                        fcst.get_missing_future(
                            h=self.spec.horizon, X_df=data_test[model_columns]
                        ),
                    ],
                    axis=0,
                    ignore_index=True,
                ).fillna(0),
                level=[level]
            )

            forcast_col = "forecast"
            lower_ci_col = f"{forcast_col}-lo-{level}"
            upper_ci_col = f"{forcast_col}-hi-{level}"
            self.fitted_values = fcst.forecast_fitted_values()

            self.fcst = fcst
            for s_id in self.datasets.list_series_ids():
                self.forecast_output.init_series_output(
                    series_id=s_id,
                    data_at_series=self.datasets.get_data_at_series(s_id),
                )

                self.forecast_output.populate_series_output(
                    series_id=s_id,
                    fit_val=self.fitted_values[
                        self.fitted_values[ForecastOutputColumns.SERIES] == s_id
                        ][forcast_col].values,
                    forecast_val=forecast[
                        forecast[ForecastOutputColumns.SERIES] == s_id
                        ][forcast_col].values,
                    upper_bound=forecast[
                        forecast[ForecastOutputColumns.SERIES] == s_id
                        ][upper_ci_col].values,
                    lower_bound=forecast[
                        forecast[ForecastOutputColumns.SERIES] == s_id
                        ][lower_ci_col].values,
                )

                self.model_parameters[s_id] = {
                    "framework": SupportedModels.XGBForecast,
                    **xgb_params,
                    **fcst.models['forecast'].get_params(),
                }

            logger.debug("===========Done===========")
            predictions_df = forecast.sort_values(
                by=[ForecastOutputColumns.SERIES, self.dt_column_name]).reset_index(drop=True)
            future_df = data_test.sort_values(
                by=[ForecastOutputColumns.SERIES, self.dt_column_name]).reset_index(drop=True)
            future_df[self.spec.target_column] = predictions_df['forecast']
            self.full_dataset_with_prediction = pd.concat([data_train, future_df], ignore_index=True, axis=0)


        except Exception as e:
            self.errors_dict[self.spec.model] = {
                "model_name": self.spec.model,
                "error": str(e),
                "error_trace": traceback.format_exc(),
            }
            logger.warning(f"Encountered Error: {e}. Skipping.")
            logger.warning(traceback.format_exc())
            raise e

    def _generate_report(self):
        return super()._generate_report()
