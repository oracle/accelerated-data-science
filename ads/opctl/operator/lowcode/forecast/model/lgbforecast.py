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


class LGBForecastOperatorModel(MLForecastBaseModel):
    """Class representing MLForecast operator model."""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config=config, datasets=datasets)
        self.model_name = "LGBForecast"
        self.model_description = """LightGBM for forecasting is a gradient-boosted tree model optimized for speed and 
            scalability that learns nonlinear patterns from lagged features and exogenous variables. It trains faster than 
            XGBoost on large datasets while delivering comparable or better accuracy with proper feature engineering."""

    def get_model_kwargs(self):
        """
        Returns the model parameters.
        """
        model_kwargs = self.spec.model_kwargs

        upper_quantile = round(0.5 + self.spec.confidence_interval_width / 2, 2)
        lower_quantile = round(0.5 - self.spec.confidence_interval_width / 2, 2)

        model_kwargs["lower_quantile"] = lower_quantile
        model_kwargs["upper_quantile"] = upper_quantile
        return model_kwargs


    def preprocess(self, df, series_id):
        pass

    @runtime_dependency(
        module="mlforecast",
        err_msg="MLForecast is not installed, please install it with 'pip install mlforecast'",
    )
    @runtime_dependency(
        module="lightgbm",
        err_msg="lightgbm is not installed, please install it with 'pip install lightgbm'",
    )
    def _train_model(self, data_train, data_test, model_kwargs):
        import lightgbm as lgb
        from mlforecast import MLForecast
        try:

            lgb_params = {
                "verbosity": model_kwargs.get("verbosity", -1),
                "num_leaves": model_kwargs.get("num_leaves", 512),
            }

            data_freq = self.datasets.get_datetime_frequency()

            additional_data_params = self.set_model_config(data_freq, model_kwargs)

            fcst = MLForecast(
                models={
                    "forecast": lgb.LGBMRegressor(**lgb_params),
                    "upper": lgb.LGBMRegressor(
                        **lgb_params,
                        objective="quantile",
                        alpha=model_kwargs["upper_quantile"],
                    ),
                    "lower": lgb.LGBMRegressor(
                        **lgb_params,
                        objective="quantile",
                        alpha=model_kwargs["lower_quantile"],
                    ),
                },
                freq=data_freq,
                date_features=['year', 'month', 'day', 'dayofweek', 'dayofyear'],
                **additional_data_params,
            )

            num_models = model_kwargs.get("recursive_models", False)

            self.model_columns = [
                ForecastOutputColumns.SERIES
            ] + data_train.select_dtypes(exclude=["object"]).columns.to_list()
            fcst.fit(
                data_train[self.model_columns],
                static_features=model_kwargs.get("static_features", []),
                id_col=ForecastOutputColumns.SERIES,
                time_col=self.date_col,
                target_col=self.spec.target_column,
                fitted=True,
                max_horizon=None if num_models is False else self.spec.horizon,
            )

            future_exog_df = pd.concat(
                [
                    data_test[self.model_columns],
                    fcst.get_missing_future(
                        h=self.spec.horizon, X_df=data_test[self.model_columns]
                    ),
                ],
                axis=0,
                ignore_index=True,
            )
            future_exog_df = future_exog_df.fillna(0)

            self.outputs = fcst.predict(
                h=self.spec.horizon,
                X_df=future_exog_df,
            )
            self.fcst = fcst

            self.fitted_values = fcst.forecast_fitted_values()
            for s_id in self.datasets.list_series_ids():
                self.forecast_output.init_series_output(
                    series_id=s_id,
                    data_at_series=self.datasets.get_data_at_series(s_id),
                )

                self.forecast_output.populate_series_output(
                    series_id=s_id,
                    fit_val=self.fitted_values[
                        self.fitted_values[ForecastOutputColumns.SERIES] == s_id
                    ].forecast.values,
                    forecast_val=self.outputs[
                        self.outputs[ForecastOutputColumns.SERIES] == s_id
                    ].forecast.values,
                    upper_bound=self.outputs[
                        self.outputs[ForecastOutputColumns.SERIES] == s_id
                    ].upper.values,
                    lower_bound=self.outputs[
                        self.outputs[ForecastOutputColumns.SERIES] == s_id
                    ].lower.values,
                )

                one_step_model = fcst.models_['forecast'][0] if isinstance(fcst.models_['forecast'], list) else \
                fcst.models_['forecast']
                self.model_parameters[s_id] = {
                    "framework": SupportedModels.LGBForecast,
                    **lgb_params,
                    **one_step_model.get_params(),
                }

            predictions_df = self.outputs.sort_values(
                by=[ForecastOutputColumns.SERIES, self.dt_column_name]).reset_index(drop=True)
            future_df = future_exog_df.sort_values(
                by=[ForecastOutputColumns.SERIES, self.dt_column_name]).reset_index(drop=True)
            future_df[self.spec.target_column] = predictions_df['forecast']
            self.full_dataset_with_prediction = pd.concat([data_train, future_df], ignore_index=True, axis=0)

            logger.debug("===========Done===========")

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