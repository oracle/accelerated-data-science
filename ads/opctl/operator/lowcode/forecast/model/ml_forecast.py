#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import pandas as pd
import numpy as np

from ads.opctl import logger
from ads.common.decorator import runtime_dependency
from ads.opctl.operator.lowcode.forecast.utils import _select_plot_list
from .base_model import ForecastOperatorBaseModel
from .forecast_datasets import ForecastDatasets, ForecastOutput
from ..operator_config import ForecastOperatorConfig
from ..const import ForecastOutputColumns, SupportedModels


class MLForecastOperatorModel(ForecastOperatorBaseModel):
    """Class representing MLForecast operator model."""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config=config, datasets=datasets)
        self.global_explanation = {}
        self.local_explanation = {}
        self.formatted_global_explanation = None
        self.formatted_local_explanation = None
        self.date_col = config.spec.datetime_column.name

    def set_kwargs(self):
        """
        Returns the model parameters.
        """
        model_kwargs = self.spec.model_kwargs

        uppper_quantile = round(0.5 + self.spec.confidence_interval_width / 2, 2)
        lower_quantile = round(0.5 - self.spec.confidence_interval_width / 2, 2)

        model_kwargs["lower_quantile"] = lower_quantile
        model_kwargs["uppper_quantile"] = uppper_quantile
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
        try:
            import lightgbm as lgb
            from mlforecast import MLForecast
            from mlforecast.lag_transforms import ExpandingMean, RollingMean
            from mlforecast.target_transforms import Differences

            lgb_params = {
                "verbosity": -1,
                "num_leaves": 512,
            }
            additional_data_params = {}
            if len(self.datasets.get_additional_data_column_names()) > 0:
                additional_data_params = {
                    "target_transforms": [Differences([12])],
                    "lags": model_kwargs.get("lags", [1, 6, 12]),
                    "lag_transforms": (
                        {
                            1: [ExpandingMean()],
                            12: [RollingMean(window_size=24)],
                        }
                    ),
                }

            fcst = MLForecast(
                models={
                    "forecast": lgb.LGBMRegressor(**lgb_params),
                    # "p" + str(int(model_kwargs["uppper_quantile"] * 100))
                    "upper": lgb.LGBMRegressor(
                        **lgb_params,
                        objective="quantile",
                        alpha=model_kwargs["uppper_quantile"],
                    ),
                    # "p" + str(int(model_kwargs["lower_quantile"] * 100))
                    "lower": lgb.LGBMRegressor(
                        **lgb_params,
                        objective="quantile",
                        alpha=model_kwargs["lower_quantile"],
                    ),
                },
                freq=pd.infer_freq(data_train[self.date_col].drop_duplicates())
                or pd.infer_freq(data_train[self.date_col].drop_duplicates()[-5:]),
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

            self.outputs = fcst.predict(
                h=self.spec.horizon,
                X_df=pd.concat(
                    [
                        data_test[self.model_columns],
                        fcst.get_missing_future(
                            h=self.spec.horizon, X_df=data_test[self.model_columns]
                        ),
                    ],
                    axis=0,
                    ignore_index=True,
                ).fillna(0),
            )
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

                self.model_parameters[s_id] = {
                    "framework": SupportedModels.MLForecast,
                    **lgb_params,
                }

            logger.debug("===========Done===========")

        except Exception as e:
            self.errors_dict[self.spec.model] = {
                "model_name": self.spec.model,
                "error": str(e),
            }
            logger.debug(f"Encountered Error: {e}. Skipping.")
            raise e

    def _build_model(self) -> pd.DataFrame:
        data_train = self.datasets.get_all_data_long(include_horizon=False)
        data_test = self.datasets.get_all_data_long_forecast_horizon()
        self.models = dict()
        model_kwargs = self.set_kwargs()
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width,
            horizon=self.spec.horizon,
            target_column=self.original_target_column,
            dt_column=self.date_col,
        )
        self._train_model(data_train, data_test, model_kwargs)
        return self.forecast_output.get_forecast_long()

    def _generate_report(self):
        """
        Generates the report for the model
        """
        import report_creator as rc
        from utilsforecast.plotting import plot_series

        # Section 1: Forecast Overview
        sec1_text = rc.Block(
            rc.Heading("Forecast Overview", level=2),
            rc.Text(
                "These plots show your forecast in the context of historical data."
            ),
        )
        sec_1 = _select_plot_list(
            lambda s_id: plot_series(
                self.datasets.get_all_data_long(include_horizon=False),
                pd.concat(
                    [self.fitted_values, self.outputs], axis=0, ignore_index=True
                ),
                id_col=ForecastOutputColumns.SERIES,
                time_col=self.spec.datetime_column.name,
                target_col=self.original_target_column,
                seed=42,
                ids=[s_id],
            ),
            self.datasets.list_series_ids(),
        )

        # Section 2: MlForecast Model Parameters
        sec2_text = rc.Block(
            rc.Heading("MlForecast Model Parameters", level=2),
            rc.Text("These are the parameters used for the MlForecast model."),
        )

        blocks = [
            rc.Html(
                str(s_id[1]),
                label=s_id[0],
            )
            for _, s_id in enumerate(self.model_parameters.items())
        ]
        sec_2 = rc.Select(blocks=blocks)

        all_sections = [sec1_text, sec_1, sec2_text, sec_2]
        model_description = rc.Text(
            "mlforecast is a framework to perform time series forecasting using machine learning models"
            "with the option to scale to massive amounts of data using remote clusters."
            "Fastest implementations of feature engineering for time series forecasting in Python."
            "Support for exogenous variables and static covariates."
        )

        return model_description, all_sections
