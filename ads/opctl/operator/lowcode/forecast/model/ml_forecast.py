#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import logging
import traceback

import pandas as pd

from ads.common.decorator import runtime_dependency
from ads.opctl import logger
from ads.opctl.operator.lowcode.forecast.utils import _select_plot_list

from ..const import ForecastOutputColumns, SupportedModels
from ..operator_config import ForecastOperatorConfig
from .base_model import ForecastOperatorBaseModel
from .forecast_datasets import ForecastDatasets, ForecastOutput


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
        import lightgbm as lgb
        from mlforecast import MLForecast
        from mlforecast.lag_transforms import ExpandingMean, RollingMean
        from mlforecast.target_transforms import Differences

        def set_model_config(freq):
            seasonal_map = {
                "H": 24,
                "D": 7,
                "W": 52,
                "M": 12,
                "Q": 4,
            }
            sp = seasonal_map.get(freq.upper(), 7)
            default_lags = [1, sp, 2 * sp]
            lags = model_kwargs.get("lags", default_lags)

            default_roll = 2 * sp
            roll = model_kwargs.get("RollingMean", default_roll)

            diff = model_kwargs.get("Differences", sp)

            return {
                "target_transforms": [Differences([diff])],
                "lags": lags,
                "lag_transforms": {
                    1: [ExpandingMean()],
                    sp: [RollingMean(window_size=roll, min_samples=1)]
                }
            }

        try:

            lgb_params = {
                "verbosity": model_kwargs.get("verbosity", -1),
                "num_leaves": model_kwargs.get("num_leaves", 512),
            }

            data_freq = pd.infer_freq(data_train[self.date_col].drop_duplicates()) \
                        or pd.infer_freq(data_train[self.date_col].drop_duplicates()[-5:])

            additional_data_params = set_model_config(data_freq)

            fcst = MLForecast(
                models={
                    "forecast": lgb.LGBMRegressor(**lgb_params),
                    "upper": lgb.LGBMRegressor(
                        **lgb_params,
                        objective="quantile",
                        alpha=model_kwargs["uppper_quantile"],
                    ),
                    "lower": lgb.LGBMRegressor(
                        **lgb_params,
                        objective="quantile",
                        alpha=model_kwargs["lower_quantile"],
                    ),
                },
                freq=data_freq,
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
                    "framework": SupportedModels.LGBForecast,
                    **lgb_params,
                    **fcst.models_['forecast'].get_params(),
                }

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

    def _build_model(self) -> pd.DataFrame:
        data_train = self.datasets.get_all_data_long(include_horizon=False)
        data_test = self.datasets.get_all_data_long_forecast_horizon()
        self.models = {}
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

        logging.getLogger("report_creator").setLevel(logging.WARNING)

        # Section 2: LGBForecast Model Parameters
        sec2_text = rc.Block(
            rc.Heading("LGBForecast Model Parameters", level=2),
            rc.Text("These are the parameters used for the LGBForecast model."),
        )

        k, v = next(iter(self.model_parameters.items()))
        sec_2 = rc.Html(
            pd.DataFrame(list(v.items())).to_html(index=False, header=False),
        )

        all_sections = [sec2_text, sec_2]
        model_description = rc.Text(
            "LGBForecast uses mlforecast framework to perform time series forecasting using machine learning models"
            "with the option to scale to massive amounts of data using remote clusters."
            "Fastest implementations of feature engineering for time series forecasting in Python."
            "Support for exogenous variables and static covariates."
        )

        return model_description, all_sections
