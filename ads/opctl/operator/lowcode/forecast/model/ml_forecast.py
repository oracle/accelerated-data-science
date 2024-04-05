import pandas as pd
import numpy as np

from ads.opctl import logger
from ads.common.decorator import runtime_dependency
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
                freq=pd.infer_freq(data_train.Date.drop_duplicates()),
                target_transforms=[Differences([12])],
                lags=model_kwargs.get("lags", [1, 6, 12]),
                lag_transforms={
                    1: [ExpandingMean()],
                    12: [RollingMean(window_size=24)],
                },
                # date_features=[hour_index],
            )

            num_models = model_kwargs.get("recursive_models", False)

            fcst.fit(
                data_train,
                static_features=model_kwargs.get("static_features", []),
                id_col=ForecastOutputColumns.SERIES,
                time_col=self.spec.datetime_column.name,
                target_col=self.spec.target_column,
                fitted=True,
                max_horizon=None if num_models is False else self.spec.horizon,
            )

            self.outputs = fcst.predict(
                h=self.spec.horizon,
                X_df=pd.concat(
                    [
                        data_test,
                        fcst.get_missing_future(h=self.spec.horizon, X_df=data_test),
                    ],
                    axis=0,
                    ignore_index=True,
                ).fillna(0),
            )
            fitted_values = fcst.forecast_fitted_values()
            for s_id in self.datasets.list_series_ids():
                self.forecast_output.init_series_output(
                    series_id=s_id,
                    data_at_series=self.datasets.get_data_at_series(s_id),
                )

                self.forecast_output.populate_series_output(
                    series_id=s_id,
                    fit_val=fitted_values[
                        fitted_values[ForecastOutputColumns.SERIES] == s_id
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

            return self.forecast_output.get_forecast_long()
        except Exception as e:
            self.errors_dict[self.spec.model] = {
                "model_name": self.spec.model,
                "error": str(e),
            }

    def _build_model(self) -> pd.DataFrame:
        data_train = self.datasets.get_all_data_long(include_horizon=False)
        data_test = self.datasets.get_all_data_long_test()
        self.models = dict()
        model_kwargs = self.set_kwargs()
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width,
            horizon=self.spec.horizon,
            target_column=self.original_target_column,
            dt_column=self.spec.datetime_column.name,
        )
        self._train_model(data_train, data_test, model_kwargs)
        pass

    def _generate_report(self):
        pass
