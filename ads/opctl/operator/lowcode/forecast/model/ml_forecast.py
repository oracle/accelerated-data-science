#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import ABC, abstractmethod

import pandas as pd

from ads.common.decorator import runtime_dependency
from .base_model import ForecastOperatorBaseModel
from .forecast_datasets import ForecastDatasets, ForecastOutput
from ..const import ForecastOutputColumns
from ..operator_config import ForecastOperatorConfig


class MLForecastBaseModel(ForecastOperatorBaseModel, ABC):
    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config=config, datasets=datasets)
        self.global_explanation = {}
        self.local_explanation = {}
        self.formatted_global_explanation = None
        self.formatted_local_explanation = None
        self.date_col = config.spec.datetime_column.name
        self.data_train = self.datasets.get_all_data_long(include_horizon=False)
        self.data_test = self.datasets.get_all_data_long_forecast_horizon()

    @runtime_dependency(
        module="mlforecast",
        err_msg="MLForecast is not installed, please install it with 'pip install mlforecast'",
    )
    def set_model_config(self, freq, model_kwargs):
        from mlforecast.lag_transforms import ExpandingMean, RollingMean
        from mlforecast.target_transforms import Differences
        seasonal_map = {
            "H": 24,
            "D": 7,
            "W": 52,
            "M": 12,
            "Q": 4,
        }
        sp = seasonal_map.get(freq.upper(), 7)
        series_lengths = self.data_train.groupby(ForecastOutputColumns.SERIES).size()
        min_len = series_lengths.min()
        max_allowed = min_len - sp

        default_lags = [lag for lag in [1, sp, 2 * sp] if lag <= max_allowed]
        lags = model_kwargs.get("lags", default_lags)

        default_roll = 2 * sp
        roll = model_kwargs.get("RollingMean", default_roll)

        default_diff = sp if sp <= max_allowed else None
        diff = model_kwargs.get("Differences", default_diff)

        return {
            "target_transforms": [Differences([diff])],
            "lags": lags,
            "lag_transforms": {
                1: [ExpandingMean()],
                sp: [RollingMean(window_size=roll, min_samples=1)]
            }
        }

    @abstractmethod
    def _train_model(self, data_train, data_test, model_kwargs) -> pd.DataFrame:
        """
        Build the model.
        The method that needs to be implemented on the particular model level.
        """

    @abstractmethod
    def get_model_kwargs(self) -> pd.DataFrame:
        """
        Build the model.
        The method that needs to be implemented on the particular model level.
        """

    def _build_model(self) -> pd.DataFrame:
        self.models = {}
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width,
            horizon=self.spec.horizon,
            target_column=self.original_target_column,
            dt_column=self.date_col,
        )
        self._train_model(self.data_train, self.data_test, self.get_model_kwargs())
        return self.forecast_output.get_forecast_long()
