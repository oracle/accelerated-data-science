#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import time
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype, is_numeric_dtype

from ..operator_config import ForecastOperatorConfig
from .transformations import Transformations
from ads.opctl import logger
from ..const import ForecastOutputColumns, PROPHET_INTERNAL_DATE_COL
from ads.common.object_storage_details import ObjectStorageDetails
from ads.opctl.operator.lowcode.common.utils import (
    load_data,
    get_frequency_in_seconds,
    get_frequency_of_datetime,
)
from ads.opctl.operator.lowcode.forecast.utils import (
    default_signer,
)
from ads.opctl.operator.lowcode.common.errors import (
    InputDataError,
    InvalidParameterError,
    PermissionsError,
    DataMismatchError,
)
from ..const import SupportedModels
from abc import ABC, abstractmethod


class AbstractForecastData(ABC):
    def __init__(self, spec: dict, name="historical_data"):
        self.data = None
        self._data_dict = dict()
        self.name = name
        self.load_transform_ingest_data(spec)

    def get_dict_by_series(self):
        if not self._data_dict:
            for s_id in self.list_series_ids():
                try:
                    self._data_dict[s_id] = self.data.xs(
                        s_id, level=ForecastOutputColumns.SERIES
                    ).reset_index()
                except KeyError as ke:
                    logger.debug(
                        f"Unable to extract series: {s_id} from data: {self.data}. This may occur due to significant missing data. Error message: {ke.args}"
                    )
                    pass
        return self._data_dict

    def get_data_for_series(self, series_id):
        data_dict = self.get_dict_by_series()
        try:
            return data_dict[series_id]
        except:
            raise InvalidParameterError(
                f"Unable to retrieve series {series_id} from {self.name}. Available series ids are: {self.list_series_ids()}"
            )

    def _load_data(self, data_spec, **kwargs):
        loading_start_time = time.time()
        try:
            raw_data = load_data(
                filename=data_spec.url,
                format=data_spec.format,
                columns=data_spec.columns,
            )
        except InvalidParameterError as e:
            e.args = e.args + (f"Invalid Parameter: {self.name}",)
            raise e
        loading_end_time = time.time()
        logger.info(
            f"{self.name} loaded in {loading_end_time - loading_start_time} seconds",
        )
        return raw_data

    def _transform_data(self, spec, raw_data, **kwargs):
        transformation_start_time = time.time()
        self._data_transformer = Transformations(spec, name=self.name)
        data = self._data_transformer.run(raw_data)
        transformation_end_time = time.time()
        logger.info(
            f"{self.name} transformations completed in {transformation_end_time - transformation_start_time} seconds"
        )
        return data

    def _ingest_data(self, spec):
        pass

    def load_transform_ingest_data(self, spec):
        raw_data = self._load_data(getattr(spec, self.name))
        self.data = self._transform_data(spec, raw_data)
        self._ingest_data(spec)

    def get_min_time(self):
        return self.data.index.get_level_values(0).min()

    def get_max_time(self):
        return self.data.index.get_level_values(0).max()

    def list_series_ids(self):
        return self.data.index.get_level_values(1).unique().tolist()

    def get_num_rows(self):
        return self.data.shape[0]


class HistoricalData(AbstractForecastData):
    def _ingest_data(self, spec):
        try:
            self.freq = get_frequency_of_datetime(self.data.index.get_level_values(0))
        except TypeError as e:
            logger.warn(
                f"Error determining frequency: {e.args}. Setting Frequency to None"
            )
            logger.debug(f"Full traceback: {e}")
            self.freq = None
        self._verify_dt_col(spec)
        super()._ingest_data(spec)

    def _verify_dt_col(self, spec):
        # Check frequency is compatible with model type
        self.freq_in_secs = get_frequency_in_seconds(
            self.data.index.get_level_values(0)
        )
        if spec.model == SupportedModels.AutoMLX:
            if abs(self.freq_in_secs) < 3600:
                message = (
                    "{} requires data with a frequency of at least one hour. Please try using a different model,"
                    " or select the 'auto' option.".format(SupportedModels.AutoMLX)
                )
                raise InvalidParameterError(message)


class AdditionalData(AbstractForecastData):
    def __init__(self, spec, historical_data):
        if spec.additional_data is not None:
            super().__init__(spec=spec, name="additional_data")
        else:
            self.name = "additional_data"
            self.data = None
            self._data_dict = dict()
            self.create_horizon(spec, historical_data)

    def create_horizon(self, spec, historical_data):
        logger.debug(f"No additional data provided. Constructing horizon.")
        future_dates = pd.Series(
            pd.date_range(
                start=historical_data.get_max_time(),
                periods=spec.horizon,
                freq=historical_data.freq,
            ),
            name=spec.datetime_column.name,
        )
        add_dfs = []
        for s_id in historical_data.list_series_ids():
            df_i = historical_data.get_data_for_series(s_id)[spec.datetime_column.name]
            df_i = pd.DataFrame(pd.concat([df_i, future_dates]))
            df_i[ForecastOutputColumns.SERIES] = s_id
            df_i = df_i.set_index(
                [spec.datetime_column.name, ForecastOutputColumns.SERIES]
            )
            add_dfs.append(df_i)
        data = pd.concat(add_dfs, axis=1)
        self.data = data.sort_values(
            [spec.datetime_column.name, ForecastOutputColumns.SERIES], ascending=True
        )
        self.additional_regressors = []

    def _ingest_data(self, spec):
        self.additional_regressors = list(self.data.columns)
        if not self.additional_regressors:
            logger.warn(
                f"No additional variables found in the additional_data. Only columns found: {self.data.columns}. Skipping for now."
            )
        # Check that datetime column matches historical datetime column


class TestData(AbstractForecastData):
    def __init__(self, spec):
        super().__init__(spec=spec, name="test_data")
        self.dt_column_name = spec.datetime_column.name
        self.target_name = spec.target_column


class ForecastDatasets:
    def __init__(self, config: ForecastOperatorConfig):
        """Instantiates the DataIO instance.

        Properties
        ----------
        config: ForecastOperatorConfig
            The forecast operator configuration.
        """
        self.historical_data: HistoricalData = None
        self.additional_data: AdditionalData = None

        self._horizon = config.spec.horizon
        self._datetime_column_name = config.spec.datetime_column.name
        self._load_data(config.spec)

    def _load_data(self, spec):
        """Loads forecasting input data."""
        self.historical_data = HistoricalData(spec)
        self.additional_data = AdditionalData(spec, self.historical_data)

        if spec.generate_explanations:
            if spec.additional_data is None:
                logger.warn(
                    f"Unable to generate explanations as there is no additional data passed in. Either set generate_explanations to False, or pass in additional data."
                )
                spec.generate_explanations = False

    def get_all_data_long(self, include_horizon=True):
        how = "outer" if include_horizon else "left"
        return pd.merge(
            self.historical_data.data,
            self.additional_data.data,
            how=how,
            on=[self._datetime_column_name, ForecastOutputColumns.SERIES],
        ).reset_index()

    def get_all_data_by_series(self, include_horizon=True):
        total_dict = dict()
        hist_data = self.historical_data.get_dict_by_series()
        add_data = self.additional_data.get_dict_by_series()
        how = "outer" if include_horizon else "left"
        for s_id in self.list_series_ids():
            # Note: ensure no duplicate column names
            total_dict[s_id] = pd.merge(
                hist_data[s_id],
                add_data[s_id],
                how=how,
                on=[self._datetime_column_name],
            )
        return total_dict

    def get_all_data_for_series_id(self, s_id, include_horizon=True):
        all_data = self.get_all_data_by_series(include_horizon=include_horizon)
        try:
            return all_data[s_id]
        except:
            raise InvalidParameterError(
                f"Unable to retrieve series id: {s_id} from data. Available series ids are: {self.list_series_ids()}"
            )

    def get_earliest_timestamp(self):
        return self.historical_data.get_min_time()

    def get_latest_timestamp(self):
        return self.historical_data.get_max_time()

    def get_additional_data_column_names(self):
        return self.additional_data.additional_regressors

    def get_datetime_frequency(self):
        return self.historical_data.freq

    def get_datetime_frequency_in_seconds(self):
        return self.historical_data.freq_in_secs

    def get_num_rows(self):
        return self.historical_data.get_num_rows()

    def list_series_ids(self):
        return self.historical_data.list_series_ids()

    def format_wide(self):
        data_merged = pd.concat(
            [
                v[v[k].notna()].set_index(self._datetime_column_name)
                for k, v in self.get_all_data_by_series().items()
            ],
            axis=1,
        ).reset_index()
        return data_merged

    def get_longest_datetime_column(self):
        return self.format_wide()[self._datetime_column_name]


class ForecastOutput:
    def __init__(self, confidence_interval_width: float):
        """Forecast Output contains all of the details required to generate the forecast.csv output file.

        Methods
        ----------

        """
        self.series_id_map = dict()
        self.confidence_interval_width = confidence_interval_width
        self.upper_bound_name = None
        self.lower_bound_name = None

    def add_series_id(
        self,
        series_id: str,
        forecast: pd.DataFrame,
        overwrite: bool = False,
    ):
        if not overwrite and series_id in self.series_id_map.keys():
            raise ValueError(
                f"Attempting to update ForecastOutput for series_id {series_id} when this already exists. Set overwrite to True."
            )
        forecast = self._check_forecast_format(forecast)
        forecast = self._set_ci_column_names(forecast)
        self.series_id_map[series_id] = forecast

    def get_forecast(self, series_id):
        return self.series_id_map[series_id]

    def list_series_ids(self):
        return list(self.series_id_map.keys())

    def _set_ci_column_names(self, forecast_i):
        yhat_lower_percentage = (100 - self.confidence_interval_width * 100) // 2
        self.upper_bound_name = "p" + str(int(100 - yhat_lower_percentage))
        self.lower_bound_name = "p" + str(int(yhat_lower_percentage))
        return forecast_i.rename(
            {
                ForecastOutputColumns.UPPER_BOUND: self.upper_bound_name,
                ForecastOutputColumns.LOWER_BOUND: self.lower_bound_name,
            },
            axis=1,
        )

    def _check_forecast_format(self, forecast):
        assert isinstance(forecast, pd.DataFrame)
        assert (
            len(forecast.columns) == 7
        ), f"Expected just 7 columns, but got: {forecast.columns}"
        assert ForecastOutputColumns.DATE in forecast.columns
        assert ForecastOutputColumns.SERIES in forecast.columns
        assert ForecastOutputColumns.INPUT_VALUE in forecast.columns
        assert ForecastOutputColumns.FITTED_VALUE in forecast.columns
        assert ForecastOutputColumns.FORECAST_VALUE in forecast.columns
        assert ForecastOutputColumns.UPPER_BOUND in forecast.columns
        assert ForecastOutputColumns.LOWER_BOUND in forecast.columns
        assert not forecast.empty
        # forecast.columns = pd.Index([
        #     ForecastOutputColumns.DATE,
        #     ForecastOutputColumns.SERIES,
        #     ForecastOutputColumns.INPUT_VALUE,
        #     ForecastOutputColumns.FITTED_VALUE,
        #     ForecastOutputColumns.FORECAST_VALUE,
        #     ForecastOutputColumns.UPPER_BOUND,
        #     ForecastOutputColumns.LOWER_BOUND,
        # ])
        return forecast
