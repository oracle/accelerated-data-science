#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import time
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype, is_numeric_dtype

from ..operator_config import ForecastOperatorConfig
from ads.opctl import logger
from ..const import ForecastOutputColumns, PROPHET_INTERNAL_DATE_COL
from ads.common.object_storage_details import ObjectStorageDetails
from ads.opctl.operator.lowcode.common.utils import (
    get_frequency_in_seconds,
    get_frequency_of_datetime,
)
from ads.opctl.operator.lowcode.common.data import AbstractData
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


class HistoricalData(AbstractData):
    def __init__(self, spec: dict):
        super().__init__(spec=spec, name="historical_data")

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


class AdditionalData(AbstractData):
    def __init__(self, spec, historical_data):
        if spec.additional_data is not None:
            super().__init__(spec=spec, name="additional_data")
            add_dates = self.data.index.get_level_values(0).unique().tolist()
            add_dates.sort()
            if historical_data.get_max_time() > add_dates[-spec.horizon]:
                raise DataMismatchError(
                    f"The Historical Data ends on {historical_data.get_max_time()}. The additional data horizon starts on {add_dates[-spec.horizon]}. The horizon should have exactly {spec.horizon} dates after the Hisotrical at a frequency of {historical_data.freq}"
                )
            elif historical_data.get_max_time() != add_dates[-(spec.horizon + 1)]:
                raise DataMismatchError(
                    f"The Additional Data must be present for all historical data and the entire horizon. The Historical Data ends on {historical_data.get_max_time()}. The additonal data horizon starts after {add_dates[-(spec.horizon+1)]}. These should be the same date."
                )
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
                periods=spec.horizon + 1,
                freq=historical_data.freq
                or pd.infer_freq(
                    historical_data.data.reset_index()[spec.datetime_column.name][-5:]
                ),
            ),
            name=spec.datetime_column.name,
        )
        add_dfs = []
        for s_id in historical_data.list_series_ids():
            df_i = historical_data.get_data_for_series(s_id)[spec.datetime_column.name]
            df_i = pd.DataFrame(pd.concat([df_i, future_dates[1:]]))
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


class TestData(AbstractData):
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
        self._target_col = config.spec.target_column
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

    def get_all_data_long_forecast_horizon(self):
        """Returns all data in long format for the forecast horizon."""
        test_data = pd.merge(
            self.historical_data.data,
            self.additional_data.data,
            how="outer",
            on=[self._datetime_column_name, ForecastOutputColumns.SERIES],
        ).reset_index()
        return test_data[test_data[self._target_col].isnull()].reset_index(drop=True)

    def get_data_multi_indexed(self):
        return pd.concat(
            [
                self.historical_data.data,
                self.additional_data.data,
            ],
            axis=1,
        )

    def get_data_by_series(self, include_horizon=True):
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

    def get_data_at_series(self, s_id, include_horizon=True):
        all_data = self.get_data_by_series(include_horizon=include_horizon)
        try:
            return all_data[s_id]
        except:
            raise InvalidParameterError(
                f"Unable to retrieve series id: {s_id} from data. Available series ids are: {self.list_series_ids()}"
            )

    def get_horizon_at_series(self, s_id):
        return self.get_data_at_series(s_id)[-self._horizon :]

    def has_artificial_series(self):
        return self.historical_data._data_transformer.has_artificial_series

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

    def list_series_ids(self, sorted=True):
        series_ids = self.historical_data.list_series_ids()
        if sorted:
            try:
                series_ids.sort()
            except:
                pass
        return series_ids

    def format_wide(self):
        data_merged = pd.concat(
            [
                v[v[k].notna()].set_index(self._datetime_column_name)
                for k, v in self.get_data_by_series().items()
            ],
            axis=1,
        ).reset_index()
        return data_merged

    def get_longest_datetime_column(self):
        return self.format_wide()[self._datetime_column_name]


class ForecastOutput:
    def __init__(
        self,
        confidence_interval_width: float,
        horizon: int,
        target_column: str,
        dt_column: str,
    ):
        """Forecast Output contains all of the details required to generate the forecast.csv output file.

        init
        -------
        confidence_interval_width: float  value from OperatorSpec
        horizon: int  length of horizon
        target_column: str the name of the original target column
        dt_column: the name of the original datetime column
        """
        self.series_id_map = dict()
        self._set_ci_column_names(confidence_interval_width)
        self.horizon = horizon
        self.target_column_name = target_column
        self.dt_column_name = dt_column

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
        self.series_id_map[series_id] = forecast

    def init_series_output(self, series_id, data_at_series):
        output_i = pd.DataFrame()

        output_i["Date"] = data_at_series[self.dt_column_name]
        output_i["Series"] = series_id
        output_i["input_value"] = data_at_series[self.target_column_name]

        output_i["fitted_value"] = float("nan")
        output_i["forecast_value"] = float("nan")
        output_i[self.lower_bound_name] = float("nan")
        output_i[self.upper_bound_name] = float("nan")
        self.series_id_map[series_id] = output_i

    def populate_series_output(
        self, series_id, fit_val, forecast_val, upper_bound, lower_bound
    ):
        """
        This method should be run after init_series_output has been run on this series_id

        Parameters:
        -----------
        series_id: [str, int] the series being forecasted
        fit_val: numpy.array of length input_value - horizon
        forecast_val: numpy.array of length horizon containing the forecasted values
        upper_bound: numpy.array of length horizon containing the upper_bound values
        lower_bound: numpy.array of length horizon containing the lower_bound values

        Returns:
        --------
        None
        """
        try:
            output_i = self.series_id_map[series_id]
        except KeyError:
            raise ValueError(
                f"Attempting to update output for series: {series_id}, however no series output has been initialized."
            )

        if (output_i.shape[0] - self.horizon) == len(fit_val):
            output_i["fitted_value"].iloc[
                : -self.horizon
            ] = fit_val  # Note: may need to do len(output_i) - (len(fit_val) + horizon) : -horizon
        elif (output_i.shape[0] - self.horizon) > len(fit_val):
            logger.debug(
                f"Fitted Values were only generated on a subset ({len(fit_val)}/{(output_i.shape[0] - self.horizon)}) of the data for Series: {series_id}."
            )
            start_idx = output_i.shape[0] - self.horizon - len(fit_val)
            output_i["fitted_value"].iloc[start_idx : -self.horizon] = fit_val
        else:
            output_i["fitted_value"].iloc[start_idx : -self.horizon] = fit_val[
                -(output_i.shape[0] - self.horizon) :
            ]

        if len(forecast_val) != self.horizon:
            raise ValueError(
                f"Attempting to set forecast along horizon ({self.horizon}) for series: {series_id}, however forecast is only length {len(forecast_val)}"
            )
        output_i["forecast_value"].iloc[-self.horizon :] = forecast_val

        if len(upper_bound) != self.horizon:
            raise ValueError(
                f"Attempting to set upper_bound along horizon ({self.horizon}) for series: {series_id}, however upper_bound is only length {len(upper_bound)}"
            )
        output_i[self.upper_bound_name].iloc[-self.horizon :] = upper_bound

        if len(lower_bound) != self.horizon:
            raise ValueError(
                f"Attempting to set lower_bound along horizon ({self.horizon}) for series: {series_id}, however lower_bound is only length {len(lower_bound)}"
            )
        output_i[self.lower_bound_name].iloc[-self.horizon :] = lower_bound

        self.series_id_map[series_id] = output_i
        self.verify_series_output(series_id)

    def verify_series_output(self, series_id):
        forecast = self.series_id_map[series_id]
        self._check_forecast_format(forecast)

    def get_horizon_by_series(self, series_id):
        return self.series_id_map[series_id][-self.horizon :]

    def get_horizon_long(self):
        df = pd.DataFrame()
        for s_id in self.list_series_ids():
            df = pd.concat([df, self.get_horizon_by_series(s_id)])
        return df.reset_index(drop=True)

    def get_forecast(self, series_id):
        try:
            return self.series_id_map[series_id]
        except KeyError as ke:
            logger.debug(
                f"No Forecast found for series_id: {series_id}. Returning empty DataFrame."
            )
            return pd.DataFrame()

    def list_series_ids(self, sorted=True):
        series_ids = list(self.series_id_map.keys())
        if sorted:
            try:
                series_ids.sort()
            except:
                pass
        return series_ids

    def _set_ci_column_names(self, confidence_interval_width):
        yhat_lower_percentage = (100 - confidence_interval_width * 100) // 2
        self.upper_bound_name = "p" + str(int(100 - yhat_lower_percentage))
        self.lower_bound_name = "p" + str(int(yhat_lower_percentage))

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
        assert self.upper_bound_name in forecast.columns
        assert self.lower_bound_name in forecast.columns
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

    def get_forecast_long(self):
        output = pd.DataFrame()
        for df in self.series_id_map.values():
            output = pd.concat([output, df])
        return output.reset_index(drop=True)
