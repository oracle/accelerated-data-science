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


class ForecastDatasets:
    def __init__(self, config: ForecastOperatorConfig):
        """Instantiates the DataIO instance.

        Properties
        ----------
        config: ForecastOperatorConfig
            The forecast operator configuration.
        """
        self.target_columns = None
        self._horizon = config.spec.horizon
        self._datetime_column_name = config.spec.datetime_column.name
        self._hist_dd = dict()
        self._add_dd = dict()
        self._data_long = None
        self._load_data(config.spec)

    def _verify_dt_col(self, spec):
        # Check frequency is compatible with model type
        self._freq_in_secs = get_frequency_in_seconds(
            self._data.index.get_level_values(0)
        )
        if spec.model == SupportedModels.AutoMLX:
            if abs(self._freq_in_secs) < 3600:
                message = (
                    "{} requires data with a frequency of at least one hour. Please try using a different model,"
                    " or select the 'auto' option.".format(SupportedModels.AutoMLX)
                )
                raise InvalidParameterError(message)

    def _load_transform_historical_data(self, spec):
        loading_start_time = time.time()
        try:
            raw_data = load_data(
                filename=spec.historical_data.url,
                format=spec.historical_data.format,
                columns=spec.historical_data.columns,
            )
        except InvalidParameterError as e:
            e.args = e.args + ("Invalid Parameter: historical_data",)
            raise e

        loading_end_time = time.time()
        logger.info(
            "Data loaded in %s seconds",
            loading_end_time - loading_start_time,
        )
        self._data_transformer = Transformations(spec)
        self._data = self._data_transformer.run(raw_data)
        transformation_end_time = time.time()
        logger.info(
            "Data Transformations completed in %s seconds",
            transformation_end_time - loading_end_time,
        )

    def _extract_historical_data_metadata(self, spec):
        try:
            self._freq = get_frequency_of_datetime(self._data.index.get_level_values(0))
        except TypeError as e:
            logger.warn(
                f"Error determining frequency: {e.args}. Setting Frequency to None"
            )
            logger.debug(f"Full traceback: {e}")
            self._freq = None

        self._num_rows = self._data.shape[0]
        self._verify_dt_col(spec)
        # These are used in generating the report
        self._min_time = self._data.index.get_level_values(0).min()
        self._max_time = self._data.index.get_level_values(0).max()
        self._categories = self._data.index.get_level_values(1).unique().tolist()

    def _load_transform_additional_data(self, spec):
        if spec.additional_data is None:
            logger.debug("No additional data detected.")
            self._additional_data = None
        loading_start_time = time.time()
        try:
            raw_data = load_data(
                filename=spec.additional_data.url,
                format=spec.additional_data.format,
                columns=spec.additional_data.columns,
            )
        except InvalidParameterError as e:
            e.args = e.args + ("Invalid Parameter: additional_data",)
            raise e

        loading_end_time = time.time()
        logger.info(
            "Loading additional data completed in %s seconds",
            loading_end_time - loading_start_time,
        )
        self._additional_data = self._data_transformer.transform_additional_data(
            raw_data
        )
        transformation_end_time = time.time()
        logger.info(
            "Transformations on additional data are completed in %s seconds",
            transformation_end_time - loading_end_time,
        )

    def _extract_additional_data_metadata(self, spec):
        if self._additional_data is None:
            self._additional_col_names = []
            return

        self._additional_col_names = list(self._additional_data.columns)
        if not self._additional_col_names:
            logger.warn(
                f"No additional variables found in the additional_data. Only columns found: {self._additional_data.columns}. Skipping for now."
            )

    # # TODO: delete me?
    # def _create_horizon(self, spec):
    #     if spec.additional_data is None:
    #         # Need to add the horizon to the data for compatibility
    #         additional_data_small = data[
    #             [spec.datetime_column.name] + spec.target_category_columns
    #         ].set_index(spec.datetime_column.name)
    #         if is_datetime64_any_dtype(additional_data_small.index):
    #             horizon_index = pd.date_range(
    #                 start=additional_data_small.index.values[-1],
    #                 freq=self._freq,
    #                 periods=spec.horizon + 1,
    #             )[1:]
    #         elif is_numeric_dtype(additional_data_small.index):
    #             # If datetime column is just ints
    #             assert (
    #                 len(additional_data_small.index.values) > 1
    #             ), "Dataset is too small to infer frequency. Please pass in the horizon explicitly through the additional data."
    #             start = additional_data_small.index.values[-1]
    #             step = (
    #                 additional_data_small.index.values[-1]
    #                 - additional_data_small.index.values[-2]
    #             )
    #             horizon_index = pd.RangeIndex(
    #                 start, start + step * (spec.horizon + 1), step=step
    #             )[1:]
    #         else:
    #             raise ValueError(
    #                 f"Unable to determine the datetime type for column: {spec.datetime_column.name}. Please specify the format explicitly."
    #             )

    #         additional_data = pd.DataFrame()

    #         for cat_col in spec.target_category_columns:
    #             for cat in additional_data_small[cat_col].unique():
    #                 add_data_i = additional_data_small[
    #                     additional_data_small[cat_col] == cat
    #                 ]
    #                 horizon_df_i = pd.DataFrame([], index=horizon_index)
    #                 horizon_df_i[cat_col] = cat
    #                 additional_data = pd.concat(
    #                     [additional_data, add_data_i, horizon_df_i]
    #                 )
    #         additional_data = additional_data.reset_index().rename(
    #             {"index": spec.datetime_column.name}, axis=1
    #         )

    def _load_data(self, spec):
        """Loads forecasting input data."""
        self._load_transform_historical_data(spec)
        self._extract_historical_data_metadata(spec)
        self._load_transform_additional_data(spec)
        self._extract_additional_data_metadata(spec)

        # Merge target category columns into 1 column
        # make the merged series column + Datatime columns the multi index
        # return the new column as ForecastOutputColumns.SERIES
        # Have a class for converting back into the series if it exists

        self.full_data_dict = self.get_all_data_by_series()
        self.target_columns = self.list_categories()
        # (
        #     self.full_data_dict,
        #     self.target_columns,
        # ) = _build_indexed_datasets(
        #     data=self._data,
        #     target_column=spec.target_column,
        #     datetime_column=spec.datetime_column.name,
        #     horizon=spec.horizon,
        #     target_category_columns=spec.target_category_columns,
        #     additional_data=additional_data,
        # )
        if spec.generate_explanations:
            if spec.additional_data is None:
                logger.warn(
                    f"Unable to generate explanations as there is no additional data passed in. Either set generate_explanations to False, or pass in additional data."
                )
                spec.generate_explanations = False

    def get_all_data_long(self):
        return pd.merge(
            self._data,
            self._additional_data,
            how="outer",
            on=[self._datetime_column_name, ForecastOutputColumns.SERIES],
        ).reset_index()

    def _get_data_by_series(self, data, include_horizon=True):
        data_by_series = dict()
        for cat in self.list_categories():
            try:
                data_by_series[cat] = data.xs(
                    cat, level=ForecastOutputColumns.SERIES
                ).reset_index()
            except KeyError as ke:
                logger.debug(
                    f"Unable to extract cat: {cat} from data: {data}. This may occur due to significant missing data. Error message: {e.args}"
                )
                pass
            if not include_horizon:
                data_by_series[cat] = data_by_series[cat][: -self._horizon]
        return data_by_series

    def get_historical_data_by_series(self):
        if self._hist_dd is not None:
            self._hist_dd = self._get_data_by_series(self._data)
        return self._hist_dd

    def get_additional_data_by_series(self, include_horizon=True):
        return self._get_data_by_series(
            self._additional_data, include_horizon=include_horizon
        )

    def get_all_data_by_series(self, include_horizon=True):
        total_dict = dict()
        hist_data = self.get_historical_data_by_series()
        add_data = self.get_additional_data_by_series(include_horizon=include_horizon)
        for cat in self.list_categories():
            # Note: ensure no duplicate column names
            total_dict[cat] = pd.merge(
                hist_data[cat],
                add_data[cat],
                how="outer",
                on=[self._datetime_column_name],
            )
        return total_dict

    def get_historical_data_for_category(self, category):
        hist_data = self.get_historical_data_by_series()
        try:
            return hist_data[category]
        except:
            raise InvalidParameterError(
                f"Unable to retrieve category {category} from historical data. Available categories are: {self.list_categories()}"
            )

    def get_additional_data_for_category(self, category, include_horizon=True):
        add_data = self.get_additional_data_by_series(include_horizon=include_horizon)
        try:
            return add_data[category]
        except:
            raise InvalidParameterError(
                f"Unable to retrieve category {category} from additional data. Available categories are: {self.list_categories()}"
            )

    def get_all_data_for_category(self, category, include_horizon=True):
        all_data = self.get_all_data_by_series(include_horizon=include_horizon)
        try:
            return all_data[category]
        except:
            raise InvalidParameterError(
                f"Unable to retrieve category {category} from data. Available categories are: {self.list_categories()}"
            )

    def get_earliest_timestamp(self):
        return self._min_time

    def get_latest_timestamp(self):
        return self._max_time

    def get_additional_data_column_names(self):
        return self._additional_col_names

    def get_datetime_frequency(self):
        return self._freq

    def get_datetime_frequency_in_seconds(self):
        return self._freq_in_secs

    def get_num_rows(self):
        return self._num_rows

    def list_categories(self):
        return self._categories

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
        self.category_map = dict()
        self.category_to_target = dict()
        self.confidence_interval_width = confidence_interval_width
        self.upper_bound_name = None
        self.lower_bound_name = None

    def add_category(
        self,
        category: str,
        target_category_column: str,
        forecast: pd.DataFrame,
        overwrite: bool = False,
    ):
        if not overwrite and category in self.category_map.keys():
            raise ValueError(
                f"Attempting to update ForecastOutput for category {category} when this already exists. Set overwrite to True."
            )
        forecast = self._check_forecast_format(forecast)
        forecast = self._set_ci_column_names(forecast)
        self.category_map[category] = forecast
        self.category_to_target[category] = target_category_column

    def get_category(self, category):  # change to by_category ?
        return self.category_map[category]

    def get_target_category(self, target_category_column):
        target_category_columns = self.list_target_category_columns()
        category = self.list_categories()[
            list(self.category_to_target.values()).index(target_category_column)
        ]
        return self.category_map[category]

    def list_categories(self):
        return list(self.category_map.keys())

    def list_target_category_columns(self):
        return list(self.category_to_target.values())

    def format_long(self):
        return pd.concat(list(self.category_map.values()))

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

    def format_wide(self):
        dataset_time_indexed = {
            k: v.set_index(ForecastOutputColumns.DATE)
            for k, v in self.category_map.items()
        }
        datasets_category_appended = [
            v.rename(lambda x: str(x) + f"_{k}", axis=1)
            for k, v in dataset_time_indexed.items()
        ]
        return pd.concat(datasets_category_appended, axis=1)

    def get_longest_datetime_column(self):
        return self.format_wide().index

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
