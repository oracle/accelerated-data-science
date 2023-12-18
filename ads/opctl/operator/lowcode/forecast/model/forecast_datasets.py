#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
from ..operator_config import ForecastOperatorConfig
from .. import utils
from .transformations import Transformations
from ads.opctl import logger
import pandas as pd
from ..const import ForecastOutputColumns, PROPHET_INTERNAL_DATE_COL
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype, is_numeric_dtype


class ForecastDatasets:
    def __init__(self, config: ForecastOperatorConfig):
        """Instantiates the DataIO instance.

        Properties
        ----------
        config: ForecastOperatorConfig
            The forecast operator configuration.
        """
        self.original_user_data = None
        self.original_total_data = None
        self.original_additional_data = None
        self.full_data_dict = None
        self.target_columns = None
        self.categories = None
        self.datetime_col = PROPHET_INTERNAL_DATE_COL
        self.datetime_format = config.spec.datetime_column.format
        self._load_data(config.spec)

    def _load_data(self, spec):
        """Loads forecasting input data."""

        raw_data = utils._load_data(
            filename=spec.historical_data.url,
            format=spec.historical_data.format,
            columns=spec.historical_data.columns,
        )
        self.original_user_data = raw_data.copy()
        data_transformer = Transformations(raw_data, spec)
        data = data_transformer.run()
        try:
            spec.freq = utils.get_frequency_of_datetime(data, spec)
        except TypeError as e:
            logger.warn(
                f"Error determining frequency: {e.args}. Setting Frequency to None"
            )
            logger.debug(f"Full traceback: {e}")
            spec.freq = None

        self.original_total_data = data
        additional_data = None

        try:
            data[spec.datetime_column.name] = pd.to_datetime(
                data[spec.datetime_column.name], format=self.datetime_format
            )
        except:
            raise ValueError(
                f"Unable to determine the datetime type for column: {spec.datetime_column.name}. Please specify the format explicitly."
            )

        if spec.additional_data is not None:
            additional_data = utils._load_data(
                filename=spec.additional_data.url,
                format=spec.additional_data.format,
                columns=spec.additional_data.columns,
            )
            additional_data = data_transformer._sort_by_datetime_col(additional_data)
            try:
                additional_data[spec.datetime_column.name] = pd.to_datetime(
                    additional_data[spec.datetime_column.name],
                    format=self.datetime_format,
                )
            except:
                raise ValueError(
                    f"Unable to determine the datetime type for column: {spec.datetime_column.name}. Please specify the format explicitly."
                )

            self.original_additional_data = additional_data.copy()
            self.original_total_data = pd.concat([data, additional_data], axis=1)
        else:
            # Need to add the horizon to the data for compatibility
            additional_data_small = data[
                [spec.datetime_column.name] + spec.target_category_columns
            ].set_index(spec.datetime_column.name)
            if is_datetime64_any_dtype(additional_data_small.index):
                horizon_index = pd.date_range(
                    start=additional_data_small.index.values[-1],
                    freq=spec.freq,
                    periods=spec.horizon + 1,
                )[1:]
            elif is_numeric_dtype(additional_data_small.index):
                # If datetime column is just ints
                assert (
                    len(additional_data_small.index.values) > 1
                ), "Dataset is too small to infer frequency. Please pass in the horizon explicitly through the additional data."
                start = additional_data_small.index.values[-1]
                step = (
                    additional_data_small.index.values[-1]
                    - additional_data_small.index.values[-2]
                )
                horizon_index = pd.RangeIndex(
                    start, start + step * (spec.horizon + 1), step=step
                )[1:]
            else:
                raise ValueError(
                    f"Unable to determine the datetime type for column: {spec.datetime_column.name}. Please specify the format explicitly."
                )

            additional_data = pd.DataFrame()

            for cat_col in spec.target_category_columns:
                for cat in additional_data_small[cat_col].unique():
                    add_data_i = additional_data_small[
                        additional_data_small[cat_col] == cat
                    ]
                    horizon_df_i = pd.DataFrame([], index=horizon_index)
                    horizon_df_i[cat_col] = cat
                    additional_data = pd.concat(
                        [additional_data, add_data_i, horizon_df_i]
                    )
            additional_data = additional_data.reset_index().rename(
                {"index": spec.datetime_column.name}, axis=1
            )

            self.original_total_data = pd.concat([data, additional_data], axis=1)

        (
            self.full_data_dict,
            self.target_columns,
            self.categories,
        ) = utils._build_indexed_datasets(
            data=data,
            target_column=spec.target_column,
            datetime_column=spec.datetime_column.name,
            horizon=spec.horizon,
            target_category_columns=spec.target_category_columns,
            additional_data=additional_data,
        )
        if spec.generate_explanations:
            if spec.additional_data is None:
                logger.warn(
                    f"Unable to generate explanations as there is no additional data passed in. Either set generate_explanations to False, or pass in additional data."
                )
                spec.generate_explanations = False

    def format_wide(self):
        data_merged = pd.concat(
            [
                v[v[k].notna()].set_index(self.datetime_col)
                for k, v in self.full_data_dict.items()
            ],
            axis=1,
        ).reset_index()
        return data_merged

    def get_longest_datetime_column(self):
        return pd.to_datetime(
            self.format_wide()[self.datetime_col], format=self.datetime_format
        )


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
