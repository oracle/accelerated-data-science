#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl import logger
from ads.opctl.operator.lowcode.common.errors import (
    InvalidParameterError,
    DataMismatchError,
)
from ads.opctl.operator.lowcode.common.const import DataColumns, OutlierTreatmentMethods
from ads.opctl.operator.lowcode.common.utils import merge_category_columns
from ads.opctl.operator.lowcode.common.const import ImputationMethods
import pandas as pd
from abc import ABC


class Transformations(ABC):
    """A class which implements transformation for operator"""

    def __init__(self, dataset_info, name="historical_data"):
        """
        Initializes the transformation.

        Parameters
        ----------
            data: The Pandas DataFrame.
            dataset_info : ForecastOperatorConfig
        """
        self.name = name
        self.has_artificial_series = False
        self.dataset_info = dataset_info
        self.target_category_columns = dataset_info.target_category_columns
        self.target_column_name = dataset_info.target_column
        self.dt_column_name = (
            dataset_info.datetime_column.name if dataset_info.datetime_column else None
        )
        self.dt_column_format = (
            dataset_info.datetime_column.format
            if dataset_info.datetime_column
            else None
        )
        self.preprocessing = dataset_info.preprocessing

    def run(self, data):
        """
        The function runs all the transformation in a particular order.

        Returns
        -------
            A new Pandas DataFrame with treated / transformed target values. Specifically:
            - Data will be in a multiIndex with Datetime always first (level 0)
            - whether 0, 1 or 2+, all target_category_columns will be merged into a single index column: Series
            - All datetime columns will be formatted as such
            - all data will be imputed (unless preprocessing disabled)
            - all trailing whitespace will be removed
            - the data will be sorted by Datetime then Series

        """
        clean_df = self._remove_trailing_whitespace(data)
        if self.name == "historical_data":
            self._check_historical_dataset(clean_df)
        clean_df = self._set_series_id_column(clean_df)
        if self.dt_column_name:
            clean_df = self._format_datetime_col(clean_df)
        clean_df = self._set_multi_index(clean_df)
        clean_df = self._fill_na(clean_df) if not self.dt_column_name else clean_df
        # preprocessing steps are not supported for additional data
        if self.name == "historical_data":
            mvi_method = self.preprocessing.missing_value_imputation
            try:
                mvi = MissingValueImputer(clean_df, self.target_column_name)
                clean_df = mvi.impute(mvi_method)
            except Exception as e:
                logger.debug(f"Missing value imputation failed with {e.args}")
            ot_method = self.preprocessing.outlier_treatment
            try:
                ot = OutlierTreatment(clean_df, self.target_column_name)
                clean_df = ot.outlier_treatment(ot_method)
            except Exception as e:
                logger.debug(f"Outlier Treatment failed with {e.args}")
        return clean_df

    def _remove_trailing_whitespace(self, df):
        return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    def _set_series_id_column(self, df):
        self._target_category_columns_map = dict()
        if not self.target_category_columns:
            df[DataColumns.Series] = "Series 1"
            self.has_artificial_series = True
        else:
            df[DataColumns.Series] = merge_category_columns(
                df, self.target_category_columns
            )
            merged_values = df[DataColumns.Series].unique().tolist()
            if self.target_category_columns:
                for value in merged_values:
                    self._target_category_columns_map[value] = (
                        df[df[DataColumns.Series] == value][
                            self.target_category_columns
                        ]
                        .drop_duplicates()
                        .iloc[0]
                        .to_dict()
                    )

            if self.target_category_columns != [DataColumns.Series]:
                df = df.drop(self.target_category_columns, axis=1)
        return df

    def _format_datetime_col(self, df):
        try:
            df[self.dt_column_name] = pd.to_datetime(
                df[self.dt_column_name], format=self.dt_column_format
            )
        except:
            raise InvalidParameterError(
                f"Unable to determine the datetime type for column: {self.dt_column_name} in dataset: {self.name}. Please specify the format explicitly. (For example adding 'format: %d/%m/%Y' underneath 'name: {self.dt_column_name}' in the datetime_column section of the yaml file if you haven't already. For reference, here is the first datetime given: {df[self.dt_column_name].values[0]}"
            )
        return df

    def _set_multi_index(self, df):
        """
        Function sorts by date

        Parameters
        ----------
            df : The Pandas DataFrame.

        Returns
        -------
            A new Pandas DataFrame with sorted dates for each series
        """
        if self.dt_column_name:
            df = df.set_index([self.dt_column_name, DataColumns.Series])
            return df.sort_values(
                [self.dt_column_name, DataColumns.Series], ascending=True
            )
        return df.set_index([df.index, DataColumns.Series])


    def _check_historical_dataset(self, df):
        expected_names = [self.target_column_name, self.dt_column_name] + (
            self.target_category_columns if self.target_category_columns else []
        )
        if set(df.columns) != set(expected_names):
            raise DataMismatchError(
                f"Expected {self.name} to have columns: {expected_names}, but instead found column names: {df.columns}. Is the {self.name} path correct?"
            )

    """
        Map between merged target category column values and target category column and its value
        If target category columns are PPG_Code, Class, Num
        Merged target category column values are Product Category 1__A__1, Product Category 2__A__2
        Then target_category_columns_map would be
        {
            "Product Category 1__A__1": {
                "PPG_Code": "Product Category 1",
                "Class": "A",
                "Num": 1
            },
             "Product Category 2__A__2": {
                "PPG_Code": "Product Category 2",
                "Class": "A",
                "Num": 2
            },
            
        }
    """

    def get_target_category_columns_map(self):
        return self._target_category_columns_map

    def _fill_na(self, df: pd.DataFrame, na_value=0) -> pd.DataFrame:
        """Fill nans in dataframe"""
        return df.fillna(value=na_value)


class MissingValueImputer:
    def __init__(self, data, target_column_name):
        self.data = data
        self.target_column_name = target_column_name

    def impute(self, method):
        """
        Impute missing values based on the given method.
        """
        self.data[self.target_column_name] = (
            self.data[self.target_column_name]
            .groupby(DataColumns.Series)
            .transform(lambda x: self._apply_imputation(x, method))
        )
        return self.data

    @staticmethod
    def _apply_imputation(x, method):
        """
        provide methods for imputation.
        """
        if method == ImputationMethods.LINEAR_INTERPOLATION or method is None:
            return x.interpolate(limit_direction="both")
        elif method == ImputationMethods.MEAN:
            return x.fillna(x.mean())
        elif method == ImputationMethods.MEDIAN:
            return x.fillna(x.median())
        elif method == "none":
            return x
        else:
            raise ValueError(f"Unknown method for missing value imputation: {method}")


class OutlierTreatment:
    def __init__(self, data, target_column_name):
        self.data = data
        self.target_column_name = target_column_name

    def outlier_treatment(self, method):
        """
        Function finds outliers using z_score and treats with mean value.
        """
        if method == OutlierTreatmentMethods.ZSCORE_WITH_MEAN or method is None:
            self.data["z_score"] = (
                self.data[self.target_column_name]
                .groupby(DataColumns.Series)
                .transform(lambda x: (x - x.mean()) / x.std())
            )
            outliers_mask = self.data["z_score"].abs() > 3
            self.data.loc[outliers_mask, self.target_column_name] = (
                self.data[self.target_column_name]
                .groupby(DataColumns.Series)
                .transform(lambda x: x.mean())
            )
            return self.data.drop("z_score", axis=1)
        elif method == OutlierTreatmentMethods.NONE:
            return self.data
        else:
            raise ValueError(f"Unknown method for outlier treatment: {method}")
