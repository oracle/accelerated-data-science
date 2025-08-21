#!/usr/bin/env python

# Copyright (c) 2023, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import ABC

import numpy as np
import pandas as pd

from ads.opctl import logger
from ads.opctl.operator.lowcode.common.const import DataColumns
from ads.opctl.operator.lowcode.common.errors import (
    DataMismatchError,
    InvalidParameterError,
)
from ads.opctl.operator.lowcode.common.utils import merge_category_columns
from ads.opctl.operator.lowcode.forecast.operator_config import ForecastOperatorSpec


class Transformations(ABC):
    """A class which implements transformation for forecast operator"""

    def __init__(self, dataset_info, name="historical_data"):
        """
        Initializes the transformation.

        Parameters
        ----------
            data: The Pandas DataFrame.
            dataset_info : ForecastOperatorConfig
        """
        self.name = name
        self.dataset_info = dataset_info
        self.target_category_columns = dataset_info.target_category_columns
        self.target_column_name = dataset_info.target_column
        self.raw_column_names = None
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
        if isinstance(self.dataset_info, ForecastOperatorSpec):
            clean_df = self._clean_column_names(clean_df)
        if self.name == "historical_data":
            self._check_historical_dataset(clean_df)
        clean_df = self._set_series_id_column(clean_df)
        if self.dt_column_name:
            clean_df = self._format_datetime_col(clean_df)
        clean_df = self._set_multi_index(clean_df)
        clean_df = self._fill_na(clean_df) if not self.dt_column_name else clean_df

        if self.preprocessing and self.preprocessing.enabled:
            if self.name == "historical_data":
                if self.preprocessing.steps.missing_value_imputation:
                    try:
                        clean_df = self._missing_value_imputation_hist(clean_df)
                    except Exception as e:
                        logger.debug(f"Missing value imputation failed with {e.args}")
                else:
                    logger.info(
                        "Skipping missing value imputation because it is disabled"
                    )
                if self.preprocessing.steps.outlier_treatment:
                    try:
                        clean_df = self._outlier_treatment(clean_df)
                    except Exception as e:
                        logger.debug(f"Outlier Treatment failed with {e.args}")
                else:
                    logger.info("Skipping outlier treatment because it is disabled")
            elif self.name == "additional_data":
                clean_df = self._missing_value_imputation_add(clean_df)
            elif self.name == "input_data" and self.preprocessing.steps.missing_value_imputation:
                clean_df = self._fill_na(clean_df)
        else:
            logger.info(
                "Skipping all preprocessing steps because preprocessing is disabled"
            )
        return clean_df

    def _remove_trailing_whitespace(self, df):
        return df.apply(
            lambda x: x.str.strip()
            if hasattr(x, "dtype") and x.dtype == "object"
            else x
        )

    def _clean_column_names(self, df):
        """
        Remove all whitespaces from column names in a DataFrame and store the original names.

        Parameters:
        df (pd.DataFrame): The DataFrame whose column names need to be cleaned.

        Returns:
        pd.DataFrame: The DataFrame with cleaned column names.
        """

        self.raw_column_names = {
            col: col.replace(" ", "") for col in df.columns if " " in col
        }
        df.columns = [self.raw_column_names.get(col, col) for col in df.columns]

        if self.target_column_name:
            self.target_column_name = self.raw_column_names.get(
                self.target_column_name, self.target_column_name
            )
        self.dt_column_name = self.raw_column_names.get(
            self.dt_column_name, self.dt_column_name
        )

        if self.target_category_columns:
            self.target_category_columns = [
                self.raw_column_names.get(col, col)
                for col in self.target_category_columns
            ]
        return df

    def _set_series_id_column(self, df):
        self._target_category_columns_map = {}
        if not self.target_category_columns:
            df[DataColumns.Series] = "Series 1"
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
        except Exception as ee:
            raise InvalidParameterError(
                f"Unable to determine the datetime type for column: {self.dt_column_name} in dataset: {self.name}. Please specify the format explicitly. (For example adding 'format: %d/%m/%Y' underneath 'name: {self.dt_column_name}' in the datetime_column section of the yaml file if you haven't already. For reference, here is the first datetime given: {df[self.dt_column_name].values[0]}"
            ) from ee
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

    def _missing_value_imputation_hist(self, df):
        """
        Function fills missing values in the pandas dataframe using liner interpolation

        Parameters
        ----------
            df : The Pandas DataFrame.

        Returns
        -------
            A new Pandas DataFrame without missing values.
        """
        # missing value imputation using linear interpolation
        df[self.target_column_name] = (
            df[self.target_column_name]
            .groupby(DataColumns.Series)
            .transform(lambda x: x.interpolate(limit_direction="both"))
        )
        return df

    def _missing_value_imputation_add(self, df):
        """
        Function fills missing values with zero

        Parameters
        ----------
            df : The Pandas DataFrame.

        Returns
        -------
            A new Pandas DataFrame without missing values.
        """
        return df.fillna(0)

    def _outlier_treatment(self, df):
        """
        Function finds outliears using z_score and treats with mean value.

        Parameters
        ----------
            df : The Pandas DataFrame.

        Returns
        -------
            A new Pandas DataFrame with treated outliears.
        """
        return df
        df["__z_score"] = (
            df[self.target_column_name]
            .groupby(DataColumns.Series)
            .transform(lambda x: (x - x.mean()) / x.std())
        )
        outliers_mask = df["__z_score"].abs() > 3

        if df[self.target_column_name].dtype == np.int:
            df[self.target_column_name].astype(np.float)

        df.loc[outliers_mask, self.target_column_name] = (
            df[self.target_column_name]
            .groupby(DataColumns.Series)
            .transform(lambda x: np.median(x))
        )
        df_ret = df.drop("__z_score", axis=1)
        return df_ret

    def _check_historical_dataset(self, df):
        expected_names = [self.target_column_name, self.dt_column_name] + (
            self.target_category_columns if self.target_category_columns else []
        )

        if self.raw_column_names:
            expected_names.extend(list(self.raw_column_names.values()))

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

    def build_fforms_meta_features(self, data, target_col=None, group_cols=None):
        """
        Build meta-features for time series based on FFORMS paper and add them to the original DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame containing time series data
        target_col : str, optional
            Name of the target column to calculate meta-features for.
            If None, uses the target column specified in dataset_info.
        group_cols : list of str, optional
            List of columns to group by before calculating meta-features.
            If None, calculates features for the entire series.

        Returns
        -------
        pandas.DataFrame
            Original DataFrame with additional meta-feature columns

        References
        ----------
        Talagala, T. S., Hyndman, R. J., & Athanasopoulos, G. (2023).
        Meta-learning how to forecast time series. Journal of Forecasting, 42(6), 1476-1501.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Use target column from dataset_info if not specified
        if target_col is None:
            target_col = self.target_column_name
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        # Check if group_cols are provided and valid
        if group_cols is not None:
            if not isinstance(group_cols, list):
                raise ValueError("group_cols must be a list of column names")
            for col in group_cols:
                if col not in data.columns:
                    raise ValueError(f"Group column '{col}' not found in DataFrame")

        # If no group_cols, get the target_category_columns else treat the entire DataFrame as a single series
        if not group_cols:
            group_cols = self.target_category_columns if self.target_category_columns else []

        # Calculate meta-features for each series
        def calculate_series_features(series):
            """Calculate features for a single series"""
            n = len(series)
            values = series.values

            # Basic statistics
            mean = series.mean()
            std = series.std()
            variance = series.var()
            skewness = series.skew()
            kurtosis = series.kurtosis()
            cv = std / mean if mean != 0 else np.inf

            # Trend features
            X = np.vstack([np.arange(n), np.ones(n)]).T
            trend_coef = np.linalg.lstsq(X, values, rcond=None)[0][0]
            trend_pred = X.dot(np.linalg.lstsq(X, values, rcond=None)[0])
            residuals = values - trend_pred
            std_residuals = np.std(residuals)

            # Turning points
            turning_points = 0
            for i in range(1, n-1):
                if (values[i-1] < values[i] and values[i] > values[i+1]) or \
                   (values[i-1] > values[i] and values[i] < values[i+1]):
                    turning_points += 1
            turning_points_rate = turning_points / (n-2) if n > 2 else 0

            # Serial correlation
            acf1 = series.autocorr(lag=1) if n > 1 else 0
            acf2 = series.autocorr(lag=2) if n > 2 else 0
            acf10 = series.autocorr(lag=10) if n > 10 else 0

            # Seasonality features
            seasonal_strength = 0
            seasonal_peak_strength = 0
            if n >= 12:
                seasonal_lags = [12, 24, 36]
                seasonal_acfs = []
                for lag in seasonal_lags:
                    if n > lag:
                        acf_val = series.autocorr(lag=lag)
                        seasonal_acfs.append(abs(acf_val))
                seasonal_peak_strength = max(seasonal_acfs) if seasonal_acfs else 0

                ma = series.rolling(window=12, center=True).mean()
                seasonal_comp = series - ma
                seasonal_strength = 1 - np.var(seasonal_comp.dropna()) / np.var(series)

            # Stability and volatility features
            values_above_mean = values >= mean
            crossing_points = np.sum(values_above_mean[1:] != values_above_mean[:-1])
            crossing_rate = crossing_points / (n - 1) if n > 1 else 0

            # First and second differences
            diff1 = np.diff(values)
            diff2 = np.diff(diff1) if len(diff1) > 1 else np.array([])

            diff1_mean = np.mean(np.abs(diff1)) if len(diff1) > 0 else 0
            diff1_var = np.var(diff1) if len(diff1) > 0 else 0
            diff2_mean = np.mean(np.abs(diff2)) if len(diff2) > 0 else 0
            diff2_var = np.var(diff2) if len(diff2) > 0 else 0

            # Nonlinearity features
            if n > 3:
                X = values[:-1].reshape(-1, 1)
                y = values[1:]
                X2 = X * X
                X3 = X * X * X
                X_aug = np.hstack([X, X2, X3])
                nonlinearity = np.linalg.lstsq(X_aug, y, rcond=None)[1][0] if len(y) > 0 else 0
            else:
                nonlinearity = 0

            # Long-term trend features
            if n >= 10:
                mid = n // 2
                trend_change = np.mean(values[mid:]) - np.mean(values[:mid])
            else:
                trend_change = 0

            # Step changes and spikes
            step_changes = np.abs(diff1).max() if len(diff1) > 0 else 0
            spikes = np.sum(np.abs(values - mean) > 2 * std) / n if std != 0 else 0

            # Hurst exponent and entropy
            lag = min(10, n // 2)
            variance_ratio = np.var(series.diff(lag)) / (lag * np.var(series.diff())) if n > lag else 0
            hurst = np.log(variance_ratio) / (2 * np.log(lag)) if variance_ratio > 0 and lag > 1 else 0

            hist, _ = np.histogram(series, bins='auto', density=True)
            entropy = -np.sum(hist[hist > 0] * np.log(hist[hist > 0]))

            return pd.Series({
                'ts_n_obs': n,
                'ts_mean': mean,
                'ts_std': std,
                'ts_variance': variance,
                'ts_cv': cv,
                'ts_skewness': skewness,
                'ts_kurtosis': kurtosis,
                'ts_trend': trend_coef,
                'ts_trend_change': trend_change,
                'ts_std_residuals': std_residuals,
                'ts_turning_points_rate': turning_points_rate,
                'ts_seasonal_strength': seasonal_strength,
                'ts_seasonal_peak_strength': seasonal_peak_strength,
                'ts_acf1': acf1,
                'ts_acf2': acf2,
                'ts_acf10': acf10,
                'ts_crossing_rate': crossing_rate,
                'ts_diff1_mean': diff1_mean,
                'ts_diff1_variance': diff1_var,
                'ts_diff2_mean': diff2_mean,
                'ts_diff2_variance': diff2_var,
                'ts_nonlinearity': nonlinearity,
                'ts_step_max': step_changes,
                'ts_spikes_rate': spikes,
                'ts_hurst': hurst,
                'ts_entropy': entropy
            })

        # Create copy of input DataFrame
        result_df = data.copy()

        if group_cols:
            # Calculate features for each group
            features = []
            # Sort by date within each group if date column exists
            date_col = self.dt_column_name if self.dt_column_name else 'Date'
            if date_col in data.columns:
                data = data.sort_values([date_col] + group_cols)

            for name, group in data.groupby(group_cols):
                # Sort group by date if exists
                if date_col in group.columns:
                    group = group.sort_values(date_col)
                group_features = calculate_series_features(group[target_col])
                if isinstance(name, tuple):
                    feature_row = dict(zip(group_cols, name))
                else:
                    feature_row = {group_cols[0]: name}
                feature_row.update(group_features)
                features.append(feature_row)

            # Create features DataFrame without merging
            features_df = pd.DataFrame(features)
            # Return only the meta-features DataFrame with group columns
            return features_df
        else:
            # Sort by date if exists and calculate features for entire series
            date_col = self.dt_column_name if self.dt_column_name else 'Date'
            if date_col in data.columns:
                data = data.sort_values(date_col)
            features = calculate_series_features(data[target_col])
            # Return single row DataFrame with meta-features
            return pd.DataFrame([features])

        return result_df
