#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl import logger


class Transformations:
    """A class which implements transformation for forecast operator"""

    def __init__(self, data, dataset_info):
        """
        Initializes the transformation.

        Parameters
        ----------
            data: The Pandas DataFrame.
            dataset_info : ForecastOperatorConfig
        """
        self.data = data
        self.dataset_info = dataset_info
        self._set_series_id_column()
        self.series_id_column = self.dataset_info.target_category_columns
        self.target_variables = dataset_info.target_column
        self.date_column = dataset_info.datetime_column.name
        self.date_format = dataset_info.datetime_column.format
        self.preprocessing = dataset_info.preprocessing

    def run(self):
        """
        The function runs all the transformation in a particular order.

        Returns
        -------
            A new Pandas DataFrame with treated / transformed target values.
        """
        imputed_df = self._missing_value_imputation(self.data)
        sorted_df = self._sort_by_datetime_col(imputed_df)
        clean_strs_df = self._remove_trailing_whitespace(sorted_df)
        if self.preprocessing:
            treated_df = self._outlier_treatment(clean_strs_df)
        else:
            logger.debug("Skipping outlier treatment as preprocessing is disabled")
            treated_df = imputed_df
        return treated_df

    def _set_series_id_column(self):
        if (
            self.dataset_info.target_category_columns is None
            or len(self.dataset_info.target_category_columns) == 0
        ):
            self.data["__Series"] = ""
            self.dataset_info.target_category_columns = ["__Series"]

    def _remove_trailing_whitespace(self, df):
        return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    def _missing_value_imputation(self, df):
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
        df[self.target_variables] = df.groupby(self.series_id_column)[
            self.target_variables
        ].transform(lambda x: x.interpolate(limit_direction="both"))
        return df

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
        df["z_score"] = df.groupby(self.series_id_column)[
            self.target_variables
        ].transform(lambda x: (x - x.mean()) / x.std())
        outliers_mask = df["z_score"].abs() > 3
        df.loc[outliers_mask, self.target_variables] = df.groupby(
            self.series_id_column
        )[self.target_variables].transform(lambda x: x.mean())
        df.drop("z_score", axis=1, inplace=True)
        return df

    def _sort_by_datetime_col(self, df):
        """
        Function sorts by date

        Parameters
        ----------
            df : The Pandas DataFrame.

        Returns
        -------
            A new Pandas DataFrame with sorted dates for each category
        """
        import pandas as pd

        # Temporary column for sorting
        df["tmp_col_for_sorting"] = pd.to_datetime(
            df[self.date_column], format=self.date_format
        )
        df = (
            df.groupby(self.series_id_column, group_keys=True)
            .apply(lambda x: x.sort_values(by="tmp_col_for_sorting", ascending=True))
            .reset_index(drop=True)
        )
        # Drop the temporary column
        df.drop(columns=["tmp_col_for_sorting"], inplace=True)
        return df
