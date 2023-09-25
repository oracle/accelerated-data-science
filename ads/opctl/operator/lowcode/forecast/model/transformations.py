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
        self.series_id_column = dataset_info.target_category_columns
        self.target_variables = dataset_info.target_column
        self.date_column = dataset_info.datetime_column.name
        self.preprocessing = dataset_info.preprocessing

    def run(self):
        """
        The function runs all the transformation in a particular order.

        Returns
        -------
            A new Pandas DataFrame with treated / transformed target values.
        """
        imputed_df = self._missing_value_imputation(self.data)
        if self.preprocessing:
            treated_df = self._outlier_treatment(imputed_df)
        else:
            logger.info("Skipping outlier treatment as preprocessing is disabled")
            treated_df = imputed_df
        return treated_df

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
        df[self.target_variables] = df.groupby(self.series_id_column)[self.target_variables].transform(
            lambda x: x.interpolate(limit_direction='both'))
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
        df['z_score'] = df.groupby(self.series_id_column)[self.target_variables].transform(
            lambda x: (x - x.mean()) / x.std())
        outliers_mask = df['z_score'].abs() > 3
        df.loc[outliers_mask, self.target_variables] = df.groupby(self.series_id_column)[
            self.target_variables].transform(lambda x: x.mean())
        df.drop('z_score', axis=1, inplace=True)
        return df
