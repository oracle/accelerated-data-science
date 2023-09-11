#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
This exploratory data analysis (EDA) Mixin is used in the ADS accessor for the Pandas Series.
The series of purpose-driven methods enable the data scientist to complete univariate analysis.

From the accessor we have access to the pandas object the user is interacting with as well as
corresponding list of feature types.
"""

import pandas as pd
import matplotlib.pyplot as plt


class EDAMixinSeries:
    def feature_stat(self) -> pd.DataFrame:
        """Summary statistics Dataframe provided.

        This returns feature stats on series using FeatureType summary method.

        Examples
        --------
        >>> df = pd.read_csv('~/advanced-ds/tests/vor_datasets/vor_titanic.csv')
        >>> df['Cabin'].ads.feature_stat()
            Metric      Value
        0	count	    891
        1	unqiue	    147
        2	missing	    687

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe with 2 columns and rows for different metric values

        """
        for feature_type in self._feature_type:
            if hasattr(feature_type, "feature_stat"):
                stat = feature_type.feature_stat(self._obj).reset_index()
                stat.columns = ["Metric", "Value"]
                return stat
        return None

    def feature_plot(self) -> plt.Axes:
        """For the series generate a summary plot based on the most relevant feature type.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
           Plot object for the series based on the most relevant feature type.
        """
        for feature_type in self._feature_type:
            if hasattr(feature_type, "feature_plot"):
                return feature_type.feature_plot(self._obj)
        return None

    def warning(self) -> pd.DataFrame:
        """Generates a data frame that lists feature specific warnings.

        Returns
        -------
        :class:`pandas.DataFrame`
            The list of feature specific warnings.

        Examples
        --------
        >>> df["Age"].ads.warning()
          Feature Type       Warning               Message         Metric      Value
         ---------------------------------------------------------------------------
        0   continuous         Zeros      Age has 38 zeros          Count         38
        1   continuous         Zeros   Age has 12.2% zeros     Percentage      12.2%
        """
        common_columns = ["Feature Type", "Warning", "Message", "Metric", "Value"]
        result_df = pd.DataFrame((), columns=common_columns)
        for feature_type in self._feature_type:
            if hasattr(feature_type, "warning"):
                warning_df = feature_type.warning(self._obj)
                if warning_df is not None:
                    warning_df.insert(0, "Feature Type", feature_type.name)
                    result_df = pd.concat([result_df, warning_df])
        result_df.reset_index(drop=True, inplace=True)
        return result_df
