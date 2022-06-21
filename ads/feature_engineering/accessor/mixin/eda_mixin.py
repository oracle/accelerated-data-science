#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
This exploratory data analysis (EDA) Mixin is used in the ADS accessor for the Pandas Dataframe.
The series of purpose-driven methods enable the data scientist to complete analysis on the dataframe.

From the accessor we have access to the pandas object the user is interacting with as well as
corresponding lists of feature types per column.
"""

import collections
import pandas as pd
import matplotlib.pyplot as plt
from ads.feature_engineering.accessor.mixin.correlation import (
    cat_vs_cat,
    cont_vs_cont,
    cat_vs_cont,
)
from ads.feature_engineering.accessor.mixin.utils import (
    _continuous_columns,
    _categorical_columns,
    _sienna_light_to_dark_color_palette,
)
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class EDAMixin:
    def feature_count(self) -> pd.DataFrame:
        """
        Counts the number of columns for each feature type and each primary feature.
        The column of primary is the number of primary feature types that is assigned to the column.

        Returns
        -------
        Dataframe with
          The number of columns for each feature type
          The number of columns for each primary feature

        Examples
        --------
        >>> df.ads.feature_type
        {'PassengerId': ['ordinal', 'category'],
        'Survived': ['ordinal'],
        'Pclass': ['ordinal'],
        'Name': ['category'],
        'Sex': ['category']}
        >>> df.ads.feature_count()
            Feature Type        Count       Primary
        0       category            3             2
        1        ordinal            3             3
        """

        feature_count = collections.defaultdict(lambda: [0, 0])

        for _, feature_types in self.feature_type.items():
            feature_count[feature_types[0]][1] += 1
            for ft in feature_types:
                feature_count[ft][0] += 1

        return pd.DataFrame(
            [
                (feature_type, count, primary)
                for feature_type, (count, primary) in feature_count.items()
            ],
            columns=["Feature Type", "Count", "Primary"],
        )

    def feature_stat(self) -> pd.DataFrame:
        """Summary statistics Dataframe provided.

        This returns feature stats on each column using FeatureType summary method.

        Examples
        --------
        >>> df = pd.read_csv('~/advanced-ds/tests/vor_datasets/vor_titanic.csv')
        >>> df.ads.feature_stat().head()
                 Column	   Metric	                Value
        0	PassengerId	    count	                891.000
        1	PassengerId	    mean	                446.000
        2	PassengerId	    standard deviation	    257.354
        3	PassengerId	    sample minimum  	    1.000
        4	PassengerId	    lower quartile	        223.500

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe with 3 columns: name, metric, value

        """
        stats = []
        for col_name, col in self._obj.items():
            for _, row in col.ads.feature_stat().iterrows():
                stats.append([col_name, row["Metric"], row["Value"]])
        df_stats = pd.DataFrame(stats, columns=["Column", "Metric", "Value"])
        df_stats.value = df_stats.Value.round(3)
        return df_stats

    def feature_plot(self) -> pd.DataFrame:
        """For every column in the dataframe plot generate a list of summary plots based on the most
        relevant feature type.

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe with 2 columns:
            1. Column - feature name
            2. Plot - plot object
        """
        plots = []
        for _, col in self._obj.items():
            try:
                plot = col.ads.feature_plot()
            except:
                plot = None
            plots.append([col.name, plot])
        return pd.DataFrame(plots, columns=["Column", "Plot"])

    def pearson(self) -> pd.DataFrame:
        """Generate a Pearson correlation data frame for all continuous variable pairs.

        Gives a warning for dropped non-numerical columns.

        Returns
        -------
        :class:`pandas.DataFrame`
        Pearson correlation data frame with the following 3 columns:
            1. Column 1 (name of the first continuous column)
            2. Column 2 (name of the second continuous column)
            3. Value (correlation value)

        Note
        ____
        Pairs will be replicated. For example for variables x and y, we'd have (x,y), (y,x) both with same correlation value. We'll also have (x,x) and (y,y) with value 1.0.
        """
        continuous_cols = _continuous_columns(self._obj.ads.feature_type)
        return cont_vs_cont(self._obj[continuous_cols])

    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def pearson_plot(self) -> plt.Axes:
        """Generate a heatmap of the Pearson correlation for all continuous variable pairs.

        Returns
        -------
        Plot object
            Pearson correlation plot object that can be updated by the customer
        """
        ax = plt.axes()
        df = (
            self.pearson()
            .pivot_table(index="Column 1", columns="Column 2", values="Value")
            .rename_axis("")
            .rename_axis("", axis="columns")
        )
        ax.set_title("Pearson's Correlation")
        return seaborn.heatmap(df, cmap=_sienna_light_to_dark_color_palette(), ax=ax)

    def cramersv(self) -> pd.DataFrame:
        """Generate a Cramer's V correlation data frame for all categorical variable pairs.

        Gives a warning for dropped non-categorical columns.

        Returns
        -------
        :class:`pandas.DataFrame`
            Cramer's V correlation data frame with the following 3 columns:
                1. Column 1 (name of the first categorical column)
                2. Column 2 (name of the second categorical column)
                3. Value (correlation value)
        Note
        ____
        Pairs will be replicated. For example for variables x and y, we would have (x,y), (y,x) both with same correlation value. We will also have (x,x) and (y,y) with value 1.0.
        """
        categorical_cols = _categorical_columns(self._obj.ads.feature_type)
        return cat_vs_cat(self._obj[categorical_cols])

    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def cramersv_plot(self) -> plt.Axes:
        """Generate a heatmap of the Cramer's V correlation for all categorical variable pairs.

        Gives a warning for dropped non-categorical columns.

        Returns
        -------
        Plot object
            Cramer's V correlation plot object that can be updated by the customer
        """
        ax = plt.axes()
        df = (
            self.cramersv()
            .pivot_table(index="Column 1", columns="Column 2", values="Value")
            .rename_axis("")
            .rename_axis("", axis="columns")
        )
        ax.set_title("Cramer's V")
        return seaborn.heatmap(df, cmap=_sienna_light_to_dark_color_palette(), ax=ax)

    def correlation_ratio(self) -> pd.DataFrame:
        """Generate a Correlation Ratio data frame for all categorical-continuous variable pairs.

        Returns
        -------
        :class:`pandas.DataFrame`
        Correlation Ratio correlation data frame with the following 3 columns:
            1. Column 1 (name of the first categorical/continuous column)
            2. Column 2 (name of the second categorical/continuous column)
            3. Value (correlation value)

        Note
        ____
        Pairs will be replicated. For example for variables x and y, we would have (x,y), (y,x) both with same correlation value. We will also have (x,x) and (y,y) with value 1.0.
        """
        categorical_cols = _categorical_columns(self._obj.ads.feature_type)
        continuous_cols = _continuous_columns(self._obj.ads.feature_type)
        return cat_vs_cont(self._obj, categorical_cols, continuous_cols)

    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def correlation_ratio_plot(self) -> plt.Axes:
        """Generate a heatmap of the Correlation Ratio correlation for all categorical-continuous variable
        pairs.

        Returns
        -------
        Plot object
            Correlation Ratio correlation plot object that can be updated by the customer
        """
        ax = plt.axes()
        df = (
            self.correlation_ratio()
            .pivot_table(index="Column 1", columns="Column 2", values="Value")
            .rename_axis("")
            .rename_axis("", axis="columns")
        )
        ax.set_title("Correlation Ratio")
        return seaborn.heatmap(df, cmap=_sienna_light_to_dark_color_palette(), ax=ax)

    def warning(self) -> pd.DataFrame:
        """Generates a data frame that lists feature specific warnings.

        Returns
        -------
        :class:`pandas.DataFrame`
            The list of feature specific warnings.

        Examples
        --------
        >>> df.ads.warning()
            Column    Feature Type         Warning               Message       Metric    Value
        --------------------------------------------------------------------------------------
        0      Age      continuous           Zeros      Age has 38 zeros        Count       38
        1      Age      continuous           Zeros   Age has 12.2% zeros   Percentage    12.2%
        """
        common_columns = ["Feature Type", "Warning", "Message", "Metric", "Value"]
        result_df = pd.DataFrame((), columns=["Column"] + common_columns)
        for col in self._obj.columns:
            warning_df = self._obj[col].ads.warning()
            if warning_df is not None:
                warning_df.insert(0, "Column", col)
                result_df = result_df.append(warning_df)
        return result_df.reset_index(drop=True)
