#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents an Ordinal feature type.

Classes:
    Ordinal
        The Ordinal feature type.
"""
import matplotlib.pyplot as plt
import pandas as pd
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.utils import (
    _count_unique_missing,
    _set_seaborn_theme,
    SchemeTeal,
)
from ads.feature_engineering import schema
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class Ordinal(FeatureType):
    """Type representing ordered values.

    Attributes
    ----------
    description: str
        The feature type description.
    name: str
        The feature type name.
    warning: FeatureWarning
        Provides functionality to register warnings and invoke them.
    validator
        Provides functionality to register validators and invoke them.

    Methods
    --------
    feature_stat(x: pd.Series) -> pd.DataFrame
        Generates feature statistics.
    feature_plot(x: pd.Series) -> plt.Axes
        Shows the counts of observations in each categorical bin using bar chart.
    """

    description = "Type representing ordered values."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, unique(count),
        and missing(count) if there is any.

        Examples
        --------
        >>> x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan], name='ordinal')
        >>> x.ads.feature_type = ['ordinal']
        >>> x.ads.feature_stat()
            Metric  Value
        0	count	10
        1	unique	9
        2	missing	1

        Returns
        -------
        :class:`pandas.DataFrame`
            Summary statistics of the Series or Dataframe provided.
        """
        return _count_unique_missing(x)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows the counts of observations in each categorical bin using bar chart.

        Examples
        --------
        >>> x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan], name='ordinal')
        >>> x.ads.feature_type = ['ordinal']
        >>> x.ads.feature_plot()

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            The bart chart plot object for the series based on the Continuous feature type.
        """
        col_name = x.name if x.name else "ordinal"
        df = x.to_frame(name=col_name)
        df = df.dropna()
        if len(df.index):
            _set_seaborn_theme()
            ax = seaborn.countplot(y=col_name, data=df, color=SchemeTeal.AREA_DARK)
            ax.set(xlabel="Count")
            return ax

    @classmethod
    def feature_domain(cls, x: pd.Series) -> schema.Domain:
        """
        Generate the domain of the data of this feature type.

        Examples
        --------
        >>> x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan], name='ordinal')
        >>> x.ads.feature_type = ['ordinal']
        >>> x.ads.feature_domain()
        constraints:
        - expression: $x in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            language: python
        stats:
            count: 10
            missing: 1
            unique: 9
        values: Ordinal

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the Ordinal feature type.
        """

        return schema.Domain(
            cls.__name__,
            cls.feature_stat(x).to_dict()[x.name],
            [schema.Expression(f"$x in {x.dropna().unique().tolist()}")],
        )
