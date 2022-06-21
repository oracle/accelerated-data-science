#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a Category feature type.

Classes:
    Category
        The Category feature type.
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


class Category(FeatureType):
    """
    Type representing discrete unordered values.

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

    description = "Type representing discrete unordered values."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, unique(count) and missing(count) if there are any.

        Parameters
        ----------
        x : :class:`pandas.Series`
            The feature being evaluated.

        Returns
        -------
        :class:`pandas.DataFrame`
            Summary statistics of the Series or Dataframe provided.

        Examples
        --------
        >>> cat = pd.Series(['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C', 'S', 'S', 'S',
                    'S', 'S', 'S', 'Q', 'S', 'S', '', np.NaN, None], name='сategory')
        >>> cat.ads.feature_type = ['сategory']
        >>> cat.ads.feature_stat()
            Metric  Value
        0	count	22
        1	unique	3
        2	missing	3
        """
        return _count_unique_missing(x)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows the counts of observations in each categorical bin using bar chart.

        Parameters
        ----------
        x : :class:`pandas.Series`
            The feature being evaluated.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the Category feature type.

        Examples
        --------
        >>> cat = pd.Series(['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C', 'S', 'S', 'S',
                    'S', 'S', 'S', 'Q', 'S', 'S', '', np.NaN, None], name='сategory')
        >>> cat.ads.feature_type = ['сategory']
        >>> cat.ads.feature_plot()
        """
        col_name = x.name if x.name else "сategory"
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
        >>> cat = pd.Series(['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C', 'S', 'S', 'S',
                    'S', 'S', 'S', 'Q', 'S', 'S', '', np.NaN, None], name='category')
        >>> cat.ads.feature_type = ['category']
        >>> cat.ads.feature_domain()
        constraints:
        - expression: $x in ['S', 'C', 'Q', '']
            language: python
        stats:
            count: 22
            missing: 3
            unique: 3
        values: Category

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the Category feature type.
        """

        return schema.Domain(
            cls.__name__,
            cls.feature_stat(x).to_dict()[x.name],
            [schema.Expression(f"$x in {x.dropna().unique().tolist()}")],
        )
