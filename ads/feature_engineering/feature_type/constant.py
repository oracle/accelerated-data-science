#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a Constant feature type.

Classes:
    Constant
        The Constant feature type.
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


class Constant(FeatureType):
    """
    Type representing constant values.

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
        Shows the counts of observations in bars.
    """

    description = "Type representing constant values."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, unique(count) and missing(count).

        Parameters
        ----------
        x : :class:`pandas.Series`
            The feature being evaluated.

        Returns
        -------
        :class:`pandas.DataFrame`
            Summary statistics of the Series provided.

        Examples
        --------
        >>> s = pd.Series([1, 1, 1, 1, 1], name='constant')
        >>> s.ads.feature_type = ['constant']
        >>> s.ads.feature_stat()
            Metric  Value
        0	count	5
        1	unique	1
        """
        return _count_unique_missing(x)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows the counts of observations in bars.

        Parameters
        ----------
        x : :class:`pandas.Series`
            The feature being shown.

        Examples
        --------
        >>> s = pd.Series([1, 1, 1, 1, 1], name='constant')
        >>> s.ads.feature_type = ['constant']
        >>> s.ads.feature_plot()

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the Constant feature type.
        """
        col_name = x.name if x.name else "constant"
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
        Example
        -------
        >>> s = pd.Series([1, 1, 1, 1, 1], name='constant')
        >>> s.ads.feature_type = ['constant']
        >>> s.ads.feature_domain()
        constraints: []
        stats:
            count: 5
            unique: 1
        values: Constant

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the Constant feature type.
        """

        return schema.Domain(
            cls.__name__,
            cls.feature_stat(x).to_dict()[x.name],
            [],
        )
