#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a Discrete feature type.

Classes:
    Discrete
        The Discrete feature type.
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


class Discrete(FeatureType):
    """
    Type representing discrete values.

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
        Shows distributions of datasets using box plot.
    """

    description = "Type representing discrete values."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, unique(count) and missing(count).

        Examples
        --------
        >>> discrete_numbers = pd.Series([35, 25, 13, 42],
                   name='discrete')
        >>> discrete_numbers.ads.feature_type = ['discrete']
        >>> discrete_numbers.ads.feature_stat()
                    discrete
        count	4
        unique	4

        Returns
        -------
        :class:`pandas.DataFrame`
            Summary statistics of the Series provided.
        """
        return _count_unique_missing(x)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows distributions of datasets using box plot.

        Examples
        --------
        >>> discrete_numbers = pd.Series([35, 25, 13, 42],
                   name='discrete')
        >>> discrete_numbers.ads.feature_type = ['discrete']
        >>> discrete_numbers.ads.feature_stat()
            Metric  Value
        0	count	4
        1	unique	4

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the Discrete feature type.
        """
        col_name = x.name if x.name else "discrete"
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
        >>> discrete_numbers = pd.Series([35, 25, 13, 42],
                   name='discrete')
        >>> discrete_numbers.ads.feature_type = ['discrete']
        >>> discrete_numbers.ads.feature_domain()
        constraints: []
        stats:
            count: 4
            unique: 4
        values: Discrete

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the Discrete feature type.
        """

        return schema.Domain(
            cls.__name__,
            cls.feature_stat(x).to_dict()[x.name],
            [],
        )
