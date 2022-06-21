#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a Boolean feature type.

Classes:
    Boolean
        The feature type that represents binary values True/False.

Functions:
    default_handler(data: pd.Series) -> pd.Series
        Processes given data and indicates if the data matches requirements.
"""
import matplotlib.pyplot as plt
import pandas as pd
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.utils import (
    _count_unique_missing,
    is_boolean,
    _set_seaborn_theme,
    SchemeTeal,
)
from ads.feature_engineering import schema
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


def default_handler(data: pd.Series, *args, **kwargs) -> pd.Series:
    """Processes given data and indicates if the data matches requirements.

    Parameters
    ----------
    data: :class:`pandas.Series`
        The data to process.

    Returns
    -------
    :class:`pandas.Series`
        The logical list indicating if the data matches requirements.
    """
    return pd.Series((is_boolean(value) for value in data.values))


class Boolean(FeatureType):
    """
    Type representing binary values True/False.

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
        Show the counts of observations in True/False using bars.

    Examples
    --------
    >>> from ads.feature_engineering.feature_type.boolean import Boolean
    >>> import pandas as pd
    >>> import numpy as np
    >>> s = pd.Series([True, False, True, False, np.NaN, None], name='bool')
    >>> s.ads.feature_type = ['boolean']
    >>> Boolean.validator.is_boolean(s)
    0     True
    1     True
    2     True
    3     True
    4    False
    5    False
    dtype: bool
    """

    description = "Type representing binary values True/False."

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
            Summary statistics of the Series or Dataframe provided.

        Examples
        --------
        >>> s = pd.Series([True, False, True, False, np.NaN, None], name='bool')
        >>> s.ads.feature_type = ['boolean']
        >>> s.ads.feature_stat()
            Metric  Value
        0	count	6
        1	unique	2
        2	missing	2
        """
        return _count_unique_missing(x)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows the counts of observations in True/False using bars.

        Parameters
        ----------
        x : :class:`pandas.Series`
            The feature being evaluated.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the Boolean feature type.

        Examples
        --------
        >>> s = pd.Series([True, False, True, False, np.NaN, None], name='bool')
        >>> s.ads.feature_type = ['boolean']
        >>> s.ads.feature_plot()
        """
        col_name = x.name if x.name else "boolean"
        df = x.to_frame(col_name)
        df["validation"] = default_handler(x)
        df = df[df["validation"] == True]
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
        >>> s = pd.Series([True, False, True, False, np.NaN, None], name='bool')
        >>> s.ads.feature_type = ['boolean']
        >>> s.ads.feature_domain()
        constraints:
        - expression: $x in [True, False]
            language: python
        stats:
            count: 6
            missing: 2
            unique: 2
        values: Boolean

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the Boolean feature type.
        """

        return schema.Domain(
            cls.__name__,
            cls.feature_stat(x).to_dict()[x.name],
            [schema.Expression("$x in [True, False]")],
        )


Boolean.validator.register("is_boolean", default_handler)
