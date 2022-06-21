#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents an Integer feature type.

Classes:
    Integer
        The Integer feature type.
"""
import matplotlib.pyplot as plt
import pandas as pd
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.utils import (
    _add_missing,
    _set_seaborn_theme,
    SchemeTeal,
    _format_stat,
)
from ads.feature_engineering import schema
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class Integer(FeatureType):
    """
    Type representing integer values.

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

    description = "Type representing integer values."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, mean, standard deviation, sample minimum,
        lower quartile, median, 75%, upper quartile, max and missing(count) if there is any.

        Examples
        --------
        >>> x = pd.Series([1, 0, 1, 2, 3, 4, np.nan], name='integer')
        >>> x.ads.feature_type = ['integer']
        >>> x.ads.feature_stat()
            Metric                  Value
        0	count	                7
        1	mean	                1
        2	standard deviation	    1
        3	sample minimum	        0
        4	lower quartile	        1
        5	median	                1
        6	upper quartile	        2
        7	sample maximum	        4
        8	missing	                1

        Returns
        -------
        :class:`pandas.DataFrame`
            Summary statistics of the Series or Dataframe provided.
        """
        df_stat = x.describe()
        _format_stat(df_stat)
        df_stat["count"] = len(x)
        df_stat = _add_missing(x, df_stat).to_frame()
        df_stat.iloc[:, 0] = df_stat.iloc[:, 0]
        return df_stat

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows distributions of datasets using box plot.

        Examples
        --------
        >>> x = pd.Series([1, 0, 1, 2, 3, 4, np.nan], name='integer')
        >>> x.ads.feature_type = ['integer']
        >>> x.ads.feature_plot()

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the Integer feature type.
        """
        col_name = x.name if x.name else "integer"
        df = x.to_frame(name=col_name)
        df = df[pd.to_numeric(df[col_name], errors="coerce").notnull()]
        if len(df.index):
            _set_seaborn_theme()
            return seaborn.boxplot(
                x=df[col_name], width=0.2, color=SchemeTeal.AREA_DARK
            )

    @classmethod
    def feature_domain(cls, x: pd.Series) -> schema.Domain:
        """
        Generate the domain of the data of this feature type.

        Examples
        --------
        >>> s = pd.Series([True, False, True, False, np.NaN, None], name='integer')
        >>> s.ads.feature_type = ['integer']
        >>> s.ads.feature_domain()
        constraints: []
        stats:
            count: 6
            freq: 2
            missing: 2
            top: true
            unique: 2
        values: Integer


        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the Integer feature type.
        """

        return schema.Domain(cls.__name__, cls.feature_stat(x).to_dict()[x.name], [])
