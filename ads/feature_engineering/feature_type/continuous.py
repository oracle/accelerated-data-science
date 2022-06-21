#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a Continuous feature type.

Classes:
    Continuous
        The Continuous feature type.
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


class Continuous(FeatureType):
    """
    Type representing continuous values.

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

    description = "Type representing continuous values."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, mean, standard deviation, sample minimum,
        lower quartile, median, 75%, upper quartile, skew and missing(count).

        Examples
        --------
        >>> cts = pd.Series([13.32, 3.32, 4.3, 2.45, 6.34, 2.25,
                            4.43, 3.26, np.NaN, None], name='continuous')
        >>> cts.ads.feature_type = ['continuous']
        >>> cts.ads.feature_stat()
            Metric                  Value
        0	count	                10.000
        1	mean	                4.959
        2	standard deviation	    3.620
        3	sample minimum	        2.250
        4	lower quartile	        3.058
        5	median	                3.810
        6	upper quartile	        4.908
        7	sample maximum	        13.320
        8	skew	                2.175
        9	missing	                2.000

        Returns
        -------
        :class:`pandas.DataFrame`
            Summary statistics of the Series or Dataframe provided.
        """
        df_stat = x.describe()
        _format_stat(df_stat)
        df_stat["count"] = len(x)
        df_stat["skew"] = x.skew()
        df_stat = _add_missing(x, df_stat).to_frame()
        df_stat.iloc[:, 0] = df_stat.iloc[:, 0].round(3)
        return df_stat

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows distributions of datasets using box plot.

        Examples
        --------
        >>> cts = pd.Series([13.32, 3.32, 4.3, 2.45, 6.34, 2.25,
                            4.43, 3.26, np.NaN, None], name='continuous')
        >>> cts.ads.feature_type = ['continuous']
        >>> cts.ads.feture_plot()

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the Continuous feature type.
        """
        col_name = x.name if x.name else "continuous"
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
        >>> cts = pd.Series([13.32, 3.32, 4.3, 2.45, 6.34, 2.25,
                            4.43, 3.26, np.NaN, None], name='continuous')
        >>> cts.ads.feature_type = ['continuous']
        >>> cts.ads.feature_domain()
        constraints: []
        stats:
            count: 10.0
            lower quartile: 3.058
            mean: 4.959
            median: 3.81
            missing: 2.0
            sample maximum: 13.32
            sample minimum: 2.25
            skew: 2.175
            standard deviation: 3.62
            upper quartile: 4.908
        values: Continuous

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the Continuous feature type.
        """

        return schema.Domain(cls.__name__, cls.feature_stat(x).to_dict()[x.name], [])
