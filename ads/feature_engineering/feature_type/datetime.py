#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a DateTime feature type.

Classes:
    DateTime
        The DateTime feature type.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.utils import (
    _add_missing,
    _set_seaborn_theme,
    SchemeTeal,
)
from ads.feature_engineering import schema
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


def default_handler(data: pd.Series, *args, **kwargs) -> pd.Series:
    """
    Processes given data and indicates if the data matches requirements.

    Parameters
    ----------
    data: :class:`pandas.Series`
        The data to process.

    Returns
    -------
    :class:`pandas.Series`
        The logical list indicating if the data matches requirements.
    """

    def _is_datetime(x: any):
        if pd.isnull(x):
            return False
        if pdtypes.is_datetime64_any_dtype(type(x)):
            return True
        if pdtypes.is_string_dtype(type(x)) or pdtypes.is_object_dtype(type(x)):
            try:
                pd.to_datetime(x)
            except:
                return False
            return True
        return False

    return data.apply(lambda x: True if _is_datetime(x) else False)


class DateTime(FeatureType):
    """
    Type representing date and/or time.

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
        Shows distributions of datetime datasets using histograms.

    Example
    -------
    >>> from ads.feature_engineering.feature_type.datetime import DateTime
    >>> import pandas as pd
    >>> s = pd.Series(["12/12/12", "12/12/13", None, "12/12/14"], name='datetime')
    >>> s.ads.feature_type = ['date_time']
    >>> DateTime.validator.is_datetime(s)
    0     True
    1     True
    2    False
    3     True
    Name: datetime, dtype: bool
    """

    description = "Type representing date and/or time."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, sample maximum, sample minimum, and
        missing(count) if there is any.

        Examples
        --------
        >>> x = pd.Series(['3/11/2000', '3/12/2000', '3/13/2000', '', None, np.nan, 'April/13/2011', 'April/15/11'], name='datetime')
        >>> x.ads.feature_type = ['date_time']
        >>> x.ads.feature_stat()
            Metric              Value
        0	count	            8
        1	sample maximum	    April/15/11
        2	sample minimum	    3/11/2000
        3	missing	            3

        Returns
        -------
        :class:`pandas.DataFrame`
            Summary statistics of the Series or Dataframe provided.
        """
        df_stat = pd.Series(
            {
                "count": len(x),
                "sample maximum": x.replace(r"", np.NaN).dropna().max(),
                "sample minimum": x.replace(r"", np.NaN).dropna().min(),
            },
            name=x.name,
        ).to_frame()
        return _add_missing(x.replace(r"", np.NaN), df_stat)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows distributions of datetime datasets using histograms.

        Examples
        --------
        >>> x = pd.Series(['3/11/2000', '3/12/2000', '3/13/2000', '', None, np.nan, 'April/13/2011', 'April/15/11'], name='datetime')
        >>> x.ads.feature_type = ['date_time']
        >>> x.ads.feature_plot()

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the DateTime feature type.
        """
        col_name = x.name if x.name else "datetime"
        df = x.to_frame(col_name)
        df["validation"] = default_handler(x)
        df = df[df["validation"] == True]
        if len(df.index):
            df[col_name] = df[col_name].apply(lambda x: pd.to_datetime(x))
            _set_seaborn_theme()
            return seaborn.histplot(data=df, y=col_name, color=SchemeTeal.AREA_DARK)

    @classmethod
    def feature_domain(cls, x: pd.Series) -> schema.Domain:
        """
        Generate the domain of the data of this feature type.

        Examples
        --------
        >>> s = pd.Series(['3/11/2000', '3/12/2000', '3/13/2000', '', None, np.nan, 'April/13/2011', 'April/15/11'], name='datetime')
        >>> s.ads.feature_type = ['date_time']
        >>> s.ads.feature_domain()
        constraints: []
        stats:
            count: 8
            missing: 3
            sample maximum: April/15/11
            sample minimum: 3/11/2000
        values: DateTime

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the DateTime feature type.
        """

        return schema.Domain(
            cls.__name__,
            cls.feature_stat(x).to_dict()[x.name],
            [],
        )


DateTime.validator.register("is_datetime", default_handler)
