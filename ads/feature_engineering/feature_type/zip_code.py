#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a ZipCode feature type.

Classes:
    ZipCode
        The ZipCode feature type.

Functions:
    default_handler(data: pd.Series) -> pd.Series
        Processes given data and indicates if the data matches requirements.
"""
import matplotlib.pyplot as plt
import pandas as pd
import re
from ads.feature_engineering.feature_type.string import String
from ads.feature_engineering.utils import (
    _count_unique_missing,
    _to_lat_long,
    _plot_gis_scatter,
    _zip_code,
)
from ads.feature_engineering import schema

PATTERN = re.compile(r"^[0-9]{5}(?:-[0-9]{4})?$", re.VERBOSE)


def default_handler(data: pd.Series, *args, **kwargs) -> pd.Series:
    """Processes given data and indicates if the data matches requirements.

    Parameters
    ----------
    data: pd.Series
        The data to process.

    Returns
    -------
    pd.Series: The logical list indicating if the data matches requirements.
    """

    def _is_zip_code(x: any):
        return (
            not pd.isnull(x) and isinstance(x, str) and re.match(PATTERN, x) is not None
        )

    return data.apply(lambda x: True if _is_zip_code(x) else False)


class ZipCode(String):
    """Type representing postal code.

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
        Shows the geometry distribution base on location of zipcode.


    Example
    -------
    >>> from ads.feature_engineering.feature_type.zip_code import ZipCode
    >>> import pandas as pd
    >>> import numpy as np
    >>> s = pd.Series(["94065", "90210", np.NaN, None], name='zipcode')
    >>> ZipCode.validator.is_zip_code(s)
    0     True
    1     True
    2    False
    3    False
    Name: zipcode, dtype: bool
    """

    description = "Type representing postal code."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, unique(count) and missing(count).

        Examples
        --------
        >>> zipcode = pd.Series([94065, 90210, np.NaN, None], name='zipcode')
        >>> zipcode.ads.feature_type = ['zip_code']
        >>> zipcode.ads.feature_stat()
            Metric  Value
        0	count	4
        1	unique	2
        2	missing	2

        Returns
        -------
        Pandas Dataframe
            Summary statistics of the Series provided.
        """
        return _count_unique_missing(x)

    @staticmethod
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows the geometry distribution base on location of zipcode.

        Examples
        --------
        >>> zipcode = pd.Series([94065, 90210, np.NaN, None], name='zipcode')
        >>> zipcode.ads.feature_type = ['zip_code']
        >>> zipcode.ads.feature_plot()
        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the ZipCode feature type.
        """
        gis = _to_lat_long(x.loc[default_handler(x)], _zip_code())
        if len(gis.index):
            return _plot_gis_scatter(gis, "longitude", "latitude")

    @classmethod
    def feature_domain(cls, x: pd.Series) -> schema.Domain:
        """
        Generate the domain of the data of this feature type.

        Examples
        --------
        >>> zipcode = pd.Series([94065, 90210, np.NaN, None], name='zipcode')
        >>> zipcode.ads.feature_type = ['zip_code']
        >>> zipcode.ads.feature_domain()
        constraints: []
        stats:
            count: 4
            missing: 2
            unique: 2
        values: ZipCode

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the ZipCode feature type.
        """

        return schema.Domain(
            cls.__name__,
            cls.feature_stat(x).to_dict()[x.name],
            [],
        )


ZipCode.validator.register("is_zip_code", default_handler)
