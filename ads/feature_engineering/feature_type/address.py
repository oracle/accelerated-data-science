#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents an Address feature type.

Classes:
    Address
        The Address feature type.
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

PATTERN = re.compile(
    r"\d{1,5} [\w\s]{1,30}(?:street|st(?:\s|\.)+|avenue|ave(?:\s|\.)+|road|rd(?:\s|\.)+|highway|hwy(?:\s|\.)+|square|sq(?:\s|\.)+|trail|trl(?:\s|\.)+|drive|dr(?:\s|\.)+|court|ct(?:\s|\.)+|park|parkway|pkwy(?:\s|\.)+|circle|cir(?:\s|\.)+|boulevard|blvd(?:\s|\.)+|island|port|view|parkways)(?:suite\s?\d+|apt\.?\s?\d+|ste\.?\s?\d+)?[\w\s,]{1,30}\d{5}\W?(?=\s|$)",
    re.IGNORECASE,
)


def default_handler(data: pd.Series, *args, **kwargs) -> pd.Series:
    """Processes given data and indicates if the data matches requirements.

    Parameters
    ----------
    data: pd.Series
        The data to process.

    Returns
    -------
    :class:`pandas.Series`
        The logical list indicating if the data matches requirements.
    """

    def _is_address(x):
        return (
            not pd.isnull(x)
            and isinstance(x, str)
            and PATTERN.match(str(x)) is not None
        )

    return data.apply(lambda x: True if _is_address(x) else False)


class Address(String):
    """
    Type representing address.

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
        Shows the location of given address on map base on zip code.

    Example
    -------
    >>> from ads.feature_engineering.feature_type.address import Address
    >>> import pandas as pd
    >>> address = pd.Series(['1 Miller Drive, New York, NY 12345',
                            '1 Berkeley Street, Boston, MA 67891',
                            '54305 Oxford Street, Seattle, WA 95132',
                            ''])
    >>> Address.validator.is_address(address)
    0     True
    1     True
    2     True
    3    False
    dtype: bool
    """

    description = "Type representing address."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, unique(count) and missing(count).

        Examples
        --------
        >>> address = pd.Series(['1 Miller Drive, New York, NY 12345',
                      '1 Berkeley Street, Boston, MA 67891',
                      '54305 Oxford Street, Seattle, WA 95132',
                      ''],
                   name='address')
        >>> address.ads.feature_type = ['address']
        >>> address.ads.feature_stat()
            Metric  Value
        0	count	4
        1	unique	3
        2	missing	1

        Returns
        -------
        :class:`pandas.DataFrame`
            Summary statistics of the Series provided.
        """
        return _count_unique_missing(x)

    @staticmethod
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows the location of given address on map base on zip code.

        Examples
        --------
        >>> address = pd.Series(['1 Miller Drive, New York, NY 12345',
                      '1 Berkeley Street, Boston, MA 67891',
                      '54305 Oxford Street, Seattle, WA 95132',
                      ''],
                   name='address')
        >>> address.ads.feature_type = ['address']
        >>> address.ads.feature_plot()
        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the Address feature type.
        """

        def _get_zipcode(n):
            return re.findall(r"\D(\d{5})", n)[-1]

        gis = _to_lat_long(x.loc[default_handler(x)].apply(_get_zipcode), _zip_code())
        if len(gis.index):
            return _plot_gis_scatter(gis, "longitude", "latitude")

    @classmethod
    def feature_domain(cls, x: pd.Series) -> schema.Domain:
        """
        Generate the domain of the data of this feature type.

        Examples
        --------
        >>> address = pd.Series(['1 Miller Drive, New York, NY 12345',
                      '1 Berkeley Street, Boston, MA 67891',
                      '54305 Oxford Street, Seattle, WA 95132',
                      ''],
                   name='address')
        >>> address.ads.feature_type = ['address']
        >>> address.ads.feature_domain()
        constraints: []
        stats:
            count: 4
            missing: 1
            unique: 3
        values: Address

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the Address feature type.
        """

        return schema.Domain(
            cls.__name__,
            cls.feature_stat(x).to_dict()[x.name],
            [],
        )


Address.validator.register("is_address", default_handler)
