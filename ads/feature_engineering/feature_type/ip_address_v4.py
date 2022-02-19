#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents an IpAddressV4 feature type.

Classes:
    IpAddressV4
        The IpAddressV4 feature type.
"""
import re

import pandas as pd
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.utils import _count_unique_missing
from ads.feature_engineering import schema

PATTERN = re.compile(
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
    re.IGNORECASE,
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
    return data.apply(
        lambda x: True
        if not pd.isnull(x) and PATTERN.match(str(x)) is not None
        else False
    )


class IpAddressV4(FeatureType):
    """
    Type representing IP Address V4.

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

    Example
    -------
    >>> from ads.feature_engineering.feature_type.ip_address_v4 import IpAddressV4
    >>> import pandas as pd
    >>> import numpy as np
    >>> s = pd.Series(['192.168.0.1', '2001:db8::', '', np.NaN, None], name='ip_address')
    >>> s.ads.feature_type = ['ip_address_v4']
    >>> IpAddressV4.validator.is_ip_address_v4(s)
    0     True
    1    False
    2    False
    3    False
    4    False
    Name: ip_address, dtype: bool
    """

    description = "Type representing IP Address V4."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, unique(count) and missing(count).

        Examples
        --------
        >>> s = pd.Series(['192.168.0.1', '192.168.0.2', '192.168.0.3', '192.168.0.4', np.NaN, None], name='ip_address')
        >>> s.ads.feature_type = ['ip_address_v4']
        >>> s.ads.feature_stat()
            Metric  Value
        0	count	6
        1	unique	4
        2	missing	2

        Returns
        -------
        :class:`pandas.DataFrame`
            Summary statistics of the Series provided.
        """
        return _count_unique_missing(x)

    @classmethod
    def feature_domain(cls, x: pd.Series) -> schema.Domain:
        """
        Generate the domain of the data of this feature type.

        Examples
        --------
        >>> s = pd.Series(['192.168.0.1', '192.168.0.2', '192.168.0.3', '192.168.0.4', np.NaN, None], name='ip_address_v4')
        >>> s.ads.feature_type = ['ip_address_v4']
        >>> s.ads.feature_domain()
        constraints: []
        stats:
            count: 6
            missing: 2
            unique: 4
        values: IpAddressV4

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the IpAddressV4 feature type.
        """

        return schema.Domain(
            cls.__name__,
            cls.feature_stat(x).to_dict()[x.name],
            [],
        )


IpAddressV4.validator.register("is_ip_address_v4", default_handler)
