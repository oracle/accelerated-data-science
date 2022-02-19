#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents an IpAddressV6 feature type.

Classes:
    IpAddressV6
        The IpAddressV6 feature type.
"""
import re

import pandas as pd
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.utils import _count_unique_missing
from ads.feature_engineering import schema

PATTERN = re.compile(
    r"\s*(?!.*::.*::)(?:(?!:)|:(?=:))(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)){6}(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)[0-9a-f]{0,4}(?:(?<=::)|(?<!:)|(?<=:)(?<!::):)|(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)){3})\s*",
    re.VERBOSE | re.IGNORECASE | re.DOTALL,
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


class IpAddressV6(FeatureType):
    """
    Type representing IP Address V6.

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
    >>> from ads.feature_engineering.feature_type.ip_address_v6 import IpAddressV6
    >>> import pandas as pd
    >>> import numpy as np
    >>> s = pd.Series(['192.168.0.1', '2001:db8::', '', np.NaN, None], name='ip_address')
    >>> s.ads.feature_type = ['ip_address_v6']
    >>> IpAddressV6.validator.is_ip_address_v6(s)
    0    False
    1     True
    2    False
    3    False
    4    False
    Name: ip_address, dtype: bool
    """

    description = "Type representing IP Address V6."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, unique(count) and missing(count).

        Examples
        --------
        >>> s = pd.Series(['2002:db8::', '2001:db8::', '2001:db8::', '2002:db8::', np.NaN, None], name='ip_address')
        >>> s.ads.feature_type = ['ip_address_v6']
        >>> s.ads.feature_stat()
            Metric  Value
        0	count	6
        1	unique	2
        2	missing	2

        Returns
        -------
        Pandas Dataframe
            Summary statistics of the Series provided.
        """
        return _count_unique_missing(x)

    @classmethod
    def feature_domain(cls, x: pd.Series) -> schema.Domain:
        """
        Generate the domain of the data of this feature type.

        Examples
        --------
        >>> s = pd.Series(['2002:db8::', '2001:db8::', '2001:db8::', '2002:db8::', np.NaN, None], name='ip_address_v6')
        >>> s.ads.feature_type = ['ip_address_v6']
        >>> s.ads.feature_domain()
        constraints: []
        stats:
            count: 6
            missing: 2
            unique: 2
        values: IpAddressV6

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the IpAddressV6 feature type.
        """

        return schema.Domain(
            cls.__name__,
            cls.feature_stat(x).to_dict()[x.name],
            [],
        )


IpAddressV6.validator.register("is_ip_address_v6", default_handler)
