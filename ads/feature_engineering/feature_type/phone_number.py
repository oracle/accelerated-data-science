#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a Phone Number feature type.

Classes:
    PhoneNumber
        The Phone Number feature type.

Functions:
    default_handler(data: pd.Series) -> pd.Series
        Processes given data and indicates if the data matches requirements.
"""
import pandas as pd
import re
from ads.feature_engineering.feature_type.string import String
from ads.feature_engineering.utils import _count_unique_missing
from ads.feature_engineering import schema


PATTERN = re.compile(
    r"^(\+?\d{1,2}[\s-])?\(?(\d{3})\)?[\s.-]?\d{3}[\s.-]?\d{4}$", re.VERBOSE
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

    def _is_phone_number(x: any):
        return (
            not pd.isnull(x) and isinstance(x, str) and re.match(PATTERN, x) is not None
        )

    return data.apply(lambda x: True if _is_phone_number(x) else False)


class PhoneNumber(String):
    """Type representing phone numbers.

    Attributes
    -----------
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

    Examples
    --------
    >>> from ads.feature_engineering.feature_type.phone_number import PhoneNumber
    >>> import pandas as pd
    >>> s = pd.Series([None, "1-640-124-5367", "1-573-916-4412"])
    >>> PhoneNumber.validator.is_phone_number(s)
    0    False
    1     True
    2     True
    dtype: bool
    """

    description = "Type representing phone numbers."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, unique(count) and
        missing(count) if there is any.

        Examples
        --------
        >>> s = pd.Series(['2068866666', '6508866666', '2068866666', '', np.NaN, np.nan, None], name='phone')
        >>> s.ads.feature_type = ['phone_number']
        >>> s.ads.feature_stat()
            Metric  Value
        1	count	7
        2	unique	2
        3	missing	4

        Returns
        -------
        :class:`pandas.DataFrame`
            Summary statistics of the Series or Dataframe provided.
        """
        return _count_unique_missing(x)

    @classmethod
    def feature_domain(cls, x: pd.Series) -> schema.Domain:
        """
        Generate the domain of the data of this feature type.

        Examples
        --------
        >>> s = pd.Series(['2068866666', '6508866666', '2068866666', '', np.NaN, np.nan, None], name='phone')
        >>> s.ads.feature_type = ['phone_number']
        >>> s.ads.feature_domain()
        constraints: []
        stats:
            count: 7
            missing: 4
            unique: 2
        values: PhoneNumber

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the PhoneNumber feature type.
        """

        return schema.Domain(cls.__name__, cls.feature_stat(x).to_dict()[x.name], [])


PhoneNumber.validator.register("is_phone_number", default_handler)
