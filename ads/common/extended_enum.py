#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from abc import ABCMeta
from enum import Enum


class ExtendedEnumMeta(ABCMeta):
    """The helper metaclass to extend functionality of a generic Enum.

    Methods
    -------
    values(cls) -> list:
        Gets the list of class attributes.

    Examples
    --------
    >>> class TestEnum(str, metaclass=ExtendedEnumMeta):
    ...    KEY1 = "value1"
    ...    KEY2 = "value2"
    >>> print(TestEnum.KEY1) # "value1"
    """

    def __contains__(cls, value):
        return value and value.lower() in tuple(value.lower() for value in cls.values())

    def values(cls) -> list:
        """Gets the list of class attributes values.

        Returns
        -------
        list
            The list of class values.
        """
        return tuple(
            value for key, value in cls.__dict__.items() if not key.startswith("_")
        )

    def keys(cls) -> list:
        """Gets the list of class attributes names.

        Returns
        -------
        list
            The list of class attributes names.
        """
        return tuple(
            key for key, value in cls.__dict__.items() if not key.startswith("_")
        )


class ExtendedEnum(Enum):
    """The base class to extend functionality of a generic Enum.

    Examples
    --------
    >>> class TestEnum(ExtendedEnumMeta):
    ...    KEY1 = "value1"
    ...    KEY2 = "value2"
    >>> print(TestEnum.KEY1.value) # "value1"
    """

    @classmethod
    def values(cls):
        return sorted(map(lambda c: c.value, cls))

    @classmethod
    def keys(cls):
        return sorted(map(lambda c: c.name, cls))
