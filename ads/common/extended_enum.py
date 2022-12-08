#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from abc import ABCMeta


class ExtendedEnumMeta(ABCMeta):
    """The helper metaclass to extend functionality of a generic Enum.

    Methods
    -------
    values(cls) -> list:
        Gets the list of class attributes.
    """

    def __contains__(cls, value):
        return value and value.lower() in tuple(value.lower() for value in cls.values())

    def values(cls) -> list:
        """Gets the list of class attributes.

        Returns
        -------
        list
            The list of class values.
        """
        return tuple(
            value for key, value in cls.__dict__.items() if not key.startswith("_")
        )
