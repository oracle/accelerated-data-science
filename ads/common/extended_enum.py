#!/usr/bin/env python

# Copyright (c) 2022, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from abc import ABCMeta


class ExtendedEnumMeta(ABCMeta):
    """
    A helper metaclass to extend functionality of a generic "Enum-like" class.

    Methods
    -------
    __contains__(cls, item) -> bool:
        Checks if `item` is among the attribute values of the class.
        Case-insensitive if `item` is a string.
    values(cls) -> tuple:
        Returns the tuple of class attribute values.
    keys(cls) -> tuple:
        Returns the tuple of class attribute names.
    """

    def __contains__(cls, item: object) -> bool:
        """
        Checks if `item` is a member of the class's values.

        - If `item` is a string, does a case-insensitive match against any string
          values stored in the class.
        - Otherwise, does a direct membership test.
        """
        # Gather the attribute values
        attr_values = cls.values()

        # If item is a string, compare case-insensitively to any str-type values
        if isinstance(item, str):
            return any(
                isinstance(val, str) and val.lower() == item.lower()
                for val in attr_values
            )
        else:
            # For non-string items (e.g., int), do a direct membership check
            return item in attr_values

    def __iter__(cls):
        # Make the class iterable by returning an iterator over its values
        return iter(cls.values())

    def values(cls) -> tuple:
        """
        Gets the tuple of class attribute values, excluding private or special
        attributes and any callables (methods, etc.).
        """
        return tuple(
            value
            for key, value in cls.__dict__.items()
            if not key.startswith("_") and not callable(value)
        )

    def keys(cls) -> tuple:
        """
        Gets the tuple of class attribute names, excluding private or special
        attributes and any callables (methods, etc.).
        """
        return tuple(
            key
            for key, value in cls.__dict__.items()
            if not key.startswith("_") and not callable(value)
        )


class ExtendedEnum(metaclass=ExtendedEnumMeta):
    """The base class to extend functionality of a generic Enum.

    Examples
    --------
    >>> class TestEnum(ExtendedEnum):
    ...    KEY1 = "v1"
    ...    KEY2 = "v2"
    """
