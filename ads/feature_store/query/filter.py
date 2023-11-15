#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.jobs.builders.base import Builder


class Filter(Builder):
    """
    Represents a filter for a query.
    """

    CONST_FEATURE = "feature"
    CONST_CONDITION = "condition"
    CONST_VALUE = "value"

    def __init__(self, feature, condition, value, **kwargs):
        super().__init__(**kwargs)

        self.with_feature(feature)
        self.with_condition(condition)
        self.with_value(value)

    def with_feature(self, feature):
        """
        Sets the feature to filter on.

        Args:
            feature (str): The feature to filter on.
        """
        self.set_spec(self.CONST_FEATURE, feature)

    @property
    def feature(self):
        return self.get_spec(self.CONST_FEATURE)

    def with_condition(self, condition):
        """
        Sets the condition to apply to the feature.

        Args:
            condition (str): The condition to apply to the feature.
        """
        self.set_spec(self.CONST_CONDITION, condition)

    @property
    def condition(self):
        return self.get_spec(self.CONST_CONDITION)

    def with_value(self, value):
        """
        Sets the value to filter for.

        Args:
            value (str): The value to filter for.
        """
        self.set_spec(self.CONST_VALUE, value)

    @property
    def value(self):
        return self.get_spec(self.CONST_VALUE)

    def to_dict(self):
        return {
            "feature": self.feature,
            "condition": self.condition,
            "value": str(self.value),
        }

    def __and__(self, other):
        """
        Overloads the & operator to create a new Logic object.

        Args:
            other (Union[Filter, Logic]): The other object to combine with.

        Returns:
            Logic: The new Logic object.
        """
        if isinstance(other, Filter):
            return Logic.And(left_f=self, right_f=other)
        elif isinstance(other, Logic):
            return Logic.And(left_f=self, right_l=other)
        else:
            raise TypeError(
                "Operator `&` expected type `Filter` or `Logic`, got `{}`".format(
                    type(other)
                )
            )

    def __or__(self, other):
        """
        Overloads the | operator to create a new Logic object.

        Args:
            other (Union[Filter, Logic]): The other object to combine with.

        Returns:
            Logic: The new Logic object.
        """
        if isinstance(other, Filter):
            return Logic.Or(left_f=self, right_f=other)
        elif isinstance(other, Logic):
            return Logic.Or(left_f=self, right_l=other)
        else:
            raise TypeError(
                "Operator `|` expected type `Filter` or `Logic`, got `{}`".format(
                    type(other)
                )
            )

    def __repr__(self):
        return f"Filter({self.feature!r}, {self.condition!r}, {self.value!r})"


class Logic(Builder):
    """
    A class representing a logical operation on filters.
    """

    AND = "AND"
    OR = "OR"
    SINGLE = "SINGLE"

    CONST_TYPE = "type"
    CONST_LEFT_FILTER = "leftFilter"
    CONST_RIGHT_FILTER = "rightFilter"
    CONST_LEFT_LOGIC = "leftLogic"
    CONST_RIGHT_LOGIC = "rightLogic"

    def __init__(
        self, type, left_f=None, right_f=None, left_l=None, right_l=None, **kwargs
    ):
        super().__init__(**kwargs)

        self.with_type(type)
        self.with_left_filter(left_f)
        self.with_right_filter(right_f)
        self.with_left_logic(left_l)
        self.with_right_logic(right_l)

    def with_type(self, type):
        """
        Sets the type of the logic.

        Parameters:
        -----------
        type: str
            A string representing the type of the logic to be performed.
        """
        self.set_spec(self.CONST_TYPE, type)

    @property
    def type(self):
        return self.get_spec(self.CONST_TYPE)

    def with_left_filter(self, left_filter):
        """
        Sets the left filter of the logic.

        Parameters:
        -----------
        left_filter: Filter or None
            The left filter of the logic.
        """
        self.set_spec(self.CONST_LEFT_FILTER, left_filter)

    @property
    def left_filter(self):
        return self.get_spec(self.CONST_LEFT_FILTER)

    def with_right_filter(self, right_filter):
        """
        Sets the right filter of the logic.

        Parameters:
        -----------
        right_filter: Filter or None
            The right filter of the logic.
        """
        self.set_spec(self.CONST_RIGHT_FILTER, right_filter)

    @property
    def right_filter(self):
        return self.get_spec(self.CONST_RIGHT_FILTER)

    def with_left_logic(self, left_logic):
        """
        Sets the left logic of the operation.

        Parameters:
        -----------
        left_logic: Logic or None
            The left logic of the operation.
        """
        self.set_spec(self.CONST_LEFT_LOGIC, left_logic)

    @property
    def left_logic(self):
        return self.get_spec(self.CONST_LEFT_LOGIC)

    def with_right_logic(self, right_logic):
        """
        Sets the right logic of the operation.

        Parameters:
        -----------
        right_logic: Logic or None
            The right logic of the operation.
        """

        self.set_spec(self.CONST_RIGHT_LOGIC, right_logic)

    @property
    def right_logic(self):
        return self.get_spec(self.CONST_RIGHT_LOGIC)

    def to_dict(self):
        return {
            "type": self.type,
            "leftFilter": self.left_filter,
            "rightFilter": self.right_filter,
            "leftLogic": self.left_logic,
            "rightLogic": self.right_logic,
        }

    @classmethod
    def And(cls, left_f=None, right_f=None, left_l=None, right_l=None):
        return cls(cls.AND, left_f, right_f, left_l, right_l)

    @classmethod
    def Or(cls, left_f=None, right_f=None, left_l=None, right_l=None):
        return cls(cls.OR, left_f, right_f, left_l, right_l)

    @classmethod
    def Single(cls, left_f):
        return cls(cls.SINGLE, left_f)

    def __and__(self, other):
        if isinstance(other, Filter):
            return Logic.And(left_l=self, right_f=other)
        elif isinstance(other, Logic):
            return Logic.And(left_l=self, right_l=other)
        else:
            raise TypeError(
                "Operator `&` expected type `Filter` or `Logic`, got `{}`".format(
                    type(other)
                )
            )

    def __or__(self, other):
        if isinstance(other, Filter):
            return Logic.Or(left_l=self, right_f=other)
        elif isinstance(other, Logic):
            return Logic.Or(left_l=self, right_l=other)
        else:
            raise TypeError(
                "Operator `|` expected type `Filter` or `Logic`, got `{}`".format(
                    type(other)
                )
            )

    def __repr__(self):
        return f"Logic({self.type!r}, {self.left_filter!r}, {self.right_filter!r}, {self.left_logic!r}, {self.right_logic!r}) "
