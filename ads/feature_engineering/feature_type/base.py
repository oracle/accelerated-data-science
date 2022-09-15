#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import ABC, ABCMeta

from ads.feature_engineering.feature_type.handler.feature_validator import (
    FeatureValidator,
)
from ads.feature_engineering.feature_type.handler.feature_warning import (
    FeatureWarning,
)
from ads.common.utils import camel_to_snake


class Name:
    def __get__(self, instance, owner):
        return camel_to_snake(owner.__name__)


class FeatureBaseType(type):
    """The helper metaclass to extend fucntionality of FeatureType class."""

    def __new__(cls, classname, bases, dictionary):
        dictionary["validator"] = FeatureValidator()
        dictionary["warning"] = FeatureWarning()
        return type.__new__(cls, classname, bases, dictionary)


class FeatureBaseTypeMeta(FeatureBaseType, ABCMeta):
    """The class to provide compatibility between ABC and FeatureBaseType metaclass."""


class FeatureType(ABC, metaclass=FeatureBaseTypeMeta):
    """
    Abstract case for feature types. Default class attribute include name and description.
    Name is auto generated using camel to snake conversion unless specified.
    """

    name = Name()
    description = "Base feature type."


class Tag:
    """Class for free form tags. Name must be specified."""

    def __init__(self, name: str) -> None:
        """
        Initialize a tag instance.

        Parameters
        ----------
        name: str
            The name of the tag.
        """
        self.name = name
