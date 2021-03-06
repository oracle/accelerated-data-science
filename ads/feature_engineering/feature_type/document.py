#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a Document feature type.

Classes:
    Document
        The Document feature type.
"""
from ads.feature_engineering.feature_type.base import FeatureType


class Document(FeatureType):
    """
    Type representing document values.

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
    """

    description = "Type representing document values."

    @classmethod
    def feature_domain(cls):
        """
        Returns
        -------
        None
            Nothing.
        """
        return None
