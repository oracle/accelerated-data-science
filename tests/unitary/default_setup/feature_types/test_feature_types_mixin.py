#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit Tests for feature type mixin module."""

from copy import copy
from unittest import TestCase
import pandas as pd

from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.feature_type.string import String
from ads.feature_engineering.feature_type_manager import FeatureTypeManager


def zeros_warning_handler(x):
    return pd.DataFrame(
        [["Zeros", "Age has 38 zeros", "Count", 38]],
        columns=["Warning", "Message", "Metric", "Value"],
    )


def default_validator_handler(x):
    return x


class CustomType(FeatureType):
    pass


CustomType.validator.register(name="is_custom_type", handler=default_validator_handler)


CustomType.warning.register("zeros_warning1", zeros_warning_handler)


class CustomType3(FeatureType):
    pass


CustomType3.warning.register("zeros_warning2", zeros_warning_handler)

CustomType3.validator.register(
    name="is_custom_type3", handler=default_validator_handler
)


class TestFeatureTypesMixin(TestCase):
    """Unittest for FeatureTypesMixin class."""

    def setUp(self):
        """Sets up the test case."""
        super().setUp()
        self._name_to_type_map = copy(FeatureTypeManager._name_to_type_map)
        FeatureTypeManager._name_to_type_map = {}
        FeatureTypeManager.feature_type_register(CustomType)
        FeatureTypeManager.feature_type_register(CustomType3)
        FeatureTypeManager.feature_type_register(String)
        self.mock_df = pd.DataFrame(
            {
                "Name": ["Alex", "Liam", "Noah"],
                "PhoneNumber": [
                    "+1-202-555-0141",
                    "+1-202-555-0198",
                    "+1-202-555-0199",
                ],
            }
        )
        self.mock_df.ads.feature_type = {"PhoneNumber": ["custom_type", "custom_type3"]}

    def tearDown(self):
        super().tearDown()
        FeatureTypeManager._name_to_type_map = self._name_to_type_map

    def test_warning_registered(self):
        """Ensures all registered warnings can be listed as a dataframe."""
        expected_result = pd.DataFrame(
            [
                ["Name", "string", "missing_values", "missing_values_handler"],
                ["Name", "string", "high_cardinality", "high_cardinality_handler"],
                [
                    "PhoneNumber",
                    CustomType.name,
                    "zeros_warning1",
                    "zeros_warning_handler",
                ],
                [
                    "PhoneNumber",
                    CustomType3.name,
                    "zeros_warning2",
                    "zeros_warning_handler",
                ],
                ["PhoneNumber", "string", "missing_values", "missing_values_handler"],
                [
                    "PhoneNumber",
                    "string",
                    "high_cardinality",
                    "high_cardinality_handler",
                ],
            ],
            columns=["Column", "Feature Type", "Warning", "Handler"],
        )
        result_df = self.mock_df.ads.warning_registered()
        print(result_df)
        assert pd.DataFrame.equals(result_df, expected_result)

    def test_validator_registered(self):
        """Ensures all registered validators can be listed as a dataframe."""
        expected_result = pd.DataFrame(
            [
                ["Name", String.name, "is_string", "()", "default_handler"],
                [
                    "PhoneNumber",
                    CustomType.name,
                    "is_custom_type",
                    "()",
                    "default_validator_handler",
                ],
                [
                    "PhoneNumber",
                    CustomType3.name,
                    "is_custom_type3",
                    "()",
                    "default_validator_handler",
                ],
                ["PhoneNumber", String.name, "is_string", "()", "default_handler"],
            ],
            columns=["Column", "Feature Type", "Validator", "Condition", "Handler"],
        )

        result_df = self.mock_df.ads.validator_registered()
        assert pd.DataFrame.equals(result_df, expected_result)
