#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit Tests for feature types manager module."""

from copy import copy
from unittest import TestCase
import pytest
import pandas as pd
from ads.feature_engineering.feature_type_manager import FeatureTypeManager
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.feature_type.boolean import Boolean
from ads.feature_engineering import exceptions


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


class CustomType2(FeatureType):
    name = "custom_type"


class CustomType3(FeatureType):
    pass


CustomType3.warning.register("zeros_warning2", zeros_warning_handler)

CustomType3.validator.register(
    name="is_custom_type3", handler=default_validator_handler
)


class CustomType4(FeatureType):
    pass


class Dummy:
    pass


class TestFeatureTypeManager(TestCase):
    """Unittest for FeatureTypesMixin class."""

    def setUp(self):
        """Sets up the test case."""
        super().setUp()
        FeatureTypeManager.feature_type_reset()
        self._name_to_type_map = copy(FeatureTypeManager._name_to_type_map)

    def tearDown(self):
        super().tearDown()
        FeatureTypeManager._name_to_type_map = self._name_to_type_map

    def test_register_type(self):
        """Ensures feature type can be registered."""
        FeatureTypeManager.feature_type_register(CustomType)
        assert FeatureTypeManager._name_to_type_map["custom_type"] == CustomType
        with pytest.raises(exceptions.TypeAlreadyRegistered):
            FeatureTypeManager.feature_type_register(CustomType)
        with pytest.raises(exceptions.InvalidFeatureType):
            FeatureTypeManager.feature_type_register(Dummy)
        with pytest.raises(exceptions.NameAlreadyRegistered):
            FeatureTypeManager.feature_type_register(CustomType2)

    def test_unregister_type(self):
        """Ensures feature type can be unregistered."""
        FeatureTypeManager.feature_type_register(CustomType)
        FeatureTypeManager.feature_type_unregister(CustomType)
        assert "custom_type" not in FeatureTypeManager._name_to_type_map
        with pytest.raises(exceptions.TypeNotFound):
            FeatureTypeManager.feature_type_unregister(CustomType)
        with pytest.raises(exceptions.TypeNotFound):
            FeatureTypeManager.feature_type_unregister("custom_type")
        with pytest.raises(TypeError):
            FeatureTypeManager.feature_type_unregister("category")

    def test_list_feature_type_registered(self):
        """Ensures all registered types can be got as a dataframe."""
        FeatureTypeManager.feature_type_register(CustomType)
        FeatureTypeManager.feature_type_register(CustomType3)
        df = FeatureTypeManager.feature_type_registered()
        assert df.shape == (len(FeatureTypeManager._name_to_type_map), 3)

    def test_reset(self):
        """Ensures feature types can be reseted to default state."""
        FeatureTypeManager.feature_type_register(CustomType)
        FeatureTypeManager.feature_type_register(CustomType3)
        assert "custom_type" in FeatureTypeManager._name_to_type_map
        assert "custom_type3" in FeatureTypeManager._name_to_type_map
        FeatureTypeManager.feature_type_reset()
        assert "custom_type" not in FeatureTypeManager._name_to_type_map
        assert "custom_type3" not in FeatureTypeManager._name_to_type_map

    def test_warning_registered(self):
        """Ensures all registered warnings can be got as a dataframe."""
        FeatureTypeManager._name_to_type_map = {}
        FeatureTypeManager.feature_type_register(CustomType)
        FeatureTypeManager.feature_type_register(CustomType3)
        expected_result = pd.DataFrame(
            [
                [CustomType.name, "zeros_warning1", "zeros_warning_handler"],
                [CustomType3.name, "zeros_warning2", "zeros_warning_handler"],
            ],
            columns=["Feature Type", "Warning", "Handler"],
        )

        result_df = FeatureTypeManager.warning_registered()
        assert pd.DataFrame.equals(result_df, expected_result)

    def test_validator_registered(self):
        """Ensures all registered vlidators can be got as a dataframe."""
        FeatureTypeManager._name_to_type_map = {}
        FeatureTypeManager.feature_type_register(CustomType)
        FeatureTypeManager.feature_type_register(CustomType3)
        expected_result = pd.DataFrame(
            [
                [CustomType.name, "is_custom_type", "()", "default_validator_handler"],
                [
                    CustomType3.name,
                    "is_custom_type3",
                    "()",
                    "default_validator_handler",
                ],
            ],
            columns=["Feature Type", "Validator", "Condition", "Handler"],
        )

        result_df = FeatureTypeManager.validator_registered()
        assert pd.DataFrame.equals(result_df, expected_result)

    def test_is_type_registered(self):
        """Tests checking if provided feature type registered in the system."""
        assert FeatureTypeManager.is_type_registered(CustomType3) is False
        assert FeatureTypeManager.is_type_registered("custom_type3") is False
        assert FeatureTypeManager.is_type_registered("string") is True
        assert FeatureTypeManager.is_type_registered(Boolean) is True
