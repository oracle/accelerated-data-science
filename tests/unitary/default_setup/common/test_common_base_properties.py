#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import dataclass
from typing import Union
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from ads.model.base_properties import BaseProperties
from ads.common.config import Config, ConfigSection


@dataclass(repr=False)
class MockTestProperties(BaseProperties):
    name: str = None
    age: int = None
    is_boolean: bool = None
    union_type_value: Union[int, float] = None


class TestBaseProperties:
    def setup_method(self):
        self.mock_test_properties = MockTestProperties()

    @pytest.mark.parametrize("test_value", [12, {"a": "a"}, []])
    def test__setattr__fail(self, test_value):
        """Ensures setattr fails when input attribute has wrong type."""
        expected_value = self.mock_test_properties.name
        with pytest.raises(TypeError):
            self.mock_test_properties.name = test_value
        assert self.mock_test_properties.name == expected_value

    @pytest.mark.parametrize("test_value", ["", {"a": "a"}, []])
    def test__setattr__fail_union_type(self, test_value):
        """Ensures setattr fails when input attribute has wrong type."""
        expected_value = self.mock_test_properties.union_type_value
        with pytest.raises(TypeError):
            self.mock_test_properties.union_type_value = test_value
        assert self.mock_test_properties.union_type_value == expected_value

    @pytest.mark.parametrize("test_value", [None, 10, 0, -1])
    def test__setattr__success(self, test_value):
        """Ensures setattr passes when input attribute has valid type."""
        self.mock_test_properties.age = test_value
        assert self.mock_test_properties.age == test_value

    def test_with_prop(self):
        """Tests setting property value."""
        self.mock_test_properties.with_prop("name", "test_name")
        assert self.mock_test_properties.name == "test_name"
        self.mock_test_properties.with_prop("age", 10)
        assert self.mock_test_properties.age == 10
        self.mock_test_properties.with_prop("union_type_value", 10)
        assert self.mock_test_properties.union_type_value == 10
        self.mock_test_properties.with_prop("union_type_value", 10.3)
        assert self.mock_test_properties.union_type_value == 10.3

    @pytest.mark.parametrize(
        "input_data", [MagicMock(), "test_value", "", 0, "test_value", 10]
    )
    def test_with_dict_fail(self, input_data):
        """Ensures populating properties from a dict fails in case of wrong input params."""
        with pytest.raises(TypeError, match="The `obj_dict` should be a dictionary."):
            self.mock_test_properties.from_dict(input_data)

    @pytest.mark.parametrize(
        "input_data, expected_result",
        [
            (
                {
                    "name": "test_name",
                    "age": 10,
                    "is_boolean": True,
                    "union_type_value": 1,
                },
                {
                    "name": "test_name",
                    "age": 10,
                    "is_boolean": True,
                    "union_type_value": 1,
                },
            ),
            (
                {
                    "name": "test_name",
                    "age": "10",
                    "is_boolean": "true",
                    "union_type_value": 10.3,
                },
                {
                    "name": "test_name",
                    "age": 10,
                    "is_boolean": True,
                    "union_type_value": 10.3,
                },
            ),
            (
                {
                    "name": "test_name",
                    "age": "10",
                    "is_boolean": "false",
                    "union_type_value": 1,
                },
                {
                    "name": "test_name",
                    "age": 10,
                    "is_boolean": False,
                    "union_type_value": 1,
                },
            ),
        ],
    )
    def test_with_dict_success(self, input_data, expected_result):
        """Ensures populating properties from a dict passes in case of valid input params."""
        self.mock_test_properties.with_dict(input_data)
        assert self.mock_test_properties.name == expected_result["name"]
        assert self.mock_test_properties.age == expected_result["age"]
        assert self.mock_test_properties.is_boolean == expected_result["is_boolean"]
        assert (
            self.mock_test_properties.union_type_value
            == expected_result["union_type_value"]
        )

    @patch.object(BaseProperties, "with_dict")
    def test_from_dict(self, mock_with_dict):
        """Tests creating an instance of the properties class from a dictionary."""
        input_data = {
            "name": "test_name",
            "age": 10,
            "is_boolean": True,
            "union_type_value": 10.3,
        }
        self.mock_test_properties.from_dict(input_data)
        mock_with_dict.assert_called_with(input_data)

    @patch.object(BaseProperties, "_adjust_with_env")
    @mock.patch.dict(os.environ, {"NAME": "test_name", "AGE": "10"}, clear=True)
    def test_with_env(self, mock_adjust_with_env):
        """Tests setting properties values from environment variables."""
        self.mock_test_properties.with_env()
        assert self.mock_test_properties.name == "test_name"
        assert self.mock_test_properties.age == 10
        mock_adjust_with_env.assert_called()

    def test_to_dict(self):
        """Tests serializing instance of class into a dictionary."""
        self.mock_test_properties.name = "test_name"
        self.mock_test_properties.age = 10
        self.mock_test_properties.is_boolean = True
        self.mock_test_properties.union_type_value = 10

        test_result = self.mock_test_properties.to_dict()
        expected_result = {
            "name": "test_name",
            "age": 10,
            "is_boolean": True,
            "union_type_value": 10,
        }
        assert test_result == expected_result

    def test_with_config_fail(self):
        """Ensures setting properties from config fails in case of wrong input data."""
        with pytest.raises(TypeError):
            self.mock_test_properties.with_config(MagicMock())

    def test_with_config_success(self):
        "Ensures setting properties from config passes with the valid input data."
        test_config_section = ConfigSection()
        test_config_section.with_dict({"name": "test_name", "age": "10"})
        self.mock_test_properties.with_config(test_config_section)
        assert self.mock_test_properties.name == "test_name"
        assert self.mock_test_properties.age == 10

    @patch.object(Config, "__init__")
    @patch.object(Config, "section_set")
    @patch.object(Config, "save")
    def test_to_config(
        self, mock_config_save, mock_config_section_set, mock_config_init
    ):
        """Tests saving properties to the config file."""
        mock_config_init.return_value = None
        self.mock_test_properties.to_config(
            uri="test_uri",
            profile="test_profile",
            force_overwrite=True,
            auth={"auth": ""},
        )
        mock_config_init.assert_called_with(uri="test_uri", auth={"auth": ""})
        mock_config_section_set.assert_called_with(
            key="test_profile", info=self.mock_test_properties.to_dict(), replace=True
        )
        mock_config_save.assert_called_once()

    @patch.object(Config, "load")
    def test_from_config(self, mock_config_load):
        """Tests loading properties from the config file."""
        test_config = Config(uri="test_uri", auth={"auth": ""})
        test_config.with_dict({"TEST_SECTION": {"name": "test_name", "age": "10"}})
        mock_config_load.return_value = test_config
        result_test_properties = MockTestProperties.from_config(
            uri="test_uri", profile="TEST_SECTION", auth={"auth": ""}
        )
        assert result_test_properties.name == "test_name"
        assert result_test_properties.age == 10
        assert result_test_properties.is_boolean == None
