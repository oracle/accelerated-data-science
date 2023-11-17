#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
import yaml

from ads.opctl.operator.common.operator_yaml_generator import YamlGenerator


class TestOperatorYamlGenerator:
    """Tests class for generating the YAML config based on the given YAML schema."""

    @pytest.mark.parametrize(
        "schema, values, expected_result",
        [
            # Test case: Basic schema with required and default values
            (
                {
                    "key1": {
                        "type": "string",
                        "default": "test_value",
                        "required": True,
                    },
                    "key2": {"type": "number", "default": 42, "required": True},
                },
                {},
                {"key1": "test_value", "key2": 42},
            ),
            # Test case: Basic schema with required and default values
            (
                {
                    "key1": {"type": "string", "required": True},
                    "key2": {"type": "number", "default": 42, "required": True},
                },
                {"key1": "test_value"},
                {"key1": "test_value", "key2": 42},
            ),
            # Test case: Basic schema with required and default values
            (
                {
                    "key1": {"type": "string", "required": True},
                    "key2": {"type": "number", "default": 42},
                },
                {"key1": "test_value"},
                {"key1": "test_value"},
            ),
            # Test case: Schema with dependencies
            (
                {
                    "model": {"type": "string", "required": True, "default": "prophet"},
                    "owner_name": {
                        "type": "string",
                        "dependencies": {"model": "prophet"},
                    },
                },
                {"owner_name": "value"},
                {"model": "prophet", "owner_name": "value"},
            ),
            # Test case: Schema with dependencies
            (
                {
                    "model": {"type": "string", "required": True, "default": "prophet"},
                    "owner_name": {
                        "type": "string",
                        "dependencies": {"model": "prophet1"},
                    },
                },
                {"owner_name": "value"},
                {"model": "prophet"},
            ),
        ],
    )
    def test_generate_example(self, schema, values, expected_result):
        yaml_generator = YamlGenerator(schema=schema)
        yaml_config = yaml_generator.generate_example(values)
        assert yaml_config == yaml.dump(expected_result)
