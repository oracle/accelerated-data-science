#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
import pytest

try:
    from ads.jobs import env_var_parser
except ImportError:
    raise unittest.SkipTest("Jobs API is not available.")


class EnvVarParserTestCase(unittest.TestCase):
    """Contains tests for environment variable parsing for creating jobs."""

    def assert_env_var_parsing(self, envs, expected):
        parsed = env_var_parser.parse(envs)
        self.assertEqual(parsed, expected)

    def test_env_parser_without_ref(self):
        envs = {"A": "Hello", "B": "World"}
        expected = envs
        self.assert_env_var_parsing(envs, expected)

    def test_env_parser_with_ref(self):
        envs = {"A": "Hello", "B": "${A} World"}
        expected = {"A": "Hello", "B": "Hello World"}
        self.assert_env_var_parsing(envs, expected)

    def test_env_parser_with_escape(self):
        envs = {"A": "Hello", "B": "$${A} World"}
        expected = {"A": "Hello", "B": "${A} World"}
        self.assert_env_var_parsing(envs, expected)

    def test_env_parser_with_invalid_syntax(self):
        envs = {"A": "Hello", "B": "$(A) World"}
        expected = {"A": "Hello", "B": "$(A) World"}
        self.assert_env_var_parsing(envs, expected)

    def test_env_parser_with_missing_var(self):
        envs = {"A": "Hello", "B": "A ${C} World"}
        expected = {"A": "Hello", "B": "A ${C} World"}
        self.assert_env_var_parsing(envs, expected)

    @pytest.mark.skip("This case is not handled.")
    def test_env_parser_with_nested_ref(self):
        envs = {
            "A": "Hello",
            "NestedHello": "Hi",
            "B": "${Nested${A}} World",
        }
        expected = {
            "NestedHello": "Hi",
            "A": "Hello",
            "B": "Hi World",
        }
        self.assert_env_var_parsing(envs, expected)
