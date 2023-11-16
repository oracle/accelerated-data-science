#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
from unittest.mock import MagicMock, patch

from ads.opctl.operator.common.errors import OperatorSchemaYamlError
from ads.opctl.operator.runtime.runtime import (
    OPERATOR_LOCAL_RUNTIME_TYPE,
    ContainerRuntime,
    ContainerRuntimeSpec,
    PythonRuntime,
    Runtime,
)


class TestRuntime(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()

    def test_kind(self):
        self.assertEqual(self.runtime.kind, "operator.local")

    def test_type(self):
        self.assertIsNone(self.runtime.type)

    def test_version(self):
        self.assertIsNone(self.runtime.version)

    @patch("ads.opctl.operator.runtime.runtime._load_yaml_from_uri")
    @patch("ads.opctl.operator.runtime.runtime.Validator")
    def test_validate_dict(self, mock_validator, mock_load_yaml):
        mock_validator.return_value.validate.return_value = True
        self.assertTrue(Runtime._validate_dict({}))
        mock_load_yaml.assert_called_once()
        mock_validator.assert_called_once()

    @patch("ads.opctl.operator.runtime.runtime._load_yaml_from_uri")
    @patch("ads.opctl.operator.runtime.runtime.Validator")
    def test_validate_dict_invalid(self, mock_validator, mock_load_yaml):
        mock_validator.return_value = MagicMock(
            errors=[{"error": "error"}], validate=MagicMock(return_value=False)
        )
        mock_validator.return_value.validate.return_value = False
        with self.assertRaises(OperatorSchemaYamlError):
            Runtime._validate_dict({})
        mock_load_yaml.assert_called_once()
        mock_validator.assert_called_once()


class TestContainerRuntime(unittest.TestCase):
    def test_init(self):
        runtime = ContainerRuntime.init(
            image="my-image",
            env=[{"name": "VAR1", "value": "value1"}],
            volume=["/data"],
        )
        self.assertIsInstance(runtime, ContainerRuntime)
        self.assertEqual(runtime.type, OPERATOR_LOCAL_RUNTIME_TYPE.CONTAINER.value)
        self.assertEqual(runtime.version, "v1")
        self.assertIsInstance(runtime.spec, ContainerRuntimeSpec)
        self.assertEqual(runtime.spec.image, "my-image")
        self.assertEqual(runtime.spec.env, [{"name": "VAR1", "value": "value1"}])
        self.assertEqual(runtime.spec.volume, ["/data"])

    def test_validate_dict(self):
        valid_dict = {
            "kind": "operator.local",
            "type": "container",
            "version": "v1",
            "spec": {
                "image": "my-image",
                "env": [{"name": "VAR1", "value": "value1"}],
                "volume": ["/data"],
            },
        }
        self.assertTrue(ContainerRuntime._validate_dict(valid_dict))

        invalid_dict = {
            "kind": "operator.local",
            "type": "unknown",
            "version": "v1",
            "spec": {
                "image": "my-image",
                "env": [{"name": "VAR1"}],
                "volume": ["/data"],
            },
        }
        with self.assertRaises(OperatorSchemaYamlError):
            ContainerRuntime._validate_dict(invalid_dict)


class TestPythonRuntime(unittest.TestCase):
    def test_init(self):
        runtime = PythonRuntime.init()
        self.assertIsInstance(runtime, PythonRuntime)
        self.assertEqual(runtime.type, "python")
        self.assertEqual(runtime.version, "v1")
