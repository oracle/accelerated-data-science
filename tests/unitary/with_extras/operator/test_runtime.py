#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from my_module.runtime import ContainerRuntime, PythonRuntime


class TestContainerRuntime:
    def test_init(self):
        runtime = ContainerRuntime(image="my-image", command="my-command")
        self.assertEqual(runtime.image, "my-image")
        self.assertEqual(runtime.command, "my-command")

    def test_to_dict(self):
        runtime = ContainerRuntime(image="my-image", command="my-command")
        runtime_dict = runtime.to_dict()
        self.assertIsInstance(runtime_dict, dict)
        self.assertEqual(runtime_dict["type"], "container")
        self.assertEqual(runtime_dict["image"], "my-image")
        self.assertEqual(runtime_dict["command"], "my-command")

    def test_from_dict(self):
        runtime_dict = {
            "type": "container",
            "image": "my-image",
            "command": "my-command",
        }
        runtime = ContainerRuntime.from_dict(runtime_dict)
        self.assertIsInstance(runtime, ContainerRuntime)
        self.assertEqual(runtime.image, "my-image")
        self.assertEqual(runtime.command, "my-command")


class TestPythonRuntime:
    def test_init(self):
        runtime = PythonRuntime(version="v2")
        self.assertEqual(runtime.version, "v2")
        self.assertEqual(runtime.type, "python")

    def test_to_dict(self):
        runtime = PythonRuntime(version="v2")
        runtime_dict = runtime.to_dict()
        self.assertIsInstance(runtime_dict, dict)
        self.assertEqual(runtime_dict["type"], "python")
        self.assertEqual(runtime_dict["version"], "v2")

    def test_from_dict(self):
        runtime_dict = {"type": "python", "version": "v2"}
        runtime = PythonRuntime.from_dict(runtime_dict)
        self.assertIsInstance(runtime, PythonRuntime)
        self.assertEqual(runtime.version, "v2")
        self.assertEqual(runtime.type, "python")
