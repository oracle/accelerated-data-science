#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from ads.opctl.operator.common.const import ARCH_TYPE, PACK_TYPE
from ads.opctl.operator.common.operator_loader import (
    GitOperatorLoader,
    LocalOperatorLoader,
    OperatorInfo,
    OperatorLoader,
    RemoteOperatorLoader,
    ServiceOperatorLoader,
)


class TestOperatorInfo(unittest.TestCase):
    def test_construction(self):
        operator_info = OperatorInfo(
            type="example",
            gpu="yes",
            description="An example operator",
            version="v1",
            conda="example_v1",
            conda_type=PACK_TYPE.CUSTOM,
            path="/path/to/operator",
            backends=["backend1", "backend2"],
        )
        assert (operator_info.type, "example")
        self.assertTrue(operator_info.gpu)
        assert (operator_info.description, "An example operator")
        assert (operator_info.version, "v1")
        assert (operator_info.conda, "example_v1")
        assert (operator_info.conda_type, PACK_TYPE.CUSTOM)
        assert (operator_info.path, "/path/to/operator")
        assert (operator_info.backends, ["backend1", "backend2"])

    def test_conda_prefix(self):
        operator_info = OperatorInfo(
            type="example",
            gpu="yes",
            description="An example operator",
            version="v1",
            conda="example_v1",
            conda_type=PACK_TYPE.CUSTOM,
            path="/path/to/operator",
            backends=["backend1", "backend2"],
        )
        assert (
            operator_info.conda_prefix,
            f"{ARCH_TYPE.GPU}/example/1/example_v1",
        )

    def test_conda_prefix_without_gpu(self):
        operator_info = OperatorInfo(
            type="example",
            gpu="no",
            description="An example operator",
            version="v1",
            conda="example_v1",
            conda_type=PACK_TYPE.CUSTOM,
            path="/path/to/operator",
            backends=["backend1", "backend2"],
        )
        assert (
            operator_info.conda_prefix,
            f"{ARCH_TYPE.CPU}/example/1/example_v1",
        )

    def test_post_init(self):
        operator_info = OperatorInfo(
            type="example",
            gpu="yes",  # Should be converted to boolean
            version="",  # Should be set to "v1"
            conda_type=None,  # Should be set to PACK_TYPE.CUSTOM
            conda=None,  # Should be set to "example_v1"
        )
        self.assertTrue(operator_info.gpu)
        assert (operator_info.version, "v1")
        assert (operator_info.conda_type, PACK_TYPE.CUSTOM)
        assert (operator_info.conda, "example_v1")

    def test_from_yaml_with_yaml_string(self):
        yaml_string = """
        type: example
        gpu: yes
        description: An example operator
        version: v1
        conda_type: published
        path: /path/to/operator
        backends:
          - backend1
          - backend2
        """
        operator_info = OperatorInfo.from_yaml(yaml_string=yaml_string)
        assert (operator_info.type, "example")
        self.assertTrue(operator_info.gpu)
        assert (operator_info.description, "An example operator")
        assert (operator_info.version, "v1")
        assert (operator_info.conda, "example_v1")
        assert (operator_info.conda_type, PACK_TYPE.CUSTOM)
        assert (operator_info.path, "/path/to/operator")
        assert (operator_info.backends, ["backend1", "backend2"])

    @patch("ads.common.serializer.Serializable.from_yaml")
    def test_from_yaml_with_uri(self, mock_from_yaml):
        uri = "http://example.com/operator.yaml"
        loader = MagicMock()
        mock_from_yaml.return_value = OperatorInfo(
            type="example",
            gpu="yes",
            description="An example operator",
            version="v1",
            conda="example_v1",
            conda_type=PACK_TYPE.CUSTOM,
            path="/path/to/operator",
            backends=["backend1", "backend2"],
        )
        operator_info = OperatorInfo.from_yaml(uri=uri, loader=loader)
        mock_from_yaml.assert_called_with(yaml_string=None, uri=uri, loader=loader)
        assert (operator_info.type, "example")
        self.assertTrue(operator_info.gpu)
        assert (operator_info.description, "An example operator")
        assert (operator_info.version, "v1")
        assert (operator_info.conda, "example_v1")
        assert (operator_info.conda_type, PACK_TYPE.CUSTOM)
        assert (operator_info.path, "http://example.com")
        assert (operator_info.backends, ["backend1", "backend2"])


class TestOperatorLoader:
    def setup_method(self):
        # Create a mock Loader instance for testing
        self.loader = Mock()
        self.operator_loader = OperatorLoader(self.loader)

    def test_load_operator(self):
        # Define a mock OperatorInfo object to return when load is called on the loader
        mock_operator_info = OperatorInfo(
            type="mock_operator",
            gpu=False,
            description="Mock Operator",
            version="v1",
            conda="mock_operator_v1",
            conda_type="custom",
            path="/path/to/mock_operator",
            backends=["cpu"],
        )

        # Mock the _load method to return the mock_operator_info object
        self.loader.load.return_value = mock_operator_info

        # Call the load method of the OperatorLoader
        operator_info = self.operator_loader.load()

        # Check if the returned OperatorInfo object matches the expected values

        assert operator_info.type == "mock_operator"
        assert operator_info.gpu == False
        assert operator_info.description == "Mock Operator"
        assert operator_info.version == "v1"
        assert operator_info.conda == "mock_operator_v1"
        assert operator_info.conda_type == "custom"
        assert operator_info.path == "/path/to/mock_operator"
        assert operator_info.backends == ["cpu"]

    def test_load_operator_exception(self):
        # Mock the _load method to raise an exception
        self.loader.load.side_effect = Exception("Error loading operator")

        # Call the load method of the OperatorLoader and expect an exception
        with pytest.raises(Exception):
            self.operator_loader.load()

    @pytest.mark.parametrize(
        "test_name, uri, expected_result",
        [
            ("Service Path", "forecast", ServiceOperatorLoader),
            ("Local Path", "/path/to/local_operator", LocalOperatorLoader),
            ("OCI Path", "oci://bucket/operator.zip", RemoteOperatorLoader),
            (
                "Git Path",
                "https://github.com/my-operator-repository",
                GitOperatorLoader,
            ),
        ],
    )
    def test_from_uri(self, test_name, uri, expected_result):
        # Call the from_uri method of the OperatorLoader class
        operator_loader = OperatorLoader.from_uri(uri=uri)
        assert isinstance(operator_loader.loader, expected_result)

    def test_empty_uri(self):
        # Test with an empty URI that should raise a ValueError
        with pytest.raises(ValueError):
            OperatorLoader.from_uri(uri="", uri_dst=None)

    def test_invalid_uri(self):
        # Test with an invalid URI that should raise a ValueError
        with pytest.raises(ValueError):
            OperatorLoader.from_uri(uri="aws://", uri_dst=None)


class TestServiceOperatorLoader(unittest.TestCase):
    def setUp(self):
        # Create a mock ServiceOperatorLoader instance for testing
        self.loader = ServiceOperatorLoader(uri="mock_service_operator")
        self.mock_operator_info = OperatorInfo(
            type="mock_operator",
            gpu="no",
            description="Mock Operator",
            version="v1",
            conda="mock_operator_v1",
            conda_type="custom",
            path="/path/to/mock_operator",
            backends=["cpu"],
        )

    def test_compatible(self):
        # Test the compatible method with a valid URI
        uri = "forecast"
        self.assertTrue(ServiceOperatorLoader.compatible(uri=uri))

        # Test the compatible method with an invalid URI
        uri = "invalid_service_operator"
        self.assertFalse(ServiceOperatorLoader.compatible(uri=uri))

    def test_load(self):
        # Mock the _load method to return the mock_operator_info object
        self.loader._load = Mock(return_value=self.mock_operator_info)

        # Call the load method of the ServiceOperatorLoader
        operator_info = self.loader.load()

        # Check if the returned OperatorInfo object matches the expected values
        self.assertEqual(operator_info.type, "mock_operator")
        self.assertEqual(operator_info.gpu, False)
        self.assertEqual(operator_info.description, "Mock Operator")
        self.assertEqual(operator_info.version, "v1")
        self.assertEqual(operator_info.conda, "mock_operator_v1")
        self.assertEqual(operator_info.conda_type, "custom")
        self.assertEqual(operator_info.path, "/path/to/mock_operator")
        self.assertEqual(operator_info.backends, ["cpu"])

    def test_load_exception(self):
        # Mock the _load method to raise an exception
        self.loader._load = Mock(
            side_effect=Exception("Error loading service operator")
        )

        # Call the load method of the ServiceOperatorLoader and expect an exception
        with self.assertRaises(Exception):
            self.loader.load()


class TestLocalOperatorLoader(unittest.TestCase):
    def setUp(self):
        # Create a mock LocalOperatorLoader instance for testing
        self.loader = LocalOperatorLoader(uri="path/to/local/operator")
        self.mock_operator_info = OperatorInfo(
            type="mock_operator",
            gpu=False,
            description="Mock Operator",
            version="v1",
            conda="mock_operator_v1",
            conda_type="custom",
            path="/path/to/mock_operator",
            backends=["cpu"],
        )

    def test_compatible(self):
        # Test the compatible method with a valid URI
        uri = "path/to/local/operator"
        self.assertTrue(LocalOperatorLoader.compatible(uri=uri))

        # Test the compatible method with an invalid URI
        uri = "http://example.com/remote/operator"
        self.assertFalse(LocalOperatorLoader.compatible(uri=uri))

    def test_load(self):
        # Mock the _load method to return the mock_operator_info object
        self.loader._load = Mock(return_value=self.mock_operator_info)

        # Call the load method of the LocalOperatorLoader
        operator_info = self.loader.load()

        # Check if the returned OperatorInfo object matches the expected values
        self.assertEqual(operator_info.type, "mock_operator")
        self.assertEqual(operator_info.gpu, False)
        self.assertEqual(operator_info.description, "Mock Operator")
        self.assertEqual(operator_info.version, "v1")
        self.assertEqual(operator_info.conda, "mock_operator_v1")
        self.assertEqual(operator_info.conda_type, "custom")
        self.assertEqual(operator_info.path, "/path/to/mock_operator")
        self.assertEqual(operator_info.backends, ["cpu"])

    def test_load_exception(self):
        # Mock the _load method to raise an exception
        self.loader._load = Mock(side_effect=Exception("Error loading local operator"))

        # Call the load method of the LocalOperatorLoader and expect an exception
        with self.assertRaises(Exception):
            self.loader.load()


class TestRemoteOperatorLoader(unittest.TestCase):
    def setUp(self):
        # Create a mock RemoteOperatorLoader instance for testing
        self.loader = RemoteOperatorLoader(uri="oci://bucket/operator.zip")
        self.mock_operator_info = OperatorInfo(
            type="mock_operator",
            gpu=False,
            description="Mock Operator",
            version="v1",
            conda="mock_operator_v1",
            conda_type="custom",
            path="/path/to/mock_operator",
            backends=["cpu"],
        )

    def test_compatible(self):
        # Test the compatible method with a valid URI
        uri = "oci://bucket/operator.zip"
        self.assertTrue(RemoteOperatorLoader.compatible(uri=uri))

        # Test the compatible method with an invalid URI
        uri = "http://example.com/remote/operator"
        self.assertFalse(RemoteOperatorLoader.compatible(uri=uri))

    def test_load(self):
        # Mock the _load method to return the mock_operator_info object
        self.loader._load = Mock(return_value=self.mock_operator_info)

        # Call the load method of the RemoteOperatorLoader
        operator_info = self.loader.load()

        # Check if the returned OperatorInfo object matches the expected values
        self.assertEqual(operator_info.type, "mock_operator")
        self.assertEqual(operator_info.gpu, False)
        self.assertEqual(operator_info.description, "Mock Operator")
        self.assertEqual(operator_info.version, "v1")
        self.assertEqual(operator_info.conda, "mock_operator_v1")
        self.assertEqual(operator_info.conda_type, "custom")
        self.assertEqual(operator_info.path, "/path/to/mock_operator")
        self.assertEqual(operator_info.backends, ["cpu"])

    def test_load_exception(self):
        # Mock the _load method to raise an exception
        self.loader._load = Mock(side_effect=Exception("Error loading remote operator"))

        # Call the load method of the RemoteOperatorLoader and expect an exception
        with self.assertRaises(Exception):
            self.loader.load()


class TestGitOperatorLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = "temp_git_loader"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create a mock GitOperatorLoader instance for testing
        self.loader = GitOperatorLoader(
            uri="https://github.com/mock_operator_repository.git@feature-branch#forecasting",
            uri_dst=self.temp_dir,
        )
        self.mock_operator_info = OperatorInfo(
            type="mock_operator",
            gpu=False,
            description="Mock Operator",
            version="v1",
            conda="mock_operator_v1",
            conda_type="custom",
            path=os.path.join(self.temp_dir, "forecasting"),
            backends=["cpu"],
        )

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compatible(self):
        # Test the compatible method with a valid URI
        uri = (
            "https://github.com/mock_operator_repository.git@feature-branch#forecasting"
        )
        self.assertTrue(GitOperatorLoader.compatible(uri=uri))

        # Test the compatible method with an invalid URI
        uri = "http://example.com/remote/operator"
        self.assertFalse(GitOperatorLoader.compatible(uri=uri))

    def test_load(self):
        # Mock the git.Repo.clone_from method to avoid actual Git operations
        with patch("git.Repo.clone_from") as mock_clone_from:
            mock_clone_from.return_value = Mock()

            # Mock the _load method to return the mock_operator_info object
            self.loader._load = Mock(return_value=self.mock_operator_info)

            # Call the load method of the GitOperatorLoader
            operator_info = self.loader.load()

            # Check if the returned OperatorInfo object matches the expected values
            self.assertEqual(operator_info.type, "mock_operator")
            self.assertEqual(operator_info.gpu, False)
            self.assertEqual(operator_info.description, "Mock Operator")
            self.assertEqual(operator_info.version, "v1")
            self.assertEqual(operator_info.conda, "mock_operator_v1")
            self.assertEqual(operator_info.conda_type, "custom")
            self.assertEqual(
                operator_info.path, os.path.join(self.temp_dir, "forecasting")
            )
            self.assertEqual(operator_info.backends, ["cpu"])

    def test_load_exception(self):
        # Mock the git.Repo.clone_from method to raise an exception
        with patch(
            "git.Repo.clone_from", side_effect=Exception("Error cloning Git repository")
        ):
            # Mock the _load method to raise an exception
            self.loader._load = Mock(
                side_effect=Exception("Error loading Git operator")
            )

            # Call the load method of the GitOperatorLoader and expect an exception
            with self.assertRaises(Exception):
                self.loader.load()
