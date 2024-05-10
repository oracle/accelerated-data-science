#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest
from unittest.mock import MagicMock, patch
from importlib import reload
from parameterized import parameterized

import ads.config
import ads.aqua
from notebook.base.handlers import IPythonHandler
from ads.aqua.extension.ui_handler import AquaUIHandler
from ads.aqua.ui import AquaUIApp
from ads.aqua.data import Tags


class TestDataset:
    USER_COMPARTMENT_ID = "ocid1.compartment.oc1..<USER_COMPARTMENT_OCID>"
    USER_PROJECT_ID = "ocid1.datascienceproject.oc1.iad.<USER_PROJECT_OCID>"
    DEPLOYMENT_SHAPE_NAME = "VM.GPU.A10.1"


class TestAquaUIHandler(unittest.TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.ui_handler = AquaUIHandler(MagicMock(), MagicMock())
        self.ui_handler.request = MagicMock()
        self.ui_handler.finish = MagicMock()

    @classmethod
    def setUpClass(cls):
        os.environ["PROJECT_COMPARTMENT_OCID"] = TestDataset.USER_COMPARTMENT_ID
        os.environ["PROJECT_OCID"] = TestDataset.USER_PROJECT_ID
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.extension.ui_handler)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("PROJECT_COMPARTMENT_OCID", None)
        os.environ.pop("PROJECT_OCID", None)
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.extension.ui_handler)

    @patch.object(AquaUIApp, "list_log_groups")
    def test_list_log_groups(self, mock_list_log_groups):
        """Test get method to fetch log groups"""
        self.ui_handler.request.path = "aqua/logging"
        self.ui_handler.get(id="")
        mock_list_log_groups.assert_called_with(
            compartment_id=TestDataset.USER_COMPARTMENT_ID
        )

    @patch.object(AquaUIApp, "list_logs")
    def test_list_logs(self, mock_list_logs):
        """Test get method to fetch logs for a given log group."""
        self.ui_handler.request.path = "aqua/logging"
        self.ui_handler.get(id="mock-log-id")
        mock_list_logs.assert_called_with(log_group_id="mock-log-id")

    @patch.object(AquaUIApp, "list_compartments")
    def test_list_compartments(self, mock_list_compartments):
        """Test get method to fetch list of compartments."""
        self.ui_handler.request.path = "aqua/compartments"
        self.ui_handler.get()
        mock_list_compartments.assert_called()

    @patch.object(AquaUIApp, "list_containers")
    def test_list_containers(self, mock_list_containers):
        """Test get method to fetch list of containers."""
        self.ui_handler.request.path = "aqua/containers"
        self.ui_handler.get()
        mock_list_containers.assert_called()

    @patch.object(AquaUIApp, "get_default_compartment")
    def test_get_default_compartment(self, mock_get_default_compartment):
        """Test get method to fetch default compartment."""
        self.ui_handler.request.path = "aqua/compartments/default"
        self.ui_handler.get()
        mock_get_default_compartment.assert_called()

    @patch.object(AquaUIApp, "list_model_version_sets")
    def test_list_experiments(self, mock_list_experiments):
        """Test get method to fetch list of experiments."""
        self.ui_handler.request.path = "aqua/experiment"
        self.ui_handler.get()
        mock_list_experiments.assert_called_with(
            compartment_id=TestDataset.USER_COMPARTMENT_ID,
            target_tag=Tags.AQUA_EVALUATION.value,
        )

    @patch.object(AquaUIApp, "list_model_version_sets")
    def test_list_model_version_sets(self, mock_list_model_version_sets):
        """Test get method to fetch version sets."""
        self.ui_handler.request.path = "aqua/versionsets"
        self.ui_handler.get()
        mock_list_model_version_sets.assert_called_with(
            compartment_id=TestDataset.USER_COMPARTMENT_ID,
            target_tag=Tags.AQUA_FINE_TUNING.value,
        )

    @parameterized.expand(["true", ""])
    @patch.object(AquaUIApp, "list_buckets")
    def test_list_buckets(self, versioned, mock_list_buckets):
        """Test get method to fetch list of buckets."""
        self.ui_handler.request.path = "aqua/buckets"
        args = {"versioned": versioned}
        self.ui_handler.get_argument = MagicMock(
            side_effect=lambda arg, default=None: args.get(arg, default)
        )
        self.ui_handler.get()
        mock_list_buckets.assert_called_with(
            compartment_id=TestDataset.USER_COMPARTMENT_ID,
            versioned=True if versioned == "true" else False,
        )

    @patch.object(AquaUIApp, "list_job_shapes")
    def test_list_job_shapes(self, mock_list_job_shapes):
        """Test get method to fetch jobs shapes list."""
        self.ui_handler.request.path = "aqua/job/shapes"
        self.ui_handler.get()
        mock_list_job_shapes.assert_called_with(
            compartment_id=TestDataset.USER_COMPARTMENT_ID
        )

    @patch.object(AquaUIApp, "list_vcn")
    def test_list_vcn(self, mock_list_vcn):
        """Test get method to fetch list of vcns."""
        self.ui_handler.request.path = "aqua/vcn"
        self.ui_handler.get()
        mock_list_vcn.assert_called_with(compartment_id=TestDataset.USER_COMPARTMENT_ID)

    @patch.object(AquaUIApp, "list_subnets")
    def test_list_subnets(self, mock_list_subnets):
        """Test the get method to fetch list of subnets."""
        self.ui_handler.request.path = "aqua/subnets"
        args = {"vcn_id": "mock-vcn-id"}
        self.ui_handler.get_argument = MagicMock(
            side_effect=lambda arg, default=None: args.get(arg, default)
        )
        self.ui_handler.get()
        mock_list_subnets.assert_called_with(
            compartment_id=TestDataset.USER_COMPARTMENT_ID, vcn_id="mock-vcn-id"
        )

    @patch.object(AquaUIApp, "get_shape_availability")
    def test_get_shape_availability(self, mock_get_shape_availability):
        """Test get shape availability."""
        self.ui_handler.request.path = "aqua/shapes/limit"
        args = {"instance_shape": TestDataset.DEPLOYMENT_SHAPE_NAME}
        self.ui_handler.get_argument = MagicMock(
            side_effect=lambda arg, default=None: args.get(arg, default)
        )
        self.ui_handler.get()
        mock_get_shape_availability.assert_called_with(
            compartment_id=TestDataset.USER_COMPARTMENT_ID,
            instance_shape=TestDataset.DEPLOYMENT_SHAPE_NAME,
        )

    @patch.object(AquaUIApp, "is_bucket_versioned")
    def test_is_bucket_versioned(self, mock_is_bucket_versioned):
        """Test get method to check if a bucket is versioned."""
        self.ui_handler.request.path = "aqua/bucket/versioning"
        args = {"bucket_uri": "oci://<bucket_name>@<namespace>/<prefix>"}
        self.ui_handler.get_argument = MagicMock(
            side_effect=lambda arg, default=None: args.get(arg, default)
        )
        self.ui_handler.get()
        mock_is_bucket_versioned.assert_called_with(bucket_uri=args["bucket_uri"])
