#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import json
import unittest
from importlib import reload
from unittest.mock import MagicMock, patch
import pytest
from parameterized import parameterized

import oci
import ads.config
from ads.aqua.ui import AquaUIApp
from ads.aqua.exception import AquaValueError
from ads.aqua.utils import load_config
from ads.config import AQUA_CONFIG_FOLDER, AQUA_RESOURCE_LIMIT_NAMES_CONFIG


class TestDataset:
    SERVICE_COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"
    USER_COMPARTMENT_ID = "ocid1.compartment.oc1..<USER_COMPARTMENT_OCID>"
    TENANCY_OCID = "ocid1.tenancy.oc1..<OCID>"
    VCN_ID = "ocid1.vcn.oc1.iad.<OCID>"
    DEPLOYMENT_SHAPE_NAMES = ["VM.GPU.A10.1", "BM.GPU4.8", "VM.Standard.2.16"]


class TestAquaUI(unittest.TestCase):
    def setUp(self):
        self.app = AquaUIApp()
        self.app.region = "region-name"

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))
        os.environ["CONDA_BUCKET_NS"] = "test-namespace"
        os.environ["ODSC_MODEL_COMPARTMENT_OCID"] = TestDataset.SERVICE_COMPARTMENT_ID
        os.environ["PROJECT_COMPARTMENT_OCID"] = TestDataset.USER_COMPARTMENT_ID
        os.environ["TENANCY_OCID"] = TestDataset.TENANCY_OCID
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.ui)

    @classmethod
    def tearDownClass(cls):
        cls.curr_dir = None
        os.environ.pop("CONDA_BUCKET_NS", None)
        os.environ.pop("ODSC_MODEL_COMPARTMENT_OCID", None)
        os.environ.pop("PROJECT_COMPARTMENT_OCID", None)
        os.environ.pop("TENANCY_OCID", None)
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.ui)

    def test_list_log_groups(self):
        """Test to lists all log groups for the specified compartment or tenancy"""
        log_groups_list = os.path.join(
            self.curr_dir, "test_data/ui/log_groups_list.json"
        )
        with open(log_groups_list, "r") as _file:
            log_groups = json.load(_file)

        self.app.logging_client.list_log_groups = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=[
                    oci.logging.models.LogGroupSummary(**log_group)
                    for log_group in log_groups
                ],
            )
        )
        results = self.app.list_log_groups()
        expected_attributes = {
            "id",
            "compartmentId",
            "displayName",
            "description",
            "definedTags",
            "freeformTags",
            "timeCreated",
            "timeLastModified",
            "lifecycleState",
        }

        for result in results:
            self.assertTrue(
                expected_attributes.issuperset(set(result)), "Attributes mismatch"
            )
        assert len(results) == len(log_groups)

    def test_list_logs(self):
        """Test to lists all logs for the specified compartment or tenancy"""
        logs_list = os.path.join(self.curr_dir, "test_data/ui/logs_list.json")
        with open(logs_list, "r") as _file:
            logs = json.load(_file)

        self.app.logging_client.list_logs = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=[oci.logging.models.LogSummary(**log) for log in logs],
            )
        )
        results = self.app.list_logs(log_group_id="ocid1.loggroup.oc1.iad.<OCID>")
        expected_attributes = {
            "id",
            "logGroupId",
            "displayName",
            "isEnabled",
            "lifecycleState",
            "logType",
            "definedTags",
            "freeformTags",
            "timeCreated",
            "timeLastModified",
            "retentionDuration",
            "compartmentId",
        }

        for result in results:
            self.assertTrue(
                expected_attributes.issuperset(set(result)), "Attributes mismatch"
            )
        assert len(results) == len(logs)

    def test_list_compartments(self):
        """Test to lists all compartments in a tenancy specified by TENANCY_OCID env variable. Also, clear
        the compartment cache and check if empty."""
        compartments_list = os.path.join(
            self.curr_dir, "test_data/ui/compartments_list.json"
        )
        with open(compartments_list, "r") as _file:
            compartments = json.load(_file)

        self.app.list_resource = MagicMock(
            return_value=[
                oci.identity.models.Compartment(**compartment)
                for compartment in compartments
            ]
        )
        root_compartment = os.path.join(
            self.curr_dir, "test_data/ui/root_compartment.json"
        )
        with open(root_compartment, "r") as _file:
            root = json.load(_file)
        self.app.identity_client.get_compartment = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=oci.identity.models.Compartment(**root),
            )
        )

        results = self.app.list_compartments()
        expected_root_attributes = {
            "id",
            "name",
            "description",
            "timeCreated",
            "lifecycleState",
            "isAccessible",
            "freeformTags",
            "definedTags",
        }
        self.assertTrue(
            expected_root_attributes.issuperset(set(results[0])), "Attributes mismatch"
        )

        expected_child_attributes = {
            "id",
            "compartmentId",
            "name",
            "description",
            "timeCreated",
            "lifecycleState",
            "freeformTags",
            "definedTags",
        }
        for idx in range(1, len(results)):
            self.assertTrue(
                expected_child_attributes.issuperset(set(results[idx])),
                "Attributes mismatch",
            )
        assert len(results) == len(compartments) + 1

        # test to check if cache is clear
        self.assertTrue(TestDataset.TENANCY_OCID in self.app._compartments_cache.keys())
        cache_clear_result = self.app.clear_compartments_list_cache()
        assert cache_clear_result["key"]["tenancy_ocid"] == TestDataset.TENANCY_OCID
        assert len(self.app._compartments_cache) == 0

    def test_get_default_compartment(self):
        """Test to check if user compartment OCID fetched from environment variables and returned."""
        result = self.app.get_default_compartment()
        assert dict(compartment_id=TestDataset.USER_COMPARTMENT_ID) == result

    @patch("ads.aqua.ui.COMPARTMENT_OCID", "")
    @patch("ads.aqua.logger.error")
    def test_get_default_compartment_unavailable(self, mock_logger_error):
        """Test to check if user compartment OCID is returned empty if not available."""
        result = self.app.get_default_compartment()
        assert dict(compartment_id="") == result
        mock_logger_error.assert_called_once_with(
            "No compartment id found from environment variables."
        )

    @parameterized.expand(
        [
            ("experiments", "aqua_evaluation"),
            ("modelversionsets", "aqua_finetuning"),
        ]
    )
    @patch("ads.aqua.logger.info")
    def test_list_model_version_sets(self, api_type, tag, mock_logger_info):
        """Tests to list all model version sets for the specified compartment or tenancy."""

        version_sets_json = os.path.join(
            self.curr_dir, f"test_data/ui/{api_type}_list.json"
        )
        with open(version_sets_json, "r") as _file:
            version_sets = json.load(_file)

        self.app.list_resource = MagicMock(
            return_value=[
                oci.data_science.models.ModelVersionSetSummary(**version_set)
                for version_set in version_sets
            ]
        )
        results = self.app.list_model_version_sets(target_tag=tag)
        mock_logger_info.assert_called_once_with(
            f"Loading {api_type} from compartment: {TestDataset.USER_COMPARTMENT_ID}"
        )
        expected_attributes = {
            "id",
            "compartmentId",
            "projectId",
            "name",
            "lifecycleState",
            "timeCreated",
            "timeUpdated",
            "createdBy",
            "freeformTags",
            "definedTags",
        }
        for result in results:
            self.assertTrue(
                expected_attributes.issuperset(set(result)), "Attributes mismatch"
            )
        assert len(results) == len(version_sets)

    @patch.object(oci.object_storage.ObjectStorageClient, "get_namespace")
    @patch.object(oci.object_storage.ObjectStorageClient, "list_buckets")
    @patch("ads.common.object_storage_details.ObjectStorageDetails.from_path")
    def test_list_buckets(self, mock_from_path, mock_list_buckets, mock_get_namespace):
        """Test to list all buckets for the specified compartment."""

        from ads.common.object_storage_details import ObjectStorageDetails

        buckets_list = os.path.join(self.curr_dir, "test_data/ui/buckets_list.json")
        with open(buckets_list, "r") as _file:
            buckets = json.load(_file)

        mock_get_namespace.return_value.data = "test-namespace"
        mock_list_buckets.return_value.data = [
            oci.object_storage.models.BucketSummary(**bucket) for bucket in buckets
        ]
        results = self.app.list_buckets()

        mock_get_namespace.assert_called_once()
        mock_list_buckets.assert_called_once()
        expected_attributes = {
            "namespace",
            "name",
            "compartmentId",
            "createdBy",
            "timeCreated",
            "etag",
        }

        for result in results:
            self.assertTrue(
                expected_attributes.issuperset(set(result)), "Attributes mismatch"
            )
        assert len(results) == len(buckets)

        # todo :test versioned buckets

    def test_list_job_shapes(self):
        """Tests to list all available job shapes for the specified compartment."""

        job_shapes_list_json = os.path.join(
            self.curr_dir, f"test_data/ui/job_shapes_list.json"
        )
        with open(job_shapes_list_json, "r") as _file:
            job_shapes = json.load(_file)

        self.app.ds_client.list_job_shapes = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=[
                    oci.data_science.models.JobShapeSummary(**job_shape)
                    for job_shape in job_shapes
                ],
            )
        )
        results = self.app.list_job_shapes()
        expected_attributes = {"name", "coreCount", "memoryInGBs", "shapeSeries"}
        for result in results:
            self.assertTrue(
                expected_attributes.issuperset(set(result)), "Attributes mismatch"
            )
        assert len(results) == len(job_shapes)

    @patch.object(oci.core.virtual_network_client.VirtualNetworkClient, "list_vcns")
    def test_list_vcn(self, mock_list_vcns):
        """Test to list the virtual cloud networks (VCNs) in the specified compartment."""
        vcn_list_json = os.path.join(self.curr_dir, f"test_data/ui/vcn_list.json")
        with open(vcn_list_json, "r") as _file:
            vcns = json.load(_file)

        mock_list_vcns.return_value.data = [oci.core.models.Vcn(**vcn) for vcn in vcns]
        results = self.app.list_vcn()

        mock_list_vcns.called_once()
        expected_attributes = {
            "cidrBlock",
            "cidrBlocks",
            "compartmentId",
            "defaultDhcpOptionsId",
            "defaultRouteTableId",
            "defaultSecurityListId",
            "definedTags",
            "displayName",
            "dnsLabel",
            "freeformTags",
            "id",
            "lifecycleState",
            "timeCreated",
            "vcnDomainName",
        }
        for result in results:
            self.assertTrue(
                expected_attributes.issuperset(set(result)), "Attributes mismatch"
            )
        assert len(results) == len(vcns)

    @patch.object(oci.core.virtual_network_client.VirtualNetworkClient, "list_subnets")
    def test_list_subnets(self, mock_list_subnets):
        """Test to list the virtual cloud networks (VCNs) in the specified compartment."""
        subnets_list_json = os.path.join(
            self.curr_dir, f"test_data/ui/subnets_list.json"
        )
        with open(subnets_list_json, "r") as _file:
            subnets = json.load(_file)

        mock_list_subnets.return_value.data = [
            oci.core.models.Subnet(**subnet) for subnet in subnets
        ]
        results = self.app.list_subnets(vcn_id=TestDataset.VCN_ID)

        mock_list_subnets.called_once()
        expected_attributes = {
            "cidrBlock",
            "compartmentId",
            "definedTags",
            "dhcpOptionsId",
            "defaultRouteTableId",
            "displayName",
            "dnsLabel",
            "freeformTags",
            "id",
            "lifecycleState",
            "prohibitInternetIngress",
            "prohibitPublicIpOnVnic",
            "routeTableId",
            "securityListIds",
            "subnetDomainName",
            "timeCreated",
            "vcnId",
            "virtualRouterIp",
            "virtualRouterMac",
        }
        for result in results:
            self.assertTrue(
                expected_attributes.issuperset(set(result)), "Attributes mismatch"
            )
        assert len(results) == len(subnets)

    @patch.object(oci.limits.limits_client.LimitsClient, "get_resource_availability")
    def test_get_shape_availability(self, mock_get_resource_availability):
        """Test whether the function returns the number of available resources associated with the given shape."""
        # todo: parameterize the instance_shape values with @parameterized.expand

        resource_availability_json = os.path.join(
            self.curr_dir, f"test_data/ui/resource_availability.json"
        )
        with open(resource_availability_json, "r") as _file:
            resource_availability = json.load(_file)

        artifact_path = AQUA_CONFIG_FOLDER
        config = load_config(
            artifact_path,
            config_file_name=AQUA_RESOURCE_LIMIT_NAMES_CONFIG,
        )

        mock_get_resource_availability.return_value.data = (
            oci.limits.models.ResourceAvailability(**resource_availability)
        )
        result = self.app.get_shape_availability(
            instance_shape=TestDataset.DEPLOYMENT_SHAPE_NAMES[0]
        )

        mock_get_resource_availability.called_once()
        expected_attributes = {"available_count"}
        self.assertTrue(
            expected_attributes.issuperset(set(result)), "Attributes mismatch"
        )
        assert result["available_count"], resource_availability["available"]

        with pytest.raises(
            AquaValueError,
            match=f"Inadequate resource is available to create the {TestDataset.DEPLOYMENT_SHAPE_NAMES[1]} resource. The number of available "
            f"resource associated with the limit name {config[TestDataset.DEPLOYMENT_SHAPE_NAMES[1]]} is {resource_availability['available']}.",
        ):
            self.app.get_shape_availability(
                instance_shape=TestDataset.DEPLOYMENT_SHAPE_NAMES[1]
            )

        with pytest.raises(
            AquaValueError,
            match=f"instance_shape argument is required.",
        ):
            self.app.get_shape_availability(instance_shape="")

        result = self.app.get_shape_availability(
            instance_shape=TestDataset.DEPLOYMENT_SHAPE_NAMES[2]
        )
        assert result == {}

    @parameterized.expand([True, False])
    @patch("ads.common.object_storage_details.ObjectStorageDetails.from_path")
    def test_is_bucket_versioned(self, versioned, mock_from_path):
        """Tests whether a bucket is versioned or not"""
        mock_from_path.return_value.is_bucket_versioned.return_value = versioned
        result = self.app.is_bucket_versioned("oci://bucket-name-@namespace/prefix")
        assert result["is_versioned"] == versioned
