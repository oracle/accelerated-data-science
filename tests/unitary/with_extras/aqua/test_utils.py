#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import unittest
from unittest.mock import MagicMock, patch

from oci.object_storage.models import ListObjects, ObjectSummary
from oci.resource_search.models.resource_summary import ResourceSummary
from parameterized import parameterized

from ads.aqua.common import utils
from ads.aqua.common.errors import AquaRuntimeError
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.config import TENANCY_OCID


class TestDataset:
    mock_ocid_1 = "ocid1.datasciencemodeldeployment.oc1.iad.<UNIQUE_OCID>"
    mock_ocid_2 = "ocid1.datasciencejobrun.oc1.iad.<UNIQUE_OCID>"
    mock_compartment_id = "ocid1.compartment.oc1..<UNIQUE_OCID>"
    mock_response = [
        ResourceSummary(
            compartment_id="ocid1.compartment.<unique_ocid>",
            display_name="RBH-Demo-Deployment",
            identifier="ocid1.datasciencemodeldeployment.<unique_ocid>",
            lifecycle_state="ACTIVE",
            resource_type="DataScienceModelDeployment",
            time_created="2021-06-01T18:40:38.586000+00:00",
        )
    ]


class TestAquaUtils(unittest.TestCase):
    @parameterized.expand(
        [
            (
                dict(field_name="freeformTags.key", allowed_values=["aqua_evaluation"]),
                " && (freeformTags.key = 'aqua_evaluation')",
            ),
            (
                dict(
                    field_name="freeformTags.key",
                    allowed_values=["aqua_evaluation", "OCI_AQUA"],
                ),
                " && (freeformTags.key = 'aqua_evaluation' && freeformTags.key = 'OCI_AQUA')",
            ),
            (
                dict(field_name="lifecycleState", allowed_values=None),
                "",
            ),
            (
                dict(
                    field_name="lifecycleState",
                    allowed_values=["ACTIVE", "FAILED"],
                    connect_by_ampersands=False,
                ),
                " && (lifecycleState = 'ACTIVE' || lifecycleState = 'FAILED')",
            ),
        ]
    )
    def test_construct_condition(self, input, expected_output):
        """Tests construct condition under different situations."""
        assert utils._construct_condition(**input) == expected_output

    @parameterized.expand(
        [
            (
                dict(ocid=TestDataset.mock_ocid_1),
                TestDataset.mock_response,
                f"query datasciencemodeldeployment resources return allAdditionalFields where (identifier = '{TestDataset.mock_ocid_1}')",
            ),
            (
                dict(
                    ocid=TestDataset.mock_ocid_2,
                    return_all=False,
                ),
                [],
                f"query datasciencejobrun resources where (identifier = '{TestDataset.mock_ocid_2}')",
            ),
        ]
    )
    @patch.object(OCIResource, "search")
    def test_query_resource(self, input, mock_response, expected_query, mock_search):
        """Tests use Search service to find a single resource."""
        utils.is_valid_ocid = MagicMock(return_value=True)
        mock_search.return_value = mock_response
        if mock_response:
            resource = utils.query_resource(**input)
            mock_search.assert_called_with(
                expected_query, type=SEARCH_TYPE.STRUCTURED, tenant_id=TENANCY_OCID
            )
            assert isinstance(resource, ResourceSummary)
        else:
            with self.assertRaises(AquaRuntimeError):
                resource = utils.query_resource(**input)

    @parameterized.expand(
        [
            (
                dict(
                    compartment_id=TestDataset.mock_compartment_id,
                    resource_type="datasciencemodeldeployment",
                ),
                TestDataset.mock_response,
                f"query datasciencemodeldeployment resources return allAdditionalFields where (compartmentId = '{TestDataset.mock_compartment_id}')",
            ),
            (
                dict(
                    compartment_id=TestDataset.mock_compartment_id,
                    resource_type="datasciencemodeldeployment",
                    return_all=False,
                ),
                [],
                f"query datasciencemodeldeployment resources where (compartmentId = '{TestDataset.mock_compartment_id}')",
            ),
        ]
    )
    @patch.object(OCIResource, "search")
    def test_query_resources(self, input, mock_response, expected_query, mock_search):
        """Tests use Search service to find resources."""
        utils.is_valid_ocid = MagicMock(return_value=True)
        mock_search.return_value = mock_response
        resources = utils.query_resources(**input)
        mock_search.assert_called_with(
            expected_query, type=SEARCH_TYPE.STRUCTURED, tenant_id=TENANCY_OCID
        )
        assert isinstance(resources, list)

    @patch("ads.common.object_storage_details.authutil.default_signer")
    def test_list_os_files_with_invalid_path(self, mock_auth: MagicMock):
        auth = MagicMock()
        auth["signer"] = MagicMock()
        mock_auth.return_value = auth
        with self.assertRaises(Exception) as e:
            utils.list_os_files_with_extension("/mock", "*.ff")
        self.assertEqual(
            "OCI path is not properly configured. It should follow the pattern `oci://<bucket-name>@<namespace>/object_path`.",
            str(e.exception),
        )

    @patch("ads.common.object_storage_details.authutil.default_signer")
    def test_list_os_files_with_empty_path(self, mock_auth: MagicMock):
        auth = MagicMock()
        auth["signer"] = MagicMock()
        mock_auth.return_value = auth
        with self.assertRaises(Exception) as e:
            utils.list_os_files_with_extension("", "*.ff")
        self.assertEqual(
            "OCI path is not properly configured. It should follow the pattern `oci://<bucket-name>@<namespace>/object_path`.",
            str(e.exception),
        )

    @patch.object(ObjectStorageDetails, "from_path")
    def test_list_os_files_with_extension(self, oss_mock: MagicMock):
        list_obj_resp: ListObjects = ListObjects()
        prefix = "/mock/test/"
        obj1 = ObjectSummary()
        obj1_name = "test1.safetensors"
        obj2_name = "test2.gguf"
        obj1.name = f"{prefix}{obj1_name}"
        obj2 = ObjectSummary()
        obj2.name = f"{prefix}{obj2_name}"
        list_obj_resp.objects = [obj1, obj2]

        oss_mock_client = MagicMock()
        oss_mock.return_value = oss_mock_client
        oss_mock_client.list_objects = MagicMock(return_value=list_obj_resp)
        oss_mock_client.filepath = prefix
        resp = utils.list_os_files_with_extension(prefix, ".gguf")
        self.assertIn(obj2_name, resp)
