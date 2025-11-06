#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import unittest
from unittest.mock import MagicMock, patch
from ads.model.service.oci_datascience_model_group import OCIDataScienceModelGroup

try:
    from oci.data_science.models import (
        ModelGroup,
        HomogeneousModelGroupDetails,
        MemberModelEntries,
        CustomMetadata,
        MemberModelDetails,
        CreateModelGroupDetails,
        UpdateModelGroupDetails,
    )
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest(
        "Support for OCI Model Group is not available. Skipping the Model Group tests."
    )

CREATE_MODEL_GROUP_DETAILS = {
    "create_type": "CREATE",
    "compartment_id": "test_model_group_compartment_id",
    "project_id": "test_model_group_project_id",
    "display_name": "test_create_model_group",
    "description": "test create model group description",
    "model_group_details": HomogeneousModelGroupDetails(
        custom_metadata_list=[
            CustomMetadata(
                key="test_key",
                value="test_value",
                description="test_description",
                category="other",
            )
        ]
    ),
    "member_model_entries": MemberModelEntries(
        member_model_details=[
            MemberModelDetails(inference_key="model_one", model_id="model_id_one"),
            MemberModelDetails(inference_key="model_two", model_id="model_id_two"),
        ]
    ),
    "freeform_tags": {"test_key": "test_value"},
    "model_group_version_history_id": "test_model_group_version_history_id",
    "version_label": "test_version_label",
}

UPDATE_MODEL_GROUP_DETAILS = {
    "display_name": "test_update_model_group",
    "description": "test update model group description",
    "model_group_version_history_id": "test_model_group_version_history_id",
    "version_label": "test_version_label",
    "freeform_tags": {"test_updated_key": "test_updated_value"},
}

OCI_MODEL_GROUP_RESPONSE = {
    "id": "test_model_group_id",
    "compartment_id": "test_model_group_compartment_id",
    "project_id": "test_model_group_project_id",
    "display_name": "test_create_model_group",
    "description": "test create model group description",
    "created_by": "test_create_by",
    "time_created": "2025-06-10T18:21:17.613000Z",
    "time_updated": "2025-06-10T18:21:17.613000Z",
    "lifecycle_state": "ACTIVE",
    "lifecycle_details": "test lifecycle details",
    "model_group_version_history_id": "test_model_group_version_history_id",
    "model_group_version_history_name": "test_model_group_version_history_name",
    "version_label": "test_version_label",
    "version_id": 1,
    "model_group_details": HomogeneousModelGroupDetails(
        custom_metadata_list=[
            CustomMetadata(
                key="test_key",
                value="test_value",
                description="test_description",
                category="other",
            )
        ]
    ),
    "member_model_entries": MemberModelEntries(
        member_model_details=[
            MemberModelDetails(inference_key="model_one", model_id="model_id_one"),
            MemberModelDetails(inference_key="model_two", model_id="model_id_two"),
        ]
    ),
    "freeform_tags": {"test_key": "test_value"},
}


class TestOCIModelGroup:
    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.sync"
    )
    @patch("oci.data_science.DataScienceClient.create_model_group")
    def test_create(self, mock_create_model_group, mock_sync):
        mock_sync.return_value = ModelGroup(**OCI_MODEL_GROUP_RESPONSE)
        create_model_group_details = CreateModelGroupDetails(
            **CREATE_MODEL_GROUP_DETAILS
        )
        oci_model_group = OCIDataScienceModelGroup().create(
            create_model_group_details=create_model_group_details,
            wait_for_completion=False,
            max_wait_time=1,
            poll_interval=2,
        )

        mock_create_model_group.assert_called_with(create_model_group_details)

        assert oci_model_group.id == OCI_MODEL_GROUP_RESPONSE["id"]
        assert oci_model_group.display_name == OCI_MODEL_GROUP_RESPONSE["display_name"]
        assert oci_model_group.description == OCI_MODEL_GROUP_RESPONSE["description"]

    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.sync"
    )
    @patch("oci.data_science.DataScienceClient.activate_model_group")
    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.from_id"
    )
    def test_activate(self, mock_from_id, mock_activate_model_group, mock_sync):
        mock_oci_model_group_activate_response = copy.deepcopy(OCI_MODEL_GROUP_RESPONSE)
        mock_oci_model_group_activate_response["lifecycle_state"] = "INACTIVE"
        mock_from_id.return_value = ModelGroup(**mock_oci_model_group_activate_response)
        mock_sync.return_value = ModelGroup(**OCI_MODEL_GROUP_RESPONSE)
        oci_model_group = OCIDataScienceModelGroup(**OCI_MODEL_GROUP_RESPONSE).activate(
            wait_for_completion=False, max_wait_time=1, poll_interval=2
        )

        mock_activate_model_group.assert_called_with(oci_model_group.id)
        assert oci_model_group.lifecycle_state == "ACTIVE"

    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.sync"
    )
    @patch("oci.data_science.DataScienceClient.deactivate_model_group")
    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.from_id"
    )
    def test_deactivate(self, mock_from_id, mock_deactivate_model_group, mock_sync):
        mock_oci_model_group_deactivate_response = copy.deepcopy(
            OCI_MODEL_GROUP_RESPONSE
        )
        mock_oci_model_group_deactivate_response["lifecycle_state"] = "INACTIVE"
        mock_from_id.return_value = ModelGroup(**OCI_MODEL_GROUP_RESPONSE)
        mock_sync.return_value = ModelGroup(**mock_oci_model_group_deactivate_response)
        oci_model_group = OCIDataScienceModelGroup(
            **mock_oci_model_group_deactivate_response
        ).deactivate(wait_for_completion=False, max_wait_time=1, poll_interval=2)

        mock_deactivate_model_group.assert_called_with(oci_model_group.id)
        assert oci_model_group.lifecycle_state == "INACTIVE"

    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.sync"
    )
    @patch("oci.data_science.DataScienceClient.delete_model_group")
    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.from_id"
    )
    def test_delete(self, mock_from_id, mock_delete_model_group, mock_sync):
        mock_oci_model_group_delete_response = copy.deepcopy(OCI_MODEL_GROUP_RESPONSE)
        mock_oci_model_group_delete_response["lifecycle_state"] = "DELETED"
        mock_from_id.return_value = ModelGroup(**OCI_MODEL_GROUP_RESPONSE)
        mock_sync.return_value = ModelGroup(**mock_oci_model_group_delete_response)

        oci_model_group = OCIDataScienceModelGroup(**OCI_MODEL_GROUP_RESPONSE).delete(
            wait_for_completion=False, max_wait_time=1, poll_interval=2
        )

        mock_delete_model_group.assert_called_with(oci_model_group.id)
        assert oci_model_group.lifecycle_state == "DELETED"

    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.sync"
    )
    @patch(
        "oci.data_science.DataScienceClientCompositeOperations.update_model_group_and_wait_for_state"
    )
    def test_update(self, mock_update_model_group, mock_sync):
        mock_oci_model_group_update_response = copy.deepcopy(OCI_MODEL_GROUP_RESPONSE)
        mock_oci_model_group_update_response.update(**UPDATE_MODEL_GROUP_DETAILS)
        mock_sync.return_value = ModelGroup(**mock_oci_model_group_update_response)
        update_model_group_details = UpdateModelGroupDetails(
            **UPDATE_MODEL_GROUP_DETAILS
        )
        oci_model_group = OCIDataScienceModelGroup(**OCI_MODEL_GROUP_RESPONSE).update(
            update_model_group_details=update_model_group_details,
            wait_for_completion=False,
            max_wait_time=1,
            poll_interval=2,
        )

        mock_update_model_group.assert_called_with(
            oci_model_group.id,
            update_model_group_details,
            wait_for_states=[],
            waiter_kwargs={
                "max_interval_seconds": 2,
                "max_wait_seconds": 1,
            },
        )

        assert oci_model_group.id == mock_oci_model_group_update_response["id"]
        assert (
            oci_model_group.display_name
            == mock_oci_model_group_update_response["display_name"]
        )
        assert (
            oci_model_group.description
            == mock_oci_model_group_update_response["description"]
        )
        assert (
            oci_model_group.freeform_tags
            == mock_oci_model_group_update_response["freeform_tags"]
        )

    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.from_id"
    )
    def test_from_id(self, mock_from_id):
        OCIDataScienceModelGroup.from_id(OCI_MODEL_GROUP_RESPONSE["id"])
        mock_from_id.assert_called_with(OCI_MODEL_GROUP_RESPONSE["id"])

    @patch("oci.pagination.list_call_get_all_results")
    def test_list(self, mock_list_call_get_all_results):
        response = MagicMock()
        response.data = [MagicMock()]
        mock_list_call_get_all_results.return_value = response
        model_groups = OCIDataScienceModelGroup.list(
            status="ACTIVE",
            compartment_id="test_compartment_id",
        )
        mock_list_call_get_all_results.assert_called()
        assert isinstance(model_groups, list)
