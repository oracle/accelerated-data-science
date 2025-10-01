#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import unittest
from unittest.mock import patch
from ads.model.datascience_model_group import DataScienceModelGroup
from ads.model.model_metadata import ModelCustomMetadata

try:
    from oci.data_science.models import (
        ModelGroup,
        HomogeneousModelGroupDetails,
        MemberModelEntries,
        CustomMetadata,
        MemberModelDetails,
        ModelGroupSummary,
    )
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest(
        "Support for OCI Model Group is not available. Skipping the Model Group tests."
    )

MODEL_GROUP_DICT = {
    "kind": "datascienceModelGroup",
    "type": "dataScienceModelGroup",
    "spec": {
        "displayName": "test_create_model_group",
        "description": "test create model group description",
        "freeformTags": {"test_key": "test_value"},
        "customMetadataList": {
            "data": [
                {
                    "key": "test_key",
                    "value": "test_value",
                    "description": "test_description",
                    "category": "other",
                    "has_artifact": False,
                }
            ]
        },
        "memberModels": [
            {"inference_key": "model_one", "model_id": "model_id_one"},
            {"inference_key": "model_two", "model_id": "model_id_two"},
        ],
    },
}

MODEL_GROUP_SPEC = {
    "display_name": "test_create_model_group",
    "description": "test create model group description",
    "freeform_tags": {"test_key": "test_value"},
    "custom_metadata_list": {
        "data": [
            {
                "key": "test_key",
                "value": "test_value",
                "description": "test_description",
                "category": "other",
                "has_artifact": False,
            }
        ]
    },
    "member_models": [
        {"inference_key": "model_one", "model_id": "model_id_one"},
        {"inference_key": "model_two", "model_id": "model_id_two"},
    ],
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


class TestModelGroup:
    def initialize_model_group(self):
        custom_metadata = ModelCustomMetadata()
        custom_metadata.add(
            key="test_key",
            value="test_value",
            description="test_description",
            category="other",
        )

        model_group = (
            DataScienceModelGroup()
            .with_display_name("test_create_model_group")
            .with_description("test create model group description")
            .with_freeform_tags(**{"test_key": "test_value"})
            .with_custom_metadata_list(custom_metadata)
            .with_member_models(
                [
                    {"inference_key": "model_one", "model_id": "model_id_one"},
                    {"inference_key": "model_two", "model_id": "model_id_two"},
                ]
            )
        )

        return model_group

    def test_initialize_model_group(self):
        model_group_one = self.initialize_model_group()
        assert model_group_one.to_dict() == MODEL_GROUP_DICT

        model_group_two = DataScienceModelGroup.from_dict(MODEL_GROUP_DICT)
        assert model_group_two.to_dict() == MODEL_GROUP_DICT

        model_group_three = DataScienceModelGroup(spec=MODEL_GROUP_SPEC)
        assert model_group_three.to_dict() == MODEL_GROUP_DICT

        model_group_four = DataScienceModelGroup(**MODEL_GROUP_SPEC)
        assert model_group_four.to_dict() == MODEL_GROUP_DICT

    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.create"
    )
    def test_create(self, mock_dsc_model_group_create):
        mock_dsc_model_group_create.return_value = ModelGroup(
            **OCI_MODEL_GROUP_RESPONSE
        )
        model_group = self.initialize_model_group()
        model_group.create()

        mock_dsc_model_group_create.assert_called()

        assert model_group.id == OCI_MODEL_GROUP_RESPONSE["id"]
        assert model_group.display_name == OCI_MODEL_GROUP_RESPONSE["display_name"]
        assert model_group.description == OCI_MODEL_GROUP_RESPONSE["description"]

    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.activate"
    )
    def test_activate(self, mock_dsc_model_group_activate):
        mock_dsc_model_group_activate.return_value = ModelGroup(
            **OCI_MODEL_GROUP_RESPONSE
        )
        model_group = self.initialize_model_group()
        model_group.activate(
            wait_for_completion=False,
            max_wait_time=1,
            poll_interval=2,
        )

        mock_dsc_model_group_activate.assert_called_with(
            wait_for_completion=False,
            max_wait_time=1,
            poll_interval=2,
        )

        assert model_group.lifecycle_state == "ACTIVE"

    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.deactivate"
    )
    def test_deactivate(self, mock_dsc_model_group_deactivate):
        mock_dsc_model_group_deactivate_response = copy.deepcopy(
            OCI_MODEL_GROUP_RESPONSE
        )
        mock_dsc_model_group_deactivate_response["lifecycle_state"] = "INACTIVE"

        mock_dsc_model_group_deactivate.return_value = ModelGroup(
            **mock_dsc_model_group_deactivate_response
        )
        model_group = self.initialize_model_group()
        model_group.deactivate(
            wait_for_completion=False,
            max_wait_time=1,
            poll_interval=2,
        )

        mock_dsc_model_group_deactivate.assert_called_with(
            wait_for_completion=False,
            max_wait_time=1,
            poll_interval=2,
        )

        assert model_group.lifecycle_state == "INACTIVE"

    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.delete"
    )
    def test_delete(self, mock_dsc_model_group_delete):
        mock_dsc_model_group_delete_response = copy.deepcopy(OCI_MODEL_GROUP_RESPONSE)
        mock_dsc_model_group_delete_response["lifecycle_state"] = "DELETED"

        mock_dsc_model_group_delete.return_value = ModelGroup(
            **mock_dsc_model_group_delete_response
        )
        model_group = self.initialize_model_group()
        model_group.delete(
            wait_for_completion=False,
            max_wait_time=1,
            poll_interval=2,
        )

        mock_dsc_model_group_delete.assert_called_with(
            wait_for_completion=False,
            max_wait_time=1,
            poll_interval=2,
        )

        assert model_group.lifecycle_state == "DELETED"

    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.update"
    )
    def test_update(self, mock_dsc_model_group_update):
        mock_dsc_model_group_update_response = copy.deepcopy(OCI_MODEL_GROUP_RESPONSE)
        mock_dsc_model_group_update_response["display_name"] = "updated display name"
        mock_dsc_model_group_update_response["description"] = "updated description"

        mock_dsc_model_group_update.return_value = ModelGroup(
            **mock_dsc_model_group_update_response
        )
        model_group = self.initialize_model_group()
        model_group.update()

        mock_dsc_model_group_update.assert_called()

        assert (
            model_group.display_name
            == mock_dsc_model_group_update_response["display_name"]
        )
        assert (
            model_group.description
            == mock_dsc_model_group_update_response["description"]
        )

    @patch(
        "ads.model.service.oci_datascience_model_group.OCIDataScienceModelGroup.list"
    )
    def test_list(self, mock_dsc_model_group_list):
        mock_dsc_model_group_list_response = copy.deepcopy(OCI_MODEL_GROUP_RESPONSE)
        mock_dsc_model_group_list_response.pop("member_model_entries")
        mock_dsc_model_group_list_response.pop("description")
        mock_dsc_model_group_list.return_value = [
            ModelGroupSummary(**mock_dsc_model_group_list_response)
        ]

        model_groups = DataScienceModelGroup.list(
            status="ACTIVE", compartment_id="test_model_group_compartment_id"
        )

        mock_dsc_model_group_list.assert_called_with(
            status="ACTIVE", compartment_id="test_model_group_compartment_id"
        )

        assert len(model_groups) == 1
