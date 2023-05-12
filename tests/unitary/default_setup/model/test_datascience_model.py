#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import hashlib
import json
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pandas
import pytest
from ads.common import utils
from ads.feature_engineering.schema import Schema
from ads.model.artifact_downloader import (
    LargeArtifactDownloader,
    SmallArtifactDownloader,
)
from ads.model.artifact_uploader import LargeArtifactUploader, SmallArtifactUploader
from ads.model.datascience_model import (
    _MAX_ARTIFACT_SIZE_IN_BYTES,
    DataScienceModel,
    ModelArtifactSizeError,
)
from ads.model.model_metadata import (
    ModelCustomMetadata,
    ModelProvenanceMetadata,
    ModelTaxonomyMetadata,
)
from ads.model.service.oci_datascience_model import (
    ModelProvenanceNotFoundError,
    OCIDataScienceModel,
)
from oci.data_science.models import ModelProvenance

MODEL_OCID = "ocid1.datasciencemodel.oc1.iad.<unique_ocid>"

OCI_MODEL_PAYLOAD = {
    "id": MODEL_OCID,
    "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
    "project_id": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
    "display_name": "Generic Model With Small Artifact new",
    "description": "The model description",
    "lifecycle_state": "ACTIVE",
    "created_by": "ocid1.user.oc1..<unique_ocid>",
    "freeform_tags": {"key1": "value1"},
    "defined_tags": {"key1": {"skey1": "value1"}},
    "time_created": "2022-08-24T17:07:39.200000Z",
    "custom_metadata_list": [
        {
            "key": "CondaEnvironment",
            "value": "oci://bucket@namespace/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/3.0/dataexpl_p37_cpu_v3",
            "description": "The conda environment where the model was trained.",
            "category": "Training Environment",
        },
    ],
    "defined_metadata_list": [
        {"key": "Algorithm", "value": "test"},
        {"key": "Framework"},
        {"key": "FrameworkVersion"},
        {"key": "UseCaseType", "value": "multinomial_classification"},
        {"key": "Hyperparameters"},
        {"key": "ArtifactTestResults"},
    ],
    "input_schema": '{"schema": [{"dtype": "int64", "feature_type": "Integer", "name": 0, "domain": {"values": "", "stats": {}, "constraints": []}, "required": true, "description": "0", "order": 0}], "version": "1.1"}',
    "output_schema": '{"schema": [{"dtype": "int64", "feature_type": "Integer", "name": 0, "domain": {"values": "", "stats": {}, "constraints": []}, "required": true, "description": "0", "order": 0}], "version": "1.1"}',
}

OCI_MODEL_PROVENANCE_PAYLOAD = {
    "git_branch": "master",
    "git_commit": "7c8c8502896ba36837f15037b67e05a3cf9722c7",
    "repository_url": "file:///home/datascience",
    "script_dir": "test_script_dir",
    "training_id": None,
    "training_script": None,
}

DSC_MODEL_PAYLOAD = {
    "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
    "projectId": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
    "displayName": "Generic Model With Small Artifact new",
    "description": "The model description",
    "freeformTags": {"key1": "value1"},
    "definedTags": {"key1": {"skey1": "value1"}},
    "inputSchema": {
        "schema": [
            {
                "feature_type": "Integer",
                "dtype": "int64",
                "name": 0,
                "domain": {"values": "", "stats": {}, "constraints": []},
                "required": True,
                "description": "0",
                "order": 0,
            }
        ],
        "version": "1.1",
    },
    "outputSchema": {
        "schema": [
            {
                "dtype": "int64",
                "feature_type": "Integer",
                "name": 0,
                "domain": {"values": "", "stats": {}, "constraints": []},
                "required": True,
                "description": "0",
                "order": 0,
            }
        ],
        "version": "1.1",
    },
    "customMetadataList": {
        "data": [
            {
                "key": "CondaEnvironment",
                "value": "oci://bucket@namespace/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/3.0/dataexpl_p37_cpu_v3",
                "description": "The conda environment where the model was trained.",
                "category": "Training Environment",
            },
        ]
    },
    "definedMetadataList": {
        "data": [
            {"key": "Algorithm", "value": "test"},
            {"key": "Framework", "value": None},
            {"key": "FrameworkVersion", "value": None},
            {"key": "UseCaseType", "value": "multinomial_classification"},
            {"key": "Hyperparameters", "value": None},
            {"key": "ArtifactTestResults", "value": None},
        ]
    },
    "provenanceMetadata": {
        "git_branch": "master",
        "git_commit": "7c8c8502896ba36837f15037b67e05a3cf9722c7",
        "repository_url": "file:///home/datascience",
        "training_script_path": None,
        "training_id": None,
        "artifact_dir": "test_script_dir",
    },
    "artifact": "ocid1.datasciencemodel.oc1.iad.<unique_ocid>.zip",
}

ARTIFACT_HEADER_INFO = {
    "Date": "Sun, 13 Nov 2022 06:01:27 GMT",
    "opc-request-id": "E4F7",
    "ETag": "77156317-8bb9-4c4a-882b-0d85f8140d93",
    "Content-Disposition": "attachment; filename=new_ocid1.datasciencemodel.oc1.iad.<unique_ocid>.zip",
    "Last-Modified": "Sun, 09 Oct 2022 16:50:14 GMT",
    "Content-Type": "application/json",
    "Content-MD5": "orMy3Gs386GZLjYWATJWuA==",
    "X-Content-Type-Options": "nosniff",
    "Content-Length": _MAX_ARTIFACT_SIZE_IN_BYTES + 100,
}


class TestDataScienceModel:

    DEFAULT_PROPERTIES_PAYLOAD = {
        "compartmentId": DSC_MODEL_PAYLOAD["compartmentId"],
        "projectId": DSC_MODEL_PAYLOAD["projectId"],
        "displayName": DSC_MODEL_PAYLOAD["displayName"],
    }

    def setup_class(cls):
        cls.mock_date = datetime.datetime(2022, 7, 1)

    def setup_method(self):
        self.payload = deepcopy(DSC_MODEL_PAYLOAD)
        self.mock_dsc_model = DataScienceModel(**self.payload)

    def prepare_dict(self, data):
        if "definedMetadataList" in data:
            if isinstance(data["definedMetadataList"], dict):
                data["definedMetadataList"]["data"] = sorted(
                    data["definedMetadataList"]["data"], key=lambda x: x["key"]
                )
            else:
                data["definedMetadataList"] = sorted(
                    data["definedMetadataList"], key=lambda x: x["key"]
                )

        if "customMetadataList" in data:
            if isinstance(data["customMetadataList"], dict):
                data["customMetadataList"]["data"] = sorted(
                    data["customMetadataList"]["data"], key=lambda x: x["key"]
                )
            else:
                data["customMetadataList"] = sorted(
                    data["customMetadataList"], key=lambda x: x["key"]
                )
        return data

    def hash_dict(self, data):
        return hashlib.sha1(
            json.dumps(self.prepare_dict(data), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def compare_dict(self, dict1, dict2):
        print(
            f"dict1_hash: {self.hash_dict(dict1)}; dict2_hash: {self.hash_dict(dict2)}"
        )
        return self.hash_dict(dict1) == self.hash_dict(dict2)

    @patch.object(
        DataScienceModel,
        "_load_default_properties",
        return_value=DEFAULT_PROPERTIES_PAYLOAD,
    )
    def test__init__default_properties(self, mock_load_default_properties):
        dsc_model = DataScienceModel()
        assert dsc_model.to_dict()["spec"] == self.DEFAULT_PROPERTIES_PAYLOAD

    @patch.object(DataScienceModel, "_load_default_properties", return_value={})
    def test__init__(self, mock_load_default_properties):
        dsc_model = DataScienceModel(**self.payload)
        assert self.prepare_dict(dsc_model.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(DataScienceModel, "_load_default_properties", return_value={})
    def test_with_methods_1(self, mock_load_default_properties):
        """Tests all with methods."""
        dsc_model = (
            DataScienceModel()
            .with_compartment_id(self.payload["compartmentId"])
            .with_project_id(self.payload["projectId"])
            .with_display_name(self.payload["displayName"])
            .with_description(self.payload["description"])
            .with_freeform_tags(**(self.payload["freeformTags"] or {}))
            .with_defined_tags(**(self.payload["definedTags"] or {}))
            .with_input_schema(self.payload["inputSchema"])
            .with_output_schema(self.payload["outputSchema"])
            .with_custom_metadata_list(self.payload["customMetadataList"])
            .with_defined_metadata_list(self.payload["definedMetadataList"])
            .with_provenance_metadata(self.payload["provenanceMetadata"])
            .with_artifact(self.payload["artifact"])
        )
        assert self.prepare_dict(dsc_model.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    def test_with_methods_2(self):
        """Tests all with methods."""
        dsc_model = (
            DataScienceModel()
            .with_compartment_id(self.payload["compartmentId"])
            .with_project_id(self.payload["projectId"])
            .with_display_name(self.payload["displayName"])
            .with_description(self.payload["description"])
            .with_freeform_tags(**(self.payload["freeformTags"] or {}))
            .with_defined_tags(**(self.payload["definedTags"] or {}))
            .with_input_schema(Schema.from_dict(self.payload["inputSchema"]))
            .with_output_schema(Schema.from_dict(self.payload["outputSchema"]))
            .with_custom_metadata_list(
                ModelCustomMetadata.from_dict(self.payload["customMetadataList"])
            )
            .with_defined_metadata_list(
                ModelTaxonomyMetadata.from_dict(self.payload["definedMetadataList"])
            )
            .with_provenance_metadata(
                ModelProvenanceMetadata.from_dict(self.payload["provenanceMetadata"])
            )
            .with_artifact(self.payload["artifact"])
        )
        assert self.prepare_dict(dsc_model.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(OCIDataScienceModel, "delete")
    @patch.object(DataScienceModel, "sync")
    def test_delete(self, mock_sync, mock_delete):
        """Tests deleting model from model catalog."""
        self.mock_dsc_model.delete(delete_associated_model_deployment=True)
        mock_sync.assert_called()
        mock_delete.assert_called_with(True)

    @patch.object(DataScienceModel, "_update_from_oci_dsc_model")
    @patch.object(OCIDataScienceModel, "list_resource")
    def test_list(self, mock_list_resource, mock__update_from_oci_dsc_model):
        """Tests listing datascience models in a given compartment."""
        mock_list_resource.return_value = [OCIDataScienceModel(**OCI_MODEL_PAYLOAD)]
        mock__update_from_oci_dsc_model.return_value = DataScienceModel(**self.payload)
        result = DataScienceModel.list(
            compartment_id="test_compartment_id",
            project_id="test_project_id",
            extra_tag="test_cvalue",
        )
        mock_list_resource.assert_called_with(
            "test_compartment_id",
            project_id="test_project_id",
            **{"extra_tag": "test_cvalue"},
        )
        assert len(result) == 1
        assert self.prepare_dict(result[0].to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(OCIDataScienceModel, "list_resource")
    def test_list_df(self, mock_list_resource):
        """Tests listing datascience models in a given compartment."""
        model = OCIDataScienceModel(**OCI_MODEL_PAYLOAD)
        mock_list_resource.return_value = [model]
        records = []
        records.append(
            {
                "id": f"...{model.id[-6:]}",
                "display_name": model.display_name,
                "description": model.description,
                "time_created": model.time_created.strftime(utils.date_format),
                "lifecycle_state": model.lifecycle_state,
                "created_by": f"...{model.created_by[-6:]}",
                "compartment_id": f"...{model.compartment_id[-6:]}",
                "project_id": f"...{model.project_id[-6:]}",
            }
        )
        expected_result = pandas.DataFrame.from_records(records)
        result = DataScienceModel.list_df(
            compartment_id="test_compartment_id",
            project_id="test_project_id",
            extra_tag="test_cvalue",
        )
        mock_list_resource.assert_called_with(
            "test_compartment_id",
            project_id="test_project_id",
            **{"extra_tag": "test_cvalue"},
        )
        assert expected_result.equals(result)

    @patch.object(DataScienceModel, "_update_from_oci_dsc_model")
    @patch.object(OCIDataScienceModel, "from_id")
    def test_from_id(self, mock_oci_from_id, mock__update_from_oci_dsc_model):
        """Tests getting an existing model by OCID."""
        mock_oci_model = OCIDataScienceModel(**OCI_MODEL_PAYLOAD)
        mock_oci_from_id.return_value = mock_oci_model
        mock__update_from_oci_dsc_model.return_value = DataScienceModel(**self.payload)
        result = DataScienceModel.from_id(MODEL_OCID)

        mock_oci_from_id.assert_called_with(MODEL_OCID)
        mock__update_from_oci_dsc_model.assert_called_with(mock_oci_model)
        assert self.prepare_dict(result.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(OCIDataScienceModel, "create_model_provenance")
    @patch.object(OCIDataScienceModel, "create")
    @patch.object(DataScienceModel, "sync")
    @patch.object(DataScienceModel, "upload_artifact")
    @patch.object(DataScienceModel, "_random_display_name", return_value="random_name")
    @patch.object(DataScienceModel, "_load_default_properties", return_value={})
    def test_create_success(
        self,
        mock__load_default_properties,
        mock__random_display_name,
        mock_upload_artifact,
        mock_sync,
        mock_oci_dsc_model_create,
        mock_create_model_provenance,
    ):
        """Tests creating datascience model."""
        oci_dsc_model = OCIDataScienceModel(**OCI_MODEL_PAYLOAD)
        mock_oci_dsc_model_create.return_value = oci_dsc_model

        # to check random display name
        self.mock_dsc_model.with_display_name("")
        result = self.mock_dsc_model.create(
            bucket_uri="test_bucket_uri",
            overwrite_existing_artifact=False,
            remove_existing_artifact=False,
        )
        mock_oci_dsc_model_create.assert_called()
        mock_create_model_provenance.assert_called_with(
            self.mock_dsc_model.provenance_metadata._to_oci_metadata()
        )
        mock_upload_artifact.assert_called_with(
            bucket_uri="test_bucket_uri",
            overwrite_existing_artifact=False,
            remove_existing_artifact=False,
            region=None,
            auth=None,
            timeout=None,
        )
        mock_sync.assert_called()
        assert self.prepare_dict(result.to_dict()["spec"]) == self.prepare_dict(
            {**self.payload, "displayName": "random_name"}
        )

    @patch.object(DataScienceModel, "_load_default_properties", return_value={})
    def test_create_fail(self, mock__load_default_properties):
        """Tests creating datascience model."""
        dsc_model = DataScienceModel()
        with pytest.raises(ValueError, match="Compartment id must be provided."):
            dsc_model.create()

        dsc_model.with_compartment_id("compartment_id")
        with pytest.raises(ValueError, match="Project id must be provided."):
            dsc_model.create()

    @patch.object(OCIDataScienceModel, "create_model_provenance")
    @patch.object(OCIDataScienceModel, "update_model_provenance")
    @patch.object(OCIDataScienceModel, "update")
    @patch.object(DataScienceModel, "sync")
    def test_update_success(
        self,
        mock_sync,
        mock_update,
        mock_update_model_provenance,
        mock_create_model_provenance,
    ):
        """Test updating datascience model in model catalog."""
        oci_dsc_model = OCIDataScienceModel(**OCI_MODEL_PAYLOAD)
        self.mock_dsc_model.dsc_model = oci_dsc_model

        mock_update.return_value = oci_dsc_model

        # With update Model Provenance
        with patch.object(
            OCIDataScienceModel, "get_model_provenance"
        ) as mock_get_model_provenance:
            mock_model_provenance = ModelProvenance(**OCI_MODEL_PROVENANCE_PAYLOAD)
            mock_get_model_provenance.return_value = mock_model_provenance

            self.mock_dsc_model.update(display_name="new_display_name")
            mock_update.assert_called()
            mock_update_model_provenance.assert_called_with(mock_model_provenance)
            mock_sync.assert_called()

        # With create Model Provenance
        with patch.object(
            OCIDataScienceModel,
            "get_model_provenance",
            side_effect=ModelProvenanceNotFoundError(),
        ):
            self.mock_dsc_model.update(display_name="new_display_name")
            mock_model_provenance = ModelProvenance(**OCI_MODEL_PROVENANCE_PAYLOAD)
            mock_update.assert_called()
            mock_create_model_provenance.assert_called_with(mock_model_provenance)
            mock_sync.assert_called()

    def test_to_dict(self):
        """Tests serializing model to a dictionary."""
        test_dict = self.mock_dsc_model.to_dict()
        assert self.prepare_dict(test_dict["spec"]) == self.prepare_dict(self.payload)
        assert test_dict["kind"] == self.mock_dsc_model.kind
        assert test_dict["type"] == self.mock_dsc_model.type

    def test_from_dict(self):
        """Tests loading model instance from a dictionary of configurations."""
        assert self.prepare_dict(
            self.mock_dsc_model.to_dict()["spec"]
        ) == self.prepare_dict(
            DataScienceModel.from_dict({"spec": self.payload}).to_dict()["spec"]
        )

    @patch("ads.common.utils.get_random_name_for_resource", return_value="test_name")
    def test__random_display_name(self, mock_get_random_name_for_resource):
        """Tests generating a random display name."""
        expected_result = f"{self.mock_dsc_model._PREFIX}-test_name"
        assert self.mock_dsc_model._random_display_name() == expected_result

    def test__to_oci_dsc_model(self):
        """Tests creating an `OCIDataScienceModel` instance from the  `DataScienceModel`."""
        with patch.object(OCIDataScienceModel, "sync"):
            test_oci_dsc_model = OCIDataScienceModel(**OCI_MODEL_PAYLOAD)
            test_oci_dsc_model.id = None
            test_oci_dsc_model.lifecycle_state = None
            test_oci_dsc_model.created_by = None
            test_oci_dsc_model.time_created = None

            assert self.prepare_dict(test_oci_dsc_model.to_dict()) == self.prepare_dict(
                self.mock_dsc_model._to_oci_dsc_model().to_dict()
            )

            test_oci_dsc_model.display_name = "new_name"
            assert self.prepare_dict(test_oci_dsc_model.to_dict()) == self.prepare_dict(
                self.mock_dsc_model._to_oci_dsc_model(display_name="new_name").to_dict()
            )

    @patch.object(OCIDataScienceModel, "get_artifact_info")
    @patch.object(OCIDataScienceModel, "get_model_provenance")
    def test__update_from_oci_dsc_model(
        self, mock_get_model_provenance, mock_get_artifact_info
    ):
        """Tests updating the properties from an OCIDataScienceModel object."""
        oci_model_payload = {
            "compartment_id": "new ocid1.compartment.oc1..<unique_ocid>",
            "project_id": "new ocid1.datascienceproject.oc1.iad.<unique_ocid>",
            "display_name": "new Generic Model With Small Artifact new",
            "description": "new The model description",
            "freeform_tags": {"newkey1": "new value1"},
            "defined_tags": {"newkey1": {"newskey1": "value1"}},
            "custom_metadata_list": [
                {
                    "key": "CondaEnvironment",
                    "value": "new oci://bucket@namespace/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/3.0/dataexpl_p37_cpu_v3",
                    "description": "new The conda environment where the model was trained.",
                    "category": "Training Environment",
                },
            ],
            "defined_metadata_list": [
                {"key": "Algorithm", "value": "new test"},
                {"key": "Framework", "value": "new test"},
                {"key": "FrameworkVersion", "value": "new test"},
                {"key": "UseCaseType", "value": "multinomial_classification"},
                {"key": "Hyperparameters", "value": "new test"},
                {"key": "ArtifactTestResults", "value": "new test"},
            ],
            "input_schema": '{"schema": [{"dtype": "int64", "feature_type": "Integer", "name": 1, "domain": {"values": "", "stats": {}, "constraints": []}, "required": true, "description": "0", "order": 0}], "version": "1.1"}',
            "output_schema": '{"schema": [{"dtype": "int64", "feature_type": "Integer", "name": 1, "domain": {"values": "", "stats": {}, "constraints": []}, "required": true, "description": "0", "order": 0}], "version": "1.1"}',
        }

        dsc_model_payload = {
            "compartmentId": "new ocid1.compartment.oc1..<unique_ocid>",
            "projectId": "new ocid1.datascienceproject.oc1.iad.<unique_ocid>",
            "displayName": "new Generic Model With Small Artifact new",
            "description": "new The model description",
            "freeformTags": {"newkey1": "new value1"},
            "definedTags": {"newkey1": {"newskey1": "value1"}},
            "inputSchema": {
                "schema": [
                    {
                        "feature_type": "Integer",
                        "dtype": "int64",
                        "name": 1,
                        "domain": {"values": "", "stats": {}, "constraints": []},
                        "required": True,
                        "description": "0",
                        "order": 0,
                    }
                ],
                "version": "1.1",
            },
            "outputSchema": {
                "schema": [
                    {
                        "dtype": "int64",
                        "feature_type": "Integer",
                        "name": 1,
                        "domain": {"values": "", "stats": {}, "constraints": []},
                        "required": True,
                        "description": "0",
                        "order": 0,
                    }
                ],
                "version": "1.1",
            },
            "customMetadataList": {
                "data": [
                    {
                        "key": "CondaEnvironment",
                        "value": "new oci://bucket@namespace/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/3.0/dataexpl_p37_cpu_v3",
                        "description": "new The conda environment where the model was trained.",
                        "category": "Training Environment",
                    },
                ]
            },
            "definedMetadataList": {
                "data": [
                    {"key": "Algorithm", "value": "new test"},
                    {"key": "Framework", "value": "new test"},
                    {"key": "FrameworkVersion", "value": "new test"},
                    {"key": "UseCaseType", "value": "multinomial_classification"},
                    {"key": "Hyperparameters", "value": "new test"},
                    {"key": "ArtifactTestResults", "value": "new test"},
                ]
            },
            "provenanceMetadata": {
                "git_branch": "master",
                "git_commit": "7c8c8502896ba36837f15037b67e05a3cf9722c7",
                "repository_url": "file:///home/datascience",
                "training_script_path": None,
                "training_id": None,
                "artifact_dir": "test_script_dir",
            },
            "artifact": "new_ocid1.datasciencemodel.oc1.iad.<unique_ocid>.zip",
        }

        with patch.object(OCIDataScienceModel, "sync"):
            mock_oci_dsc_model = OCIDataScienceModel(**oci_model_payload)
            mock_model_provenance = ModelProvenance(**OCI_MODEL_PROVENANCE_PAYLOAD)
            mock_get_model_provenance.return_value = mock_model_provenance
            mock_get_artifact_info.return_value = ARTIFACT_HEADER_INFO
            self.mock_dsc_model._update_from_oci_dsc_model(mock_oci_dsc_model)
            assert self.prepare_dict(
                self.mock_dsc_model.to_dict()["spec"]
            ) == self.prepare_dict(dsc_model_payload)

    def test_upload_artifact(self):
        """Tests uploading artifacts to the model catalog."""
        self.mock_dsc_model.dsc_model = MagicMock(__class__=MagicMock())

        # Artifact size greater than 2GB
        with patch.object(
            LargeArtifactUploader, "__init__", return_value=None
        ) as mock_init:
            with patch.object(LargeArtifactUploader, "upload") as mock_upload:
                with patch(
                    "ads.common.utils.folder_size",
                    return_value=_MAX_ARTIFACT_SIZE_IN_BYTES + 100,
                ):
                    # If artifact is large and bucket_uri not provided
                    with pytest.raises(ModelArtifactSizeError):
                        self.mock_dsc_model.upload_artifact()

                    self.mock_dsc_model.upload_artifact(
                        bucket_uri="test_bucket_uri",
                        auth={"config": {}},
                        region="test_region",
                        overwrite_existing_artifact=False,
                        remove_existing_artifact=False,
                        timeout=1,
                    )
                    assert self.mock_dsc_model.dsc_model.__class__.kwargs == {
                        "timeout": 1
                    }
                    mock_init.assert_called_with(
                        dsc_model=self.mock_dsc_model.dsc_model,
                        artifact_path=self.mock_dsc_model.artifact,
                        auth={"config": {}},
                        region="test_region",
                        bucket_uri="test_bucket_uri",
                        overwrite_existing_artifact=False,
                        remove_existing_artifact=False,
                    )
                    mock_upload.assert_called()

        # Artifact size less than 2GB
        with patch.object(
            SmallArtifactUploader, "__init__", return_value=None
        ) as mock_init:
            with patch.object(SmallArtifactUploader, "upload") as mock_upload:
                with patch(
                    "ads.common.utils.folder_size",
                    return_value=_MAX_ARTIFACT_SIZE_IN_BYTES - 100,
                ):
                    self.mock_dsc_model.upload_artifact(timeout=2)
                    assert self.mock_dsc_model.dsc_model.__class__.kwargs == {
                        "timeout": 2
                    }
                    mock_init.assert_called_with(
                        dsc_model=self.mock_dsc_model.dsc_model,
                        artifact_path=self.mock_dsc_model.artifact,
                    )
                    mock_upload.assert_called()

    def test_download_artifact(self):
        """Tests downloading artifacts from the model catalog."""
        # Artifact size greater than 2GB
        mock_get_artifact_info = MagicMock(
            return_value={"content-length": _MAX_ARTIFACT_SIZE_IN_BYTES + 100}
        )
        self.mock_dsc_model.dsc_model = MagicMock(
            id="test_id",
            get_artifact_info=mock_get_artifact_info,
            __class__=MagicMock(),
        )
        with patch.object(
            LargeArtifactDownloader, "__init__", return_value=None
        ) as mock_init:
            with patch.object(LargeArtifactDownloader, "download") as mock_download:

                # If artifact is large and bucket_uri not provided
                with pytest.raises(ModelArtifactSizeError):
                    self.mock_dsc_model.download_artifact(target_dir="test_target_dir")

                self.mock_dsc_model.download_artifact(
                    target_dir="test_target_dir",
                    auth={"config": {}},
                    force_overwrite=True,
                    bucket_uri="test_bucket_uri",
                    region="test_region",
                    overwrite_existing_artifact=False,
                    remove_existing_artifact=False,
                    timeout=1,
                )
                assert self.mock_dsc_model.dsc_model.__class__.kwargs == {"timeout": 1}

                mock_init.assert_called_with(
                    dsc_model=self.mock_dsc_model.dsc_model,
                    target_dir="test_target_dir",
                    auth={"config": {}},
                    force_overwrite=True,
                    region="test_region",
                    bucket_uri="test_bucket_uri",
                    overwrite_existing_artifact=False,
                    remove_existing_artifact=False,
                )
                mock_download.assert_called()

        # Artifact size less than 2GB
        mock_get_artifact_info = MagicMock(
            return_value={"content-length": _MAX_ARTIFACT_SIZE_IN_BYTES - 100}
        )
        self.mock_dsc_model.dsc_model = MagicMock(
            id="test_id",
            get_artifact_info=mock_get_artifact_info,
            __class__=MagicMock(),
        )
        with patch.object(
            SmallArtifactDownloader, "__init__", return_value=None
        ) as mock_init:
            with patch.object(SmallArtifactDownloader, "download") as mock_download:
                self.mock_dsc_model.download_artifact(
                    target_dir="test_target_dir", force_overwrite=True, timeout=2
                )
                mock_init.assert_called_with(
                    dsc_model=self.mock_dsc_model.dsc_model,
                    target_dir="test_target_dir",
                    force_overwrite=True,
                )
                assert self.mock_dsc_model.dsc_model.__class__.kwargs == {"timeout": 2}
                mock_download.assert_called()
