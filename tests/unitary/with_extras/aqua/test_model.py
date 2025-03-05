#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import shlex
import tempfile
from dataclasses import asdict
from importlib import reload
from unittest.mock import MagicMock, patch

import oci
import pytest
from huggingface_hub.hf_api import HfApi, ModelInfo
from parameterized import parameterized

import ads.aqua.model
import ads.common
import ads.common.oci_client
import ads.config
from ads.aqua.common.enums import ModelFormat
from ads.aqua.common.errors import (
    AquaFileNotFoundError,
    AquaRuntimeError,
    AquaValueError,
)
from ads.aqua.common.utils import get_hf_model_info
from ads.aqua.constants import HF_METADATA_FOLDER
from ads.aqua.model import AquaModelApp
from ads.aqua.model.entities import (
    AquaModel,
    AquaModelSummary,
    ImportModelDetails,
    ModelValidationResult,
)
from ads.common.object_storage_details import ObjectStorageDetails
from ads.model.datascience_model import DataScienceModel
from ads.model.model_metadata import (
    ModelCustomMetadata,
    ModelProvenanceMetadata,
    ModelTaxonomyMetadata,
)
from ads.model.service.oci_datascience_model import OCIDataScienceModel


@pytest.fixture(autouse=True, scope="class")
def mock_auth():
    with patch("ads.common.auth.default_signer") as mock_default_signer:
        yield mock_default_signer


def get_container_config():
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_data/ui/container_index.json",
        ),
        "r",
    ) as _file:
        container_index_json = json.load(_file)

    return container_index_json


@pytest.fixture(autouse=True, scope="class")
def mock_get_container_config():
    with patch("ads.aqua.model.model.get_container_config") as mock_config:
        mock_config.return_value = get_container_config()
        yield mock_config


@pytest.fixture(autouse=True, scope="function")
def mock_get_hf_model_info():
    with patch.object(HfApi, "model_info") as mock_get_hf_model_info:
        test_hf_model_info = ModelInfo(
            disabled=False,
            id="oracle/aqua-1t-mega-model",
            pipeline_tag="text-generation",
            author="test_author",
            private=False,
            downloads=10,
            likes=10,
            tags=None,
            siblings=[
                {
                    "rfilename": ".gitattributes",
                    "size": None,
                    "blob_id": None,
                    "lfs": None,
                },
                {"rfilename": "README.md", "size": None, "blob_id": None, "lfs": None},
                {
                    "rfilename": "config.json",
                    "size": None,
                    "blob_id": None,
                    "lfs": None,
                },
                {
                    "rfilename": "model_file.gguf",
                    "size": None,
                    "blob_id": None,
                    "lfs": None,
                },
                {
                    "rfilename": "generation_config.json",
                    "size": None,
                    "blob_id": None,
                    "lfs": None,
                },
                {
                    "rfilename": "model-00001-of-00001.safetensors",
                    "size": None,
                    "blob_id": None,
                    "lfs": None,
                },
                {
                    "rfilename": "model-00002-of-00002.safetensors",
                    "size": None,
                    "blob_id": None,
                    "lfs": None,
                },
                {
                    "rfilename": "model.safetensors.index.json",
                    "size": None,
                    "blob_id": None,
                    "lfs": None,
                },
                {
                    "rfilename": "special_tokens_map.json",
                    "size": None,
                    "blob_id": None,
                    "lfs": None,
                },
                {
                    "rfilename": "tokenizer.json",
                    "size": None,
                    "blob_id": None,
                    "lfs": None,
                },
                {
                    "rfilename": "tokenizer.model",
                    "size": None,
                    "blob_id": None,
                    "lfs": None,
                },
                {
                    "rfilename": "tokenizer_config.json",
                    "size": None,
                    "blob_id": None,
                    "lfs": None,
                },
            ],
        )
        mock_get_hf_model_info.return_value = test_hf_model_info
        yield mock_get_hf_model_info


@pytest.fixture(autouse=True, scope="class")
def mock_init_client():
    with patch(
        "ads.common.oci_datascience.OCIDataScienceMixin.init_client"
    ) as mock_client:
        yield mock_client


class TestDataset:
    model_summary_objects = [
        {
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "created_by": "ocid1.datasciencenotebooksession.oc1.iad.<OCID>",
            "defined_tags": {},
            "display_name": "Model1",
            "freeform_tags": {
                "OCI_AQUA": "active",
                "aqua_service_model": "ocid1.datasciencemodel.oc1.iad.<OCID>#Model1",
                "license": "UPL",
                "organization": "Oracle AI",
                "task": "text_generation",
            },
            "id": "ocid1.datasciencemodel.oc1.iad.<OCID>",
            "lifecycle_state": "ACTIVE",
            "project_id": "ocid1.datascienceproject.oc1.iad.<OCID>",
            "time_created": "2024-01-19T17:57:39.158000+00:00",
        },
        {
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "created_by": "ocid1.datasciencenotebooksession.oc1.iad.<OCID>",
            "defined_tags": {},
            "display_name": "VerifiedModel",
            "freeform_tags": {
                "OCI_AQUA": "",
                "license": "UPL",
                "organization": "Oracle AI",
                "task": "text_generation",
                "ready_to_import": "true",
            },
            "id": "ocid1.datasciencemodel.oc1.iad.<OCID>",
            "lifecycle_state": "ACTIVE",
            "project_id": "ocid1.datascienceproject.oc1.iad.<OCID>",
            "time_created": "2024-01-19T17:57:39.158000+00:00",
        },
    ]

    resource_summary_objects = [
        {
            "additional_details": {},
            "availability_domain": "",
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "defined_tags": {},
            "display_name": "Model1-Fine-Tuned",
            "freeform_tags": {
                "OCI_AQUA": "",
                "aqua_fine_tuned_model": "ocid1.datasciencemodel.oc1.iad.<OCID>#Model1",
                "license": "UPL",
                "organization": "Oracle AI",
                "task": "text_generation",
            },
            "identifier": "ocid1.datasciencemodel.oc1.iad.<OCID>",
            "identity_context": {},
            "lifecycle_state": "ACTIVE",
            "resource_type": "DataScienceModel",
            "search_context": "",
            "system_tags": {},
            "time_created": "2024-01-19T19:33:58.078000+00:00",
        },
    ]

    SERVICE_COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"
    COMPARTMENT_ID = "ocid1.compartment.oc1..<UNIQUE_OCID>"


@patch("ads.config.COMPARTMENT_OCID", "ocid1.compartment.oc1.<unique_ocid>")
@patch("ads.config.PROJECT_OCID", "ocid1.datascienceproject.oc1.iad.<unique_ocid>")
class TestAquaModel:
    """Contains unittests for AquaModelApp."""

    # @pytest.fixture(autouse=True, scope="class")
    # def mock_auth(cls):
    #     with patch("ads.common.auth.default_signer") as mock_default_signer:
    #         yield mock_default_signer
    #
    # @pytest.fixture(autouse=True, scope="class")
    # def mock_init_client(cls):
    #     with patch(
    #         "ads.common.oci_datascience.OCIDataScienceMixin.init_client"
    #     ) as mock_client:
    #         yield mock_client

    def setup_method(self):
        self.default_signer_patch = patch(
            "ads.common.auth.default_signer", new_callable=MagicMock
        )
        self.create_signer_patch = patch(
            "ads.common.auth.APIKey.create_signer", new_callable=MagicMock
        )
        self.validate_config_patch = patch(
            "oci.config.validate_config", new_callable=MagicMock
        )
        self.create_client_patch = patch(
            "ads.common.oci_client.OCIClientFactory.create_client",
            new_callable=MagicMock,
        )
        self.mock_default_signer = self.default_signer_patch.start()
        self.mock_create_signer = self.create_signer_patch.start()
        self.mock_validate_config = self.validate_config_patch.start()
        self.mock_create_client = self.create_client_patch.start()
        self.app = AquaModelApp()

    def teardown_method(self):
        self.default_signer_patch.stop()
        self.create_signer_patch.stop()
        self.validate_config_patch.stop()
        self.create_client_patch.stop()
        get_hf_model_info.cache_clear()

    @classmethod
    def setup_class(cls):
        os.environ["CONDA_BUCKET_NS"] = "test-namespace"
        os.environ["ODSC_MODEL_COMPARTMENT_OCID"] = TestDataset.SERVICE_COMPARTMENT_ID
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.model.model)

    @classmethod
    def teardown_class(cls):
        os.environ.pop("CONDA_BUCKET_NS", None)
        os.environ.pop("ODSC_MODEL_COMPARTMENT_OCID", None)
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.model.model)

    @patch.object(DataScienceModel, "create")
    @patch("ads.model.datascience_model.validate")
    @patch.object(DataScienceModel, "from_id")
    def test_create_model(self, mock_from_id, mock_validate, mock_create):
        mock_model = MagicMock()
        mock_model.model_file_description = {"test_key": "test_value"}
        mock_model.display_name = "test_display_name"
        mock_model.description = "test_description"
        mock_model.freeform_tags = {
            "OCI_AQUA": "ACTIVE",
            "license": "test_license",
            "organization": "test_organization",
            "task": "test_task",
            "ready_to_fine_tune": "true",
        }
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            **{"key": "test_metadata_item_key", "value": "test_metadata_item_value"}
        )
        mock_model.custom_metadata_list = custom_metadata_list
        mock_model.provenance_metadata = ModelProvenanceMetadata(
            training_id="test_training_id"
        )
        mock_from_id.return_value = mock_model

        # will not copy service model
        self.app.create(
            model_id="test_model_id",
            project_id="test_project_id",
            compartment_id="test_compartment_id",
        )

        mock_from_id.assert_called_with("test_model_id")
        mock_validate.assert_not_called()
        mock_create.assert_not_called()

        mock_model.compartment_id = TestDataset.SERVICE_COMPARTMENT_ID
        mock_from_id.return_value = mock_model
        mock_create.return_value = mock_model

        # will copy service model
        model = self.app.create(
            model_id="test_model_id",
            project_id="test_project_id",
            compartment_id="test_compartment_id",
        )

        mock_from_id.assert_called_with("test_model_id")
        mock_validate.assert_called()
        mock_create.assert_called_with(model_by_reference=True)

        assert model.display_name == "test_display_name"
        assert model.description == "test_description"
        assert model.description == "test_description"
        assert model.freeform_tags == {
            "OCI_AQUA": "ACTIVE",
            "license": "test_license",
            "organization": "test_organization",
            "task": "test_task",
            "ready_to_fine_tune": "true",
        }
        assert (
            model.custom_metadata_list.get("test_metadata_item_key").value
            == "test_metadata_item_value"
        )
        assert model.provenance_metadata.training_id == "test_training_id"

    @pytest.mark.parametrize(
        "foundation_model_type",
        [
            "service",
            "verified",
        ],
    )
    @patch("ads.aqua.model.model.get_container_config")
    @patch("ads.aqua.model.model.read_file")
    @patch.object(DataScienceModel, "from_id")
    @patch(
        "ads.aqua.model.model.get_artifact_path",
        return_value="oci://bucket@namespace/prefix",
    )
    def test_get_foundation_models(
        self,
        mock_get_artifact_path,
        mock_from_id,
        mock_read_file,
        mock_get_container_config,
        foundation_model_type,
        mock_auth,
    ):
        mock_get_container_config.return_value = get_container_config()
        ds_model = MagicMock()
        ds_model.id = "test_id"
        ds_model.compartment_id = "test_compartment_id"
        ds_model.project_id = "test_project_id"
        ds_model.display_name = "test_display_name"
        ds_model.description = "test_description"
        ds_model.freeform_tags = {
            "OCI_AQUA": "" if foundation_model_type == "verified" else "ACTIVE",
            "license": "test_license",
            "organization": "test_organization",
            "task": "test_task",
        }
        if foundation_model_type == "verified":
            ds_model.freeform_tags["ready_to_import"] = "true"
        ds_model.time_created = "2024-01-19T17:57:39.158000+00:00"
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            **{
                "key": "artifact_location",
                "value": "oci://bucket@namespace/prefix/",
            }
        )
        custom_metadata_list.add(
            **{
                "key": "deployment-container",
                "value": "odsc-vllm-serving",
            }
        )
        custom_metadata_list.add(
            **{
                "key": "evaluation-container",
                "value": "odsc-llm-evaluate",
            }
        )
        custom_metadata_list.add(
            **{
                "key": "finetune-container",
                "value": "odsc-llm-fine-tuning",
            }
        )
        ds_model.custom_metadata_list = custom_metadata_list

        mock_from_id.return_value = ds_model
        mock_read_file.return_value = "test_model_card"

        model_id = (
            "verified_model_id"
            if foundation_model_type == "verified"
            else "service_model_id"
        )
        aqua_model = self.app.get(model_id=model_id)

        mock_from_id.assert_called_with(model_id)

        if foundation_model_type == "verified":
            mock_read_file.assert_called_with(
                file_path="oci://bucket@namespace/prefix/config/README.md",
                auth=mock_auth(),
            )
        else:
            mock_read_file.assert_called_with(
                file_path="oci://bucket@namespace/prefix/README.md",
                auth=mock_auth(),
            )

        assert asdict(aqua_model) == {
            "arm_cpu_supported": False,
            "artifact_location": "oci://bucket@namespace/prefix/",
            "compartment_id": f"{ds_model.compartment_id}",
            "console_link": f"https://cloud.oracle.com/data-science/models/{ds_model.id}?region={self.app.region}",
            "icon": "",
            "id": f"{ds_model.id}",
            "is_fine_tuned_model": False,
            "license": f'{ds_model.freeform_tags["license"]}',
            "model_card": f"{mock_read_file.return_value}",
            "model_formats": [ModelFormat.SAFETENSORS],
            "model_file": "",
            "name": f"{ds_model.display_name}",
            "nvidia_gpu_supported": True,
            "organization": f'{ds_model.freeform_tags["organization"]}',
            "project_id": f"{ds_model.project_id}",
            "ready_to_deploy": False if foundation_model_type == "verified" else True,
            "ready_to_finetune": False,
            "ready_to_import": True if foundation_model_type == "verified" else False,
            "search_text": (
                ",test_license,test_organization,test_task,true"
                if foundation_model_type == "verified"
                else "ACTIVE,test_license,test_organization,test_task"
            ),
            "tags": ds_model.freeform_tags,
            "task": f'{ds_model.freeform_tags["task"]}',
            "time_created": f"{ds_model.time_created}",
            "inference_container": "odsc-vllm-serving",
            "inference_container_uri": None,
            "finetuning_container": "odsc-llm-fine-tuning",
            "evaluation_container": "odsc-llm-evaluate",
        }

    @patch("ads.aqua.common.utils.query_resource")
    @patch("ads.aqua.model.model.read_file")
    @patch.object(DataScienceModel, "from_id")
    @patch(
        "ads.aqua.model.model.get_artifact_path",
        return_value="oci://bucket@namespace/prefix",
    )
    def test_get_model_fine_tuned(
        self,
        mock_get_artifact_path,
        mock_from_id,
        mock_read_file,
        mock_query_resource,
        mock_get_container_config,
        mock_auth,
    ):
        mock_get_container_config.return_value = get_container_config()
        ds_model = MagicMock()
        ds_model.id = "test_id"
        ds_model.compartment_id = "test_model_compartment_id"
        ds_model.project_id = "test_project_id"
        ds_model.display_name = "test_display_name"
        ds_model.description = "test_description"
        ds_model.model_version_set_id = "test_model_version_set_id"
        ds_model.model_version_set_name = "test_model_version_set_name"
        ds_model.freeform_tags = {
            "OCI_AQUA": "ACTIVE",
            "license": "test_license",
            "organization": "test_organization",
            "task": "test_task",
            "aqua_fine_tuned_model": "test_finetuned_model",
        }
        self.app._service_model_details_cache.get = MagicMock(return_value=None)
        ds_model.time_created = "2024-01-19T17:57:39.158000+00:00"
        ds_model.lifecycle_state = "ACTIVE"
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            **{"key": "artifact_location", "value": "oci://bucket@namespace/prefix/"}
        )
        custom_metadata_list.add(
            **{"key": "fine_tune_source", "value": "test_fine_tuned_source_id"}
        )
        custom_metadata_list.add(
            **{"key": "fine_tune_source_name", "value": "test_fine_tuned_source_name"}
        )
        custom_metadata_list.add(
            **{
                "key": "deployment-container",
                "value": "odsc-vllm-serving",
            }
        )
        custom_metadata_list.add(
            **{
                "key": "evaluation-container",
                "value": "odsc-llm-evaluate",
            }
        )
        custom_metadata_list.add(
            **{
                "key": "finetune-container",
                "value": "odsc-llm-fine-tuning",
            }
        )
        ds_model.custom_metadata_list = custom_metadata_list
        defined_metadata_list = ModelTaxonomyMetadata()
        defined_metadata_list["Hyperparameters"].value = {
            "training_data": "test_training_data",
            "val_set_size": "test_val_set_size",
        }
        ds_model.defined_metadata_list = defined_metadata_list
        ds_model.provenance_metadata = ModelProvenanceMetadata(
            training_id="test_training_job_run_id"
        )

        mock_from_id.return_value = ds_model
        mock_read_file.return_value = "test_model_card"

        response = MagicMock()
        job_run = MagicMock()
        job_run.id = "test_job_run_id"
        job_run.lifecycle_state = "SUCCEEDED"
        job_run.lifecycle_details = "test lifecycle details"
        job_run.identifier = ("test_job_id",)
        job_run.display_name = "test_job_name"
        job_run.compartment_id = "test_job_run_compartment_id"
        job_infrastructure_configuration_details = MagicMock()
        job_infrastructure_configuration_details.shape_name = "test_shape_name"

        job_configuration_override_details = MagicMock()
        job_configuration_override_details.environment_variables = {"NODE_COUNT": 1}
        job_run.job_infrastructure_configuration_details = (
            job_infrastructure_configuration_details
        )
        job_run.job_configuration_override_details = job_configuration_override_details
        log_details = MagicMock()
        log_details.log_id = "test_log_id"
        log_details.log_group_id = "test_log_group_id"
        job_run.log_details = log_details
        response.data = job_run
        self.app.ds_client.get_job_run = MagicMock(return_value=response)

        query_resource = MagicMock()
        query_resource.display_name = "test_display_name"
        mock_query_resource.return_value = query_resource

        model = self.app.get(model_id="test_model_id")

        mock_from_id.assert_called_with("test_model_id")
        mock_read_file.assert_called_with(
            file_path="oci://bucket@namespace/prefix/README.md",
            auth=mock_auth(),
        )
        mock_query_resource.assert_called()

        assert asdict(model) == {
            "arm_cpu_supported": False,
            "artifact_location": "oci://bucket@namespace/prefix/",
            "compartment_id": f"{ds_model.compartment_id}",
            "console_link": f"https://cloud.oracle.com/data-science/models/{ds_model.id}?region={self.app.region}",
            "dataset": "test_training_data",
            "experiment": {"id": "", "name": "", "url": ""},
            "icon": "",
            "id": f"{ds_model.id}",
            "is_fine_tuned_model": True,
            "job": {"id": "", "name": "", "url": ""},
            "license": "test_license",
            "lifecycle_details": f"{job_run.lifecycle_details}",
            "lifecycle_state": f"{ds_model.lifecycle_state}",
            "log": {
                "id": f"{log_details.log_id}",
                "name": f"{query_resource.display_name}",
                "url": "https://cloud.oracle.com/logging/search?searchQuery=search "
                f'"{job_run.compartment_id}/{log_details.log_group_id}/{log_details.log_id}" | '
                f"source='{job_run.id}' | sort by datetime desc&regions={self.app.region}",
            },
            "log_group": {
                "id": f"{log_details.log_group_id}",
                "name": f"{query_resource.display_name}",
                "url": f"https://cloud.oracle.com/logging/log-groups/{log_details.log_group_id}?region={self.app.region}",
            },
            "metrics": [
                {"category": "validation", "name": "validation_metrics", "scores": []},
                {"category": "training", "name": "training_metrics", "scores": []},
                {
                    "category": "validation",
                    "name": "validation_metrics_final",
                    "scores": [],
                },
                {
                    "category": "training",
                    "name": "training_metrics_final",
                    "scores": [],
                },
            ],
            "model_card": f"{mock_read_file.return_value}",
            "model_formats": [ModelFormat.SAFETENSORS],
            "model_file": "",
            "name": f"{ds_model.display_name}",
            "nvidia_gpu_supported": True,
            "organization": "test_organization",
            "project_id": f"{ds_model.project_id}",
            "ready_to_deploy": True,
            "ready_to_finetune": False,
            "ready_to_import": False,
            "search_text": "ACTIVE,test_license,test_organization,test_task,test_finetuned_model",
            "shape_info": {
                "instance_shape": f"{job_infrastructure_configuration_details.shape_name}",
                "replica": 1,
            },
            "source": {"id": "", "name": "", "url": ""},
            "tags": ds_model.freeform_tags,
            "task": "test_task",
            "time_created": f"{ds_model.time_created}",
            "validation": {"type": "Automatic split", "value": "test_val_set_size"},
            "inference_container": "odsc-vllm-serving",
            "inference_container_uri": None,
            "finetuning_container": "odsc-llm-fine-tuning",
            "evaluation_container": "odsc-llm-evaluate",
        }

    @pytest.mark.parametrize(
        ("artifact_location_set", "download_from_hf", "cleanup_model_cache"),
        [
            (True, True, True),
            (True, False, True),
            (False, True, False),
            (False, False, True),
        ],
    )
    @patch("ads.model.service.oci_datascience_model.OCIDataScienceModel.create")
    @patch("ads.model.datascience_model.DataScienceModel.sync")
    @patch("ads.model.datascience_model.DataScienceModel.upload_artifact")
    @patch.object(AquaModelApp, "_find_matching_aqua_model")
    @patch("ads.common.object_storage_details.ObjectStorageDetails.list_objects")
    @patch("ads.aqua.common.utils.load_config", return_value={})
    @patch("huggingface_hub.snapshot_download")
    @patch("subprocess.check_call")
    def test_import_verified_model(
        self,
        mock_subprocess,
        mock_snapshot_download,
        mock_load_config,
        mock_list_objects,
        mock__find_matching_aqua_model,
        mock_upload_artifact,
        mock_sync,
        mock_ocidsc_create,
        artifact_location_set,
        download_from_hf,
        cleanup_model_cache,
        mock_get_hf_model_info,
        mock_init_client,
    ):
        ObjectStorageDetails.is_bucket_versioned = MagicMock(return_value=True)
        # The name attribute cannot be mocked during creation of the mock object,
        # hence attach it separately to the mocked objects.
        artifact_path = "service_models/model-name/commit-id/artifact"
        obj1 = MagicMock(etag="12345-1234-1234-1234-123456789", size=150)
        obj1.name = f"{artifact_path}/config/deployment_config.json"
        obj2 = MagicMock(etag="12345-1234-1234-1234-123456789", size=150)
        obj2.name = f"{artifact_path}/config/ft_config.json"
        objects = [obj1, obj2]
        mock_list_objects.return_value = MagicMock(objects=objects)

        ds_model = DataScienceModel()
        os_path = "oci://aqua-bkt@aqua-ns/prefix/path"
        model_name = "oracle/aqua-1t-mega-model"
        ds_freeform_tags = {
            "OCI_AQUA": "ACTIVE",
            "license": "aqua-license",
            "model_format": "SAFETENSORS",
            "organization": "oracle",
            "task": "text-generation",
            "ready_to_import": "true",
        }
        ds_model = (
            ds_model.with_compartment_id("test_model_compartment_id")
            .with_project_id("test_project_id")
            .with_display_name(model_name)
            .with_description("test_description")
            .with_model_version_set_id("test_model_version_set_id")
            .with_freeform_tags(**ds_freeform_tags)
            .with_version_id("ocid1.blah.blah")
        )
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            **{"key": "deployment-container", "value": "odsc-tgi-serving"}
        )
        custom_metadata_list.add(
            **{"key": "evaluation-container", "value": "odsc-llm-evaluate"}
        )
        if not artifact_location_set:
            custom_metadata_list.add(
                **{
                    "key": "artifact_location",
                    "value": artifact_path,
                    "description": "artifact location",
                }
            )
        ds_model.with_custom_metadata_list(custom_metadata_list)
        ds_model.set_spec(ds_model.CONST_MODEL_FILE_DESCRIPTION, {})
        ds_model.dsc_model = MagicMock(id="test_model_id")
        DataScienceModel.from_id = MagicMock(return_value=ds_model)
        mock__find_matching_aqua_model.return_value = "test_model_id"

        reload(ads.aqua.model.model)
        app = AquaModelApp()
        if download_from_hf:
            with tempfile.TemporaryDirectory() as tmpdir:
                mock_snapshot_download.return_value = f"{str(tmpdir)}/{model_name}"
                model: AquaModel = app.register(
                    model=model_name,
                    os_path=os_path,
                    local_dir=str(tmpdir),
                    download_from_hf=True,
                    cleanup_model_cache=cleanup_model_cache,
                    allow_patterns=["*.json"],
                    ignore_patterns=["test.json"],
                )
                mock_snapshot_download.assert_called_with(
                    repo_id=model_name,
                    local_dir=f"{str(tmpdir)}/{model_name}",
                    allow_patterns=["*.json"],
                    ignore_patterns=["test.json"],
                )
                mock_subprocess.assert_called_with(
                    shlex.split(
                        f"oci os object bulk-upload --src-dir {str(tmpdir)}/{model_name} --prefix prefix/path/{model_name}/ -bn aqua-bkt -ns aqua-ns --auth api_key --profile DEFAULT --no-overwrite --exclude {HF_METADATA_FOLDER}*"
                    )
                )
                if cleanup_model_cache:
                    cache_dir = os.path.join(
                        os.path.expanduser("~"),
                        ".cache",
                        "huggingface",
                        "hub",
                        "models--oracle--aqua-1t-mega-model",
                    )
                    assert (
                        os.path.exists(f"{str(tmpdir)}/{os.path.dirname(model_name)}")
                        is False
                    )
                    assert os.path.exists(cache_dir) is False

        else:
            model: AquaModel = app.register(
                model="ocid1.datasciencemodel.xxx.xxxx.",
                os_path=os_path,
                download_from_hf=False,
            )
            mock_snapshot_download.assert_not_called()
            mock_subprocess.assert_not_called()
            mock_load_config.assert_called()

        ds_freeform_tags.pop(
            "ready_to_import"
        )  # The imported model should not have this tag
        assert model.tags == {
            "aqua_custom_base_model": "true",
            "aqua_service_model": "test_model_id",
            **ds_freeform_tags,
        }

        assert model.inference_container == "odsc-tgi-serving"
        assert model.finetuning_container is None
        assert model.evaluation_container == "odsc-llm-evaluate"
        assert model.ready_to_import is False
        assert model.ready_to_deploy is True
        assert model.ready_to_finetune is False

    @patch("ads.model.service.oci_datascience_model.OCIDataScienceModel.create")
    @patch("ads.model.datascience_model.DataScienceModel.sync")
    @patch("ads.model.datascience_model.DataScienceModel.upload_artifact")
    @patch.object(AquaModelApp, "_validate_model")
    @patch("ads.aqua.common.utils.load_config", return_value={})
    def test_import_any_model_no_containers_specified(
        self,
        mock_load_config,
        mock__validate_model,
        mock_upload_artifact,
        mock_sync,
        mock_ocidsc_create,
        mock_get_hf_model_info,
    ):
        ObjectStorageDetails.is_bucket_versioned = MagicMock(return_value=True)
        os_path = "oci://aqua-bkt@aqua-ns/prefix/path"
        model_name = "oracle/aqua-1t-mega-model"
        ds_freeform_tags = {
            "OCI_AQUA": "ACTIVE",
            "license": "aqua-license",
            "organization": "oracle",
            "task": "text-generation",
        }
        mock__validate_model.return_value = ModelValidationResult(
            model_file="model_file.gguf",
            model_formats=[ModelFormat.SAFETENSORS],
            telemetry_model_name=model_name,
            tags=ds_freeform_tags,
        )

        reload(ads.aqua.model.model)
        app = AquaModelApp()
        with pytest.raises(AquaRuntimeError):
            with patch.object(AquaModelApp, "list") as aqua_model_mock_list:
                aqua_model_mock_list.return_value = [
                    AquaModelSummary(
                        id="test_id1",
                        name="organization1/name1",
                        organization="organization1",
                    ),
                ]
                model: AquaModel = app.register(
                    model=model_name,
                    os_path=os_path,
                    download_from_hf=False,
                )

    @pytest.mark.parametrize(
        "download_from_hf",
        [True, False],
    )
    @patch("ads.model.service.oci_datascience_model.OCIDataScienceModel.create")
    @patch("ads.model.datascience_model.DataScienceModel.sync")
    @patch("ads.model.datascience_model.DataScienceModel.upload_artifact")
    @patch.object(AquaModelApp, "_find_matching_aqua_model")
    @patch("ads.common.object_storage_details.ObjectStorageDetails.list_objects")
    @patch("ads.aqua.common.utils.load_config", return_value={})
    @patch("huggingface_hub.snapshot_download")
    @patch("subprocess.check_call")
    def test_import_model_with_project_compartment_override(
        self,
        mock_subprocess,
        mock_snapshot_download,
        mock_load_config,
        mock_list_objects,
        mock__find_matching_aqua_model,
        mock_upload_artifact,
        mock_sync,
        mock_ocidsc_create,
        download_from_hf,
        mock_get_hf_model_info,
    ):
        ObjectStorageDetails.is_bucket_versioned = MagicMock(return_value=True)

        mock_list_objects.return_value = MagicMock(objects=[])
        ds_model = DataScienceModel()
        os_path = "oci://aqua-bkt@aqua-ns/prefix/path"
        model_name = "oracle/aqua-1t-mega-model"
        compartment_override = "my.blah.compartment"
        project_override = "my.blah.project"
        ds_freeform_tags = {
            "OCI_AQUA": "ACTIVE",
            "license": "aqua-license",
            "organization": "oracle",
            "task": "text-generation",
        }
        ds_model = (
            ds_model.with_compartment_id("test_model_compartment_id")
            .with_project_id("test_project_id")
            .with_display_name(model_name)
            .with_description("test_description")
            .with_model_version_set_id("test_model_version_set_id")
            .with_freeform_tags(**ds_freeform_tags)
            .with_version_id("ocid1.blah.blah")
        )
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            **{"key": "deployment-container", "value": "odsc-tgi-serving"}
        )
        custom_metadata_list.add(
            **{"key": "evaluation-container", "value": "odsc-llm-evaluate"}
        )
        ds_model.with_custom_metadata_list(custom_metadata_list)
        ds_model.set_spec(ds_model.CONST_MODEL_FILE_DESCRIPTION, {})
        DataScienceModel.from_id = MagicMock(return_value=ds_model)
        mock__find_matching_aqua_model.return_value = "test_model_id"
        reload(ads.aqua.model.model)
        app = AquaModelApp()

        if download_from_hf:
            with tempfile.TemporaryDirectory() as tmpdir:
                model: AquaModel = app.register(
                    compartment_id=compartment_override,
                    project_id=project_override,
                    model=model_name,
                    os_path=os_path,
                    local_dir=str(tmpdir),
                    download_from_hf=True,
                )
        else:
            model: AquaModel = app.register(
                compartment_id=compartment_override,
                project_id=project_override,
                model="ocid1.datasciencemodel.xxx.xxxx.",
                os_path=os_path,
                download_from_hf=False,
            )
        assert model.compartment_id == compartment_override
        assert model.project_id == project_override

    @pytest.mark.parametrize(
        ("ignore_artifact_check", "download_from_hf"),
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
            (None, False),
            (None, True),
        ],
    )
    @patch("ads.model.service.oci_datascience_model.OCIDataScienceModel.create")
    @patch("ads.model.datascience_model.DataScienceModel.sync")
    @patch("ads.model.datascience_model.DataScienceModel.upload_artifact")
    @patch("ads.common.object_storage_details.ObjectStorageDetails.list_objects")
    @patch("ads.aqua.common.utils.load_config", side_effect=AquaFileNotFoundError)
    @patch("huggingface_hub.snapshot_download")
    @patch("subprocess.check_call")
    def test_import_model_with_missing_config(
        self,
        mock_subprocess,
        mock_snapshot_download,
        mock_load_config,
        mock_list_objects,
        mock_upload_artifact,
        mock_sync,
        mock_ocidsc_create,
        ignore_artifact_check,
        download_from_hf,
        mock_get_hf_model_info,
        mock_init_client,
    ):
        my_model = "oracle/aqua-1t-mega-model"
        ObjectStorageDetails.is_bucket_versioned = MagicMock(return_value=True)
        # set object list from OSS without config.json
        os_path = "oci://aqua-bkt@aqua-ns/prefix/path"

        # set object list from HF without config.json
        if download_from_hf:
            mock_get_hf_model_info.return_value.siblings = [
                MagicMock(rfilename="model.safetensors")
            ]
        else:
            obj1 = MagicMock(etag="12345-1234-1234-1234-123456789", size=150)
            obj1.name = f"prefix/path/model.safetensors"
            objects = [obj1]
            mock_list_objects.return_value = MagicMock(objects=objects)

        reload(ads.aqua.model.model)
        app = AquaModelApp()
        with patch.object(AquaModelApp, "list") as aqua_model_mock_list:
            aqua_model_mock_list.return_value = [
                AquaModelSummary(
                    id="test_id1",
                    name="organization1/name1",
                    organization="organization1",
                )
            ]

            if ignore_artifact_check:
                model: AquaModel = app.register(
                    model=my_model,
                    os_path=os_path,
                    inference_container="odsc-vllm-or-tgi-container",
                    finetuning_container="odsc-llm-fine-tuning",
                    download_from_hf=download_from_hf,
                    ignore_model_artifact_check=ignore_artifact_check,
                )
                assert model.ready_to_deploy is True
            else:
                with pytest.raises(AquaRuntimeError):
                    model: AquaModel = app.register(
                        model=my_model,
                        os_path=os_path,
                        inference_container="odsc-vllm-or-tgi-container",
                        finetuning_container="odsc-llm-fine-tuning",
                        download_from_hf=download_from_hf,
                        ignore_model_artifact_check=ignore_artifact_check,
                    )

    @patch("ads.model.service.oci_datascience_model.OCIDataScienceModel.create")
    @patch("ads.model.datascience_model.DataScienceModel.sync")
    @patch("ads.model.datascience_model.DataScienceModel.upload_artifact")
    @patch("ads.common.object_storage_details.ObjectStorageDetails.list_objects")
    @patch("ads.aqua.common.utils.load_config", return_value={})
    def test_import_any_model_smc_container(
        self,
        mock_load_config,
        mock_list_objects,
        mock_upload_artifact,
        mock_sync,
        mock_ocidsc_create,
        mock_get_hf_model_info,
        mock_init_client,
    ):
        my_model = "oracle/aqua-1t-mega-model"
        ObjectStorageDetails.is_bucket_versioned = MagicMock(return_value=True)

        os_path = "oci://aqua-bkt@aqua-ns/prefix/path"
        ds_freeform_tags = {
            "OCI_AQUA": "active",
        }
        mock_list_objects.return_value = MagicMock(objects=[])

        reload(ads.aqua.model.model)
        app = AquaModelApp()
        with patch.object(AquaModelApp, "list") as aqua_model_mock_list:
            aqua_model_mock_list.return_value = [
                AquaModelSummary(
                    id="test_id1",
                    name="organization1/name1",
                    organization="organization1",
                ),
                AquaModelSummary(
                    id="test_id2",
                    name="organization1/name2",
                    organization="organization1",
                ),
                AquaModelSummary(
                    id="test_id3",
                    name="organization2/name3",
                    organization="organization2",
                ),
            ]
            model: AquaModel = app.register(
                model=my_model,
                os_path=os_path,
                inference_container="odsc-vllm-or-tgi-container",
                finetuning_container="odsc-llm-fine-tuning",
                download_from_hf=False,
            )
            assert model.tags == {
                "aqua_custom_base_model": "true",
                "model_format": "SAFETENSORS",
                "ready_to_fine_tune": "true",
                **ds_freeform_tags,
            }
            assert model.inference_container == "odsc-vllm-or-tgi-container"
            assert model.finetuning_container == "odsc-llm-fine-tuning"
            assert model.evaluation_container == "odsc-llm-evaluate"
            assert model.ready_to_import is False
            assert model.ready_to_deploy is True
            assert model.ready_to_finetune is True

    @pytest.mark.parametrize(
        "download_from_hf",
        [True, False],
    )
    @patch("ads.model.service.oci_datascience_model.OCIDataScienceModel.create")
    @patch("ads.model.datascience_model.DataScienceModel.sync")
    @patch("ads.model.datascience_model.DataScienceModel.upload_artifact")
    @patch.object(AquaModelApp, "_find_matching_aqua_model")
    @patch("ads.common.object_storage_details.ObjectStorageDetails.list_objects")
    @patch("ads.aqua.common.utils.load_config", return_value={})
    @patch("huggingface_hub.snapshot_download")
    @patch("subprocess.check_call")
    def test_import_tei_model_byoc(
        self,
        mock_subprocess,
        mock_snapshot_download,
        mock_load_config,
        mock_list_objects,
        mock__find_matching_aqua_model,
        mock_upload_artifact,
        mock_sync,
        mock_ocidsc_create,
        download_from_hf,
        mock_get_hf_model_info,
        mock_init_client,
    ):
        ObjectStorageDetails.is_bucket_versioned = MagicMock(return_value=True)

        artifact_path = "service_models/model-name/commit-id/artifact"
        obj1 = MagicMock(etag="12345-1234-1234-1234-123456789", size=150)
        obj1.name = f"{artifact_path}/config.json"
        objects = [obj1]
        mock_list_objects.return_value = MagicMock(objects=objects)
        ds_model = DataScienceModel()
        os_path = "oci://aqua-bkt@aqua-ns/prefix/path"
        model_name = "oracle/aqua-1t-mega-model"
        ds_freeform_tags = {
            "OCI_AQUA": "ACTIVE",
            "license": "aqua-license",
            "organization": "oracle",
            "task": "text_embedding",
        }
        ds_model = (
            ds_model.with_compartment_id("test_model_compartment_id")
            .with_project_id("test_project_id")
            .with_display_name(model_name)
            .with_description("test_description")
            .with_model_version_set_id("test_model_version_set_id")
            .with_freeform_tags(**ds_freeform_tags)
            .with_version_id("ocid1.version.id")
        )
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            **{"key": "deployment-container", "value": "odsc-tei-serving"}
        )
        ds_model.with_custom_metadata_list(custom_metadata_list)
        ds_model.set_spec(ds_model.CONST_MODEL_FILE_DESCRIPTION, {})
        DataScienceModel.from_id = MagicMock(return_value=ds_model)
        mock__find_matching_aqua_model.return_value = None
        reload(ads.aqua.model.model)
        app = AquaModelApp()

        if download_from_hf:
            with tempfile.TemporaryDirectory() as tmpdir:
                model: AquaModel = app.register(
                    model=model_name,
                    os_path=os_path,
                    local_dir=str(tmpdir),
                    download_from_hf=True,
                    inference_container="odsc-tei-serving",
                    inference_container_uri="region.ocir.io/your_tenancy/your_image",
                )
        else:
            model: AquaModel = app.register(
                model="ocid1.datasciencemodel.xxx.xxxx.",
                os_path=os_path,
                download_from_hf=False,
                inference_container="odsc-tei-serving",
                inference_container_uri="region.ocir.io/your_tenancy/your_image",
            )
        assert model.inference_container == "odsc-tei-serving"
        assert model.ready_to_deploy is True
        assert model.ready_to_finetune is False

    @patch("ads.model.service.oci_datascience_model.OCIDataScienceModel.create")
    @patch("ads.model.datascience_model.DataScienceModel.sync")
    @patch("ads.model.datascience_model.DataScienceModel.upload_artifact")
    @patch("ads.common.object_storage_details.ObjectStorageDetails.list_objects")
    @patch.object(HfApi, "model_info")
    @patch("ads.aqua.common.utils.load_config", return_value={})
    def test_import_model_with_input_tags(
        self,
        mock_load_config,
        mock_list_objects,
        mock_upload_artifact,
        mock_sync,
        mock_ocidsc_create,
        mock_get_hf_model_info,
        mock_init_client,
    ):
        my_model = "oracle/aqua-1t-mega-model"
        ObjectStorageDetails.is_bucket_versioned = MagicMock(return_value=True)

        os_path = "oci://aqua-bkt@aqua-ns/prefix/path"
        ds_freeform_tags = {
            "OCI_AQUA": "active",
        }
        mock_list_objects.return_value = MagicMock(objects=[])

        reload(ads.aqua.model.model)
        app = AquaModelApp()
        with patch.object(AquaModelApp, "list") as aqua_model_mock_list:
            aqua_model_mock_list.return_value = [
                AquaModelSummary(
                    id="test_id1",
                    name="organization1/name1",
                    organization="organization1",
                )
            ]
            model: AquaModel = app.register(
                model=my_model,
                os_path=os_path,
                inference_container="odsc-vllm-or-tgi-container",
                finetuning_container="odsc-llm-fine-tuning",
                download_from_hf=False,
                freeform_tags={"ftag1": "fvalue1", "ftag2": "fvalue2"},
                defined_tags={"dtag1": "dvalue1", "dtag2": "dvalue2"},
            )
            assert model.tags == {
                "aqua_custom_base_model": "true",
                "model_format": "SAFETENSORS",
                "ready_to_fine_tune": "true",
                "dtag1": "dvalue1",
                "dtag2": "dvalue2",
                "ftag1": "fvalue1",
                "ftag2": "fvalue2",
                **ds_freeform_tags,
            }

    @pytest.mark.parametrize(
        "data, expected_output",
        [
            (
                {
                    "os_path": "oci://aqua-bkt@aqua-ns/path",
                    "model": "oracle/oracle-1it",
                    "inference_container": "odsc-vllm-serving",
                },
                "ads aqua model register --model oracle/oracle-1it --os_path oci://aqua-bkt@aqua-ns/path --download_from_hf True --cleanup_model_cache False --inference_container odsc-vllm-serving",
            ),
            (
                {
                    "os_path": "oci://aqua-bkt@aqua-ns/path",
                    "model": "ocid1.datasciencemodel.oc1.iad.<OCID>",
                },
                "ads aqua model register --model ocid1.datasciencemodel.oc1.iad.<OCID> --os_path oci://aqua-bkt@aqua-ns/path --download_from_hf True --cleanup_model_cache False",
            ),
            (
                {
                    "os_path": "oci://aqua-bkt@aqua-ns/path",
                    "model": "oracle/oracle-1it",
                    "download_from_hf": False,
                },
                "ads aqua model register --model oracle/oracle-1it --os_path oci://aqua-bkt@aqua-ns/path --download_from_hf False --cleanup_model_cache False",
            ),
            (
                {
                    "os_path": "oci://aqua-bkt@aqua-ns/path",
                    "model": "oracle/oracle-1it",
                    "download_from_hf": True,
                    "model_file": "test_model_file",
                },
                "ads aqua model register --model oracle/oracle-1it --os_path oci://aqua-bkt@aqua-ns/path --download_from_hf True --cleanup_model_cache False --model_file test_model_file",
            ),
            (
                {
                    "os_path": "oci://aqua-bkt@aqua-ns/path",
                    "model": "oracle/oracle-1it",
                    "inference_container": "odsc-tei-serving",
                    "inference_container_uri": "<region>.ocir.io/<your_tenancy>/<your_image>",
                },
                "ads aqua model register --model oracle/oracle-1it --os_path oci://aqua-bkt@aqua-ns/path --download_from_hf True --cleanup_model_cache False --inference_container odsc-tei-serving --inference_container_uri <region>.ocir.io/<your_tenancy>/<your_image>",
            ),
            (
                {
                    "os_path": "oci://aqua-bkt@aqua-ns/path",
                    "model": "oracle/oracle-1it",
                    "inference_container": "odsc-vllm-serving",
                    "freeform_tags": {"ftag1": "fvalue1", "ftag2": "fvalue2"},
                    "defined_tags": {"dtag1": "dvalue1", "dtag2": "dvalue2"},
                },
                "ads aqua model register --model oracle/oracle-1it --os_path oci://aqua-bkt@aqua-ns/path "
                "--download_from_hf True --cleanup_model_cache False --inference_container odsc-vllm-serving --freeform_tags "
                '{"ftag1": "fvalue1", "ftag2": "fvalue2"} --defined_tags {"dtag1": "dvalue1", "dtag2": "dvalue2"}',
            ),
            (
                {
                    "os_path": "oci://aqua-bkt@aqua-ns/path",
                    "model": "oracle/oracle-1it",
                    "inference_container": "odsc-vllm-serving",
                    "ignore_model_artifact_check": True,
                    "cleanup_model_cache": True,
                },
                "ads aqua model register --model oracle/oracle-1it --os_path oci://aqua-bkt@aqua-ns/path --download_from_hf True --cleanup_model_cache True --inference_container odsc-vllm-serving --ignore_model_artifact_check True",
            ),
        ],
    )
    def test_import_cli(self, data, expected_output):
        import_details = ImportModelDetails(**data)
        assert import_details.build_cli() == expected_output

    @patch("ads.aqua.model.model.read_file")
    @patch("ads.aqua.model.model.get_artifact_path")
    def test_load_license(self, mock_get_artifact_path, mock_read_file):
        self.app.ds_client.get_model = MagicMock()
        mock_get_artifact_path.return_value = (
            "oci://bucket@namespace/prefix/config/LICENSE.txt"
        )
        mock_read_file.return_value = "test_license"

        license = self.app.load_license(model_id="test_model_id")

        mock_get_artifact_path.assert_called()
        mock_read_file.assert_called()

        assert asdict(license) == {"id": "test_model_id", "license": "test_license"}

    def test_list_service_models(self):
        """Tests listing service models succesfully."""

        self.app.list_resource = MagicMock(
            return_value=[
                oci.data_science.models.ModelSummary(**item)
                for item in TestDataset.model_summary_objects
            ]
        )

        results = self.app.list()

        received_args = self.app.list_resource.call_args.kwargs
        assert received_args.get("compartment_id") == TestDataset.SERVICE_COMPARTMENT_ID

        assert len(results) == 2

        attributes = AquaModelSummary.__annotations__.keys()
        for r in results:
            rdict = asdict(r)
            print("############ Expected Response ############")
            print(rdict)

            for attr in attributes:
                assert rdict.get(attr) is not None

    def test_list_custom_models(self):
        """Tests list custom models succesfully."""

        self.app._rqs = MagicMock(
            return_value=[
                oci.resource_search.models.ResourceSummary(**item)
                for item in TestDataset.resource_summary_objects
            ]
        )

        results = self.app.list(TestDataset.COMPARTMENT_ID)

        self.app._rqs.assert_called_with(TestDataset.COMPARTMENT_ID, model_type="FT")

        assert len(results) == 1

        attributes = AquaModelSummary.__annotations__.keys()
        for r in results:
            rdict = asdict(r)
            print("############ Expected Response ############")
            print(rdict)

            for attr in attributes:
                assert rdict.get(attr) is not None

    @parameterized.expand(
        [
            (
                None,
                {"license": "UPL", "org": "Oracle", "task": "text_generation"},
                "UPL,Oracle,text_generation",
            ),
            (
                "This is a description.",
                {"license": "UPL", "org": "Oracle", "task": "text_generation"},
                "This is a description. UPL,Oracle,text_generation",
            ),
        ]
    )
    def test_build_search_text(self, description, tags, expected_output):
        assert (
            self.app._build_search_text(tags=tags, description=description)
            == expected_output
        )
