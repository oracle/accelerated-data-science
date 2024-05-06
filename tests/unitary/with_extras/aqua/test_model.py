#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest
from dataclasses import asdict
from importlib import reload
from unittest.mock import MagicMock, PropertyMock

from mock import patch
import oci
import ads.common
from ads.common.object_storage_details import ObjectStorageDetails
import ads.common.oci_client
from ads.model.service.oci_datascience_model import OCIDataScienceModel
from parameterized import parameterized

import ads.aqua.model
import ads.config
from ads.aqua.model import AquaModelApp, AquaModelSummary
from ads.model.datascience_model import DataScienceModel
from ads.model.model_metadata import (
    ModelCustomMetadata,
    ModelProvenanceMetadata,
    ModelTaxonomyMetadata,
)

import shlex
import tempfile
import huggingface_hub
import pytest


@pytest.fixture(autouse=True, scope="class")
def mock_auth():
    with patch("ads.common.auth.default_signer") as mock_default_signer:
        yield mock_default_signer


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
                "OCI_AQUA": "",
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

    def setup_method(self):
        ads.common.auth.default_signer = MagicMock()
        ads.common.auth.APIKey.create_signer = MagicMock()
        oci.config.validate_config = MagicMock()
        ads.common.oci_client.OCIClientFactory.create_client = MagicMock()
        self.app = AquaModelApp()

    @classmethod
    def setup_class(cls):
        os.environ["CONDA_BUCKET_NS"] = "test-namespace"
        os.environ["ODSC_MODEL_COMPARTMENT_OCID"] = TestDataset.SERVICE_COMPARTMENT_ID
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.model)

    @classmethod
    def teardown_class(cls):
        os.environ.pop("CONDA_BUCKET_NS", None)
        os.environ.pop("ODSC_MODEL_COMPARTMENT_OCID", None)
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.model)

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

    @patch("ads.aqua.model.read_file")
    @patch.object(DataScienceModel, "from_id")
    def test_get_model_not_fine_tuned(self, mock_from_id, mock_read_file):
        ds_model = MagicMock()
        ds_model.id = "test_id"
        ds_model.compartment_id = "test_compartment_id"
        ds_model.project_id = "test_project_id"
        ds_model.display_name = "test_display_name"
        ds_model.description = "test_description"
        ds_model.freeform_tags = {
            "OCI_AQUA": "ACTIVE",
            "license": "test_license",
            "organization": "test_organization",
            "task": "test_task",
        }
        ds_model.time_created = "2024-01-19T17:57:39.158000+00:00"
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            **{"key": "artifact_location", "value": "oci://bucket@namespace/prefix/"}
        )
        ds_model.custom_metadata_list = custom_metadata_list

        mock_from_id.return_value = ds_model
        mock_read_file.return_value = "test_model_card"

        aqua_model = self.app.get(model_id="test_model_id")

        mock_from_id.assert_called_with("test_model_id")
        mock_read_file.assert_called_with(
            file_path="oci://bucket@namespace/prefix/README.md",
            auth=self.app._auth,
        )

        assert asdict(aqua_model) == {
            "compartment_id": f"{ds_model.compartment_id}",
            "console_link": (
                f"https://cloud.oracle.com/data-science/models/{ds_model.id}?region={self.app.region}",
            ),
            "icon": "",
            "id": f"{ds_model.id}",
            "is_fine_tuned_model": False,
            "license": f'{ds_model.freeform_tags["license"]}',
            "model_card": f"{mock_read_file.return_value}",
            "name": f"{ds_model.display_name}",
            "organization": f'{ds_model.freeform_tags["organization"]}',
            "project_id": f"{ds_model.project_id}",
            "ready_to_deploy": True,
            "ready_to_finetune": False,
            "search_text": "ACTIVE,test_license,test_organization,test_task",
            "tags": ds_model.freeform_tags,
            "task": f'{ds_model.freeform_tags["task"]}',
            "time_created": f"{ds_model.time_created}",
        }

    @patch("ads.aqua.utils.query_resource")
    @patch("ads.aqua.model.read_file")
    @patch.object(DataScienceModel, "from_id")
    def test_get_model_fine_tuned(
        self, mock_from_id, mock_read_file, mock_query_resource
    ):
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
            auth=self.app._auth,
        )
        mock_query_resource.assert_called()

        assert asdict(model) == {
            "compartment_id": f"{ds_model.compartment_id}",
            "console_link": (
                f"https://cloud.oracle.com/data-science/models/{ds_model.id}?region={self.app.region}",
            ),
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
            "name": f"{ds_model.display_name}",
            "organization": "test_organization",
            "project_id": f"{ds_model.project_id}",
            "ready_to_deploy": True,
            "ready_to_finetune": False,
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
        }

    @patch("huggingface_hub.snapshot_download")
    @patch("subprocess.check_call")
    def test_import_shadow_model(
        self,
        mock_subprocess,
        mock_snapshot_download,
    ):
        ObjectStorageDetails.is_bucket_versioned = MagicMock(return_value=True)
        ads.common.oci_datascience.OCIDataScienceMixin.init_client = MagicMock()
        huggingface_hub.HfApi.model_info = MagicMock(return_value={})
        DataScienceModel.upload_artifact = MagicMock()
        # oci.data_science.DataScienceClient = MagicMock()
        DataScienceModel.sync = MagicMock()
        OCIDataScienceModel.create = MagicMock()

        ds_model = DataScienceModel()
        os_path = "oci://aqua-bkt@aqua-ns/prefix/path"
        hf_model = "oracle/aqua-1t-mega-model"
        ds_freeform_tags = {
            "OCI_AQUA": "ACTIVE",
            "license": "aqua-license",
            "organization": "oracle",
            "task": "text-generation",
        }
        ds_model = (
            ds_model.with_compartment_id("test_model_compartment_id")
            .with_project_id("test_project_id")
            .with_display_name(hf_model)
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
        reload(ads.aqua.model)
        app = AquaModelApp()
        with tempfile.TemporaryDirectory() as tmpdir:
            model: DataScienceModel = app.register(
                model="ocid1.datasciencemodel.xxx.xxxx.",
                os_path=os_path,
                local_dir=str(tmpdir),
            )
            mock_snapshot_download.assert_called_with(
                repo_id=hf_model,
                local_dir=f"{str(tmpdir)}/{hf_model}",
                local_dir_use_symlinks=False,
            )
            mock_subprocess.assert_called_with(
                shlex.split(
                    f"oci os object bulk-upload --src-dir {str(tmpdir)}/{hf_model} --prefix prefix/path/{hf_model}/ -bn aqua-bkt -ns aqua-ns --auth api_key --profile DEFAULT"
                )
            )
            assert model.freeform_tags == {
                "aqua_custom_base_model": "true",
                **ds_freeform_tags,
            }
            expected_metadata = [
                {
                    "key": "evaluation-container",
                    "value": "odsc-llm-evaluate",
                    "description": "",
                    "category": "Other",
                },
                {
                    "key": "deployment-container",
                    "value": "odsc-tgi-serving",
                    "description": "",
                    "category": "Other",
                },
                {
                    "key": "modelDescription",
                    "value": "true",
                    "description": "model by reference flag",
                    "category": "Other",
                },
                {
                    "key": "artifact_location",
                    "value": f"{os_path}/{hf_model}/",
                    "description": "artifact location",
                    "category": "Other",
                },
            ]
            for item in expected_metadata:
                assert model.custom_metadata_list[item["key"]].to_dict() == item
            assert model.version_id != ds_model.version_id

    @patch("huggingface_hub.snapshot_download")
    @patch("subprocess.check_call")
    def test_import_any_hf_model_no_containers_specified(
        self,
        mock_subprocess,
        mock_snapshot_download,
    ):
        ObjectStorageDetails.is_bucket_versioned = MagicMock(return_value=True)
        ads.common.oci_datascience.OCIDataScienceMixin.init_client = MagicMock()
        huggingface_hub.HfApi.model_info = MagicMock(return_value={})
        DataScienceModel.upload_artifact = MagicMock()
        DataScienceModel.sync = MagicMock()
        OCIDataScienceModel.create = MagicMock()

        ds_model = DataScienceModel()
        os_path = "oci://aqua-bkt@aqua-ns/prefix/path"
        hf_model = "oracle/aqua-1t-mega-model"
        ds_freeform_tags = {
            "OCI_AQUA": "ACTIVE",
            "license": "aqua-license",
            "organization": "oracle",
            "task": "text-generation",
        }

        reload(ads.aqua.model)
        app = AquaModelApp()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                model: DataScienceModel = app.register(
                    model=hf_model,
                    os_path=os_path,
                    local_dir=str(tmpdir),
                )

    @patch("huggingface_hub.snapshot_download")
    @patch("subprocess.check_call")
    def test_import_any_hf_model_custom_container(
        self,
        mock_subprocess,
        mock_snapshot_download,
    ):
        hf_model = "oracle/aqua-1t-mega-model"
        ObjectStorageDetails.is_bucket_versioned = MagicMock(return_value=True)
        ads.common.oci_datascience.OCIDataScienceMixin.init_client = MagicMock()
        huggingface_hub.HfApi.model_info = MagicMock(
            return_value=huggingface_hub.hf_api.ModelInfo(
                **{
                    "id": hf_model,
                    "license": "aqua-license",
                    "author": "oracle",
                    "pipeline_tag": "text-generation",
                    "private": False,
                    "downloads": 100,
                    "likes": 10,
                    "tags": ["text-generation"],
                }
            )
        )
        DataScienceModel.upload_artifact = MagicMock()
        DataScienceModel.sync = MagicMock()
        OCIDataScienceModel.create = MagicMock()

        os_path = "oci://aqua-bkt@aqua-ns/prefix/path"
        ds_freeform_tags = {
            "OCI_AQUA": "active",
            "organization": "oracle",
            "task": "text-generation",
        }

        reload(ads.aqua.model)
        app = AquaModelApp()
        with tempfile.TemporaryDirectory() as tmpdir:
            model: DataScienceModel = app.register(
                model=hf_model,
                os_path=os_path,
                local_dir=str(tmpdir),
                inference_container="iad.ocir.io/my/own/md-container",
                inference_container_type_smc=False,
                finetuning_container="iad.ocir.io/my/own/ft-container",
                finetuning_container_type_smc=False,
            )
            mock_snapshot_download.assert_called_with(
                repo_id=hf_model,
                local_dir=f"{str(tmpdir)}/{hf_model}",
                local_dir_use_symlinks=False,
            )
            mock_subprocess.assert_called_with(
                shlex.split(
                    f"oci os object bulk-upload --src-dir {str(tmpdir)}/{hf_model} --prefix prefix/path/{hf_model}/ -bn aqua-bkt -ns aqua-ns --auth api_key --profile DEFAULT"
                )
            )
            assert model.freeform_tags == {
                "aqua_custom_base_model": "true",
                "aqua_finetuning": "true",
                **ds_freeform_tags,
            }
            expected_metadata = [
                {
                    "key": "evaluation-container",
                    "value": "odsc-llm-evaluate",
                    "description": "Evaluation container mapping for SMC",
                    "category": "Other",
                },
                {
                    "key": "deployment-container",
                    "value": "iad.ocir.io/my/own/md-container",
                    "description": f"Inference container mapping for {hf_model}",
                    "category": "Other",
                },
                {
                    "key": "finetune-container",
                    "value": "iad.ocir.io/my/own/ft-container",
                    "description": f"Fine-tuning container mapping for {hf_model}",
                    "category": "Other",
                },
                {
                    "key": "modelDescription",
                    "value": "true",
                    "description": "model by reference flag",
                    "category": "Other",
                },
                {
                    "key": "artifact_location",
                    "value": f"{os_path}/{hf_model}/",
                    "description": "artifact location",
                    "category": "Other",
                },
            ]
            for item in expected_metadata:
                assert model.custom_metadata_list[item["key"]].to_dict() == item
            assert model.version_id is None

    @patch("huggingface_hub.snapshot_download")
    @patch("subprocess.check_call")
    def test_import_any_hf_model_smc_container(
        self,
        mock_subprocess,
        mock_snapshot_download,
    ):
        hf_model = "oracle/aqua-1t-mega-model"
        ObjectStorageDetails.is_bucket_versioned = MagicMock(return_value=True)
        ads.common.oci_datascience.OCIDataScienceMixin.init_client = MagicMock()
        huggingface_hub.HfApi.model_info = MagicMock(
            return_value=huggingface_hub.hf_api.ModelInfo(
                **{
                    "id": hf_model,
                    "license": "aqua-license",
                    "author": "oracle",
                    "pipeline_tag": "text-generation",
                    "private": False,
                    "downloads": 100,
                    "likes": 10,
                    "tags": ["text-generation"],
                }
            )
        )
        DataScienceModel.upload_artifact = MagicMock()
        DataScienceModel.sync = MagicMock()
        OCIDataScienceModel.create = MagicMock()

        os_path = "oci://aqua-bkt@aqua-ns/prefix/path"
        ds_freeform_tags = {
            "OCI_AQUA": "active",
            "organization": "oracle",
            "task": "text-generation",
        }

        reload(ads.aqua.model)
        app = AquaModelApp()
        with tempfile.TemporaryDirectory() as tmpdir:
            model: DataScienceModel = app.register(
                model=hf_model,
                os_path=os_path,
                local_dir=str(tmpdir),
                inference_container="dsmc://md-container",
                inference_container_type_smc=False,
                finetuning_container="dsmc://ft-container",
                finetuning_container_type_smc=False,
            )
            mock_snapshot_download.assert_called_with(
                repo_id=hf_model,
                local_dir=f"{str(tmpdir)}/{hf_model}",
                local_dir_use_symlinks=False,
            )
            mock_subprocess.assert_called_with(
                shlex.split(
                    f"oci os object bulk-upload --src-dir {str(tmpdir)}/{hf_model} --prefix prefix/path/{hf_model}/ -bn aqua-bkt -ns aqua-ns --auth api_key --profile DEFAULT"
                )
            )
            assert model.freeform_tags == {
                "aqua_custom_base_model": "true",
                "aqua_finetuning": "true",
                **ds_freeform_tags,
            }
            expected_metadata = [
                {
                    "key": "evaluation-container",
                    "value": "odsc-llm-evaluate",
                    "description": "Evaluation container mapping for SMC",
                    "category": "Other",
                },
                {
                    "key": "deployment-container",
                    "value": "dsmc://md-container",
                    "description": f"Inference container mapping for {hf_model}",
                    "category": "Other",
                },
                {
                    "key": "finetune-container",
                    "value": "dsmc://ft-container",
                    "description": f"Fine-tuning container mapping for {hf_model}",
                    "category": "Other",
                },
                {
                    "key": "modelDescription",
                    "value": "true",
                    "description": "model by reference flag",
                    "category": "Other",
                },
                {
                    "key": "artifact_location",
                    "value": f"{os_path}/{hf_model}/",
                    "description": "artifact location",
                    "category": "Other",
                },
            ]
            for item in expected_metadata:
                assert model.custom_metadata_list[item["key"]].to_dict() == item
            assert model.version_id is None

    @patch("ads.aqua.model.read_file")
    @patch("ads.aqua.model.get_artifact_path")
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

        assert len(results) == 1

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
