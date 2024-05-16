#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import json
import pytest
from parameterized import parameterized
from unittest import TestCase
from unittest.mock import MagicMock, PropertyMock
from mock import patch
from dataclasses import asdict
from importlib import reload

import ads.aqua
import ads.aqua.finetuning.finetuning
from ads.aqua.model.entities import AquaFineTuneModel
import ads.config
from ads.aqua.app import AquaApp
from ads.aqua.finetuning import AquaFineTuningApp
from ads.aqua.finetuning.constants import FineTuneCustomMetadata
from ads.aqua.finetuning.entities import AquaFineTuningParams
from ads.jobs.ads_job import Job
from ads.model.datascience_model import DataScienceModel
from ads.model.model_metadata import ModelCustomMetadata
from ads.aqua.common.errors import AquaValueError


class FineTuningTestCase(TestCase):
    SERVICE_COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"

    def setUp(self):
        self.app = AquaFineTuningApp()

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))
        os.environ["ODSC_MODEL_COMPARTMENT_OCID"] = cls.SERVICE_COMPARTMENT_ID
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.finetuning.finetuning)

    @classmethod
    def tearDownClass(cls):
        cls.curr_dir = None
        os.environ.pop("ODSC_MODEL_COMPARTMENT_OCID", None)
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.finetuning.finetuning)

    @patch.object(Job, "run")
    @patch("ads.jobs.ads_job.Job.name", new_callable=PropertyMock)
    @patch("ads.jobs.ads_job.Job.id", new_callable=PropertyMock)
    @patch.object(Job, "create")
    @patch("ads.aqua.finetuning.finetuning.get_container_image")
    @patch.object(AquaFineTuningApp, "get_finetuning_config")
    @patch.object(AquaApp, "create_model_catalog")
    @patch.object(AquaApp, "create_model_version_set")
    @patch.object(AquaApp, "get_source")
    def test_create_fine_tuning(
        self,
        mock_get_source,
        mock_mvs_create,
        mock_ds_model_create,
        mock_get_finetuning_config,
        mock_get_container_image,
        mock_job_create,
        mock_job_id,
        mock_job_name,
        mock_job_run,
    ):
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            key=FineTuneCustomMetadata.SERVICE_MODEL_ARTIFACT_LOCATION,
            value="test_service_model_artifact_location",
        )
        custom_metadata_list.add(
            key=FineTuneCustomMetadata.SERVICE_MODEL_DEPLOYMENT_CONTAINER,
            value="test_service_model_deployment_container",
        )
        custom_metadata_list.add(
            key=FineTuneCustomMetadata.SERVICE_MODEL_FINE_TUNE_CONTAINER,
            value="test_service_model_fine_tune_container",
        )

        ft_source = MagicMock()
        ft_source.id = "test_ft_source_id"
        ft_source.compartment_id = self.SERVICE_COMPARTMENT_ID
        ft_source.display_name = "test_ft_source_model"
        ft_source.custom_metadata_list = custom_metadata_list
        mock_get_source.return_value = ft_source

        mock_mvs_create.return_value = ("test_experiment_id", "test_experiment_name")

        ft_model = MagicMock()
        ft_model.id = "test_ft_model_id"
        ft_model.display_name = "test_ft_model_name"
        ft_model.time_created = "test_time_created"
        mock_ds_model_create.return_value = ft_model

        mock_get_finetuning_config.return_value = {
            "shape": {
                "VM.GPU.A10.1": {"batch_size": 1, "replica": 1},
            }
        }
        mock_get_container_image.return_value = "test_container_image"

        mock_job_id.return_value = "test_ft_job_id"
        mock_job_name.return_value = "test_ft_job_name"

        ft_job_run = MagicMock()
        ft_job_run.id = "test_ft_job_run_id"
        ft_job_run.lifecycle_details = "Job run artifact execution in progress."
        ft_job_run.lifecycle_state = "IN_PROGRESS"
        mock_job_run.return_value = ft_job_run

        self.app.ds_client.update_model = MagicMock()
        self.app.ds_client.update_model_provenance = MagicMock()

        create_aqua_ft_details = dict(
            ft_source_id="ocid1.datasciencemodel.oc1.iad.<OCID>",
            ft_name="test_ft_name",
            dataset_path="oci://ds_bucket@namespace/prefix/dataset.jsonl",
            report_path="oci://report_bucket@namespace/prefix/",
            ft_parameters={
                "epochs": 1,
                "learning_rate": 0.02,
                "lora_target_linear": False,
            },
            shape_name="VM.GPU.A10.1",
            replica=1,
            validation_set_size=0.2,
            block_storage_size=1,
            experiment_name="test_experiment_name",
        )

        aqua_ft_summary = self.app.create(**create_aqua_ft_details)

        assert asdict(aqua_ft_summary) == {
            "console_url": f"https://cloud.oracle.com/data-science/models/{ft_model.id}?region={self.app.region}",
            "experiment": {
                "id": f"{mock_mvs_create.return_value[0]}",
                "name": f"{mock_mvs_create.return_value[1]}",
                "url": f"https://cloud.oracle.com/data-science/model-version-sets/{mock_mvs_create.return_value[0]}?region={self.app.region}",
            },
            "id": f"{ft_model.id}",
            "job": {
                "id": f"{mock_job_id.return_value}",
                "name": f"{mock_job_name.return_value}",
                "url": f"https://cloud.oracle.com/data-science/jobs/{mock_job_id.return_value}?region={self.app.region}",
            },
            "lifecycle_details": f"{ft_job_run.lifecycle_details}",
            "lifecycle_state": f"{ft_job_run.lifecycle_state}",
            "name": f"{ft_model.display_name}",
            "parameters": {
                "epochs": 1,
                "learning_rate": 0.02,
                "sample_packing": "auto",
                "batch_size": 1,
                "lora_target_linear": False,
            },
            "source": {
                "id": f"{ft_source.id}",
                "name": f"{ft_source.display_name}",
                "url": f"https://cloud.oracle.com/data-science/models/{ft_source.id}?region={self.app.region}",
            },
            "tags": {
                "aqua_finetuning": "aqua_finetuning",
                "finetuning_experiment_id": f"{mock_mvs_create.return_value[0]}",
                "finetuning_job_id": f"{mock_job_id.return_value}",
                "finetuning_source": f"{ft_source.id}",
            },
            "time_created": f"{ft_model.time_created}",
        }

    def test_exit_code_message(self):
        message = AquaFineTuneModel(
            model=DataScienceModel()
        )._extract_job_lifecycle_details(
            "Job run artifact execution failed with exit code 100."
        )
        print(message)
        self.assertEqual(
            message,
            "CUDA out of memory. GPU does not have enough memory to train the model. "
            "Please use a shape with more GPU memory. (exit code 100)",
        )
        # No change should be made for exit code 1
        message = AquaFineTuneModel(
            model=DataScienceModel()
        )._extract_job_lifecycle_details(
            "Job run artifact execution failed with exit code 1."
        )
        print(message)
        self.assertEqual(message, "Job run artifact execution failed with exit code 1.")

        # No change should be made for other status.
        message = AquaFineTuneModel(
            model=DataScienceModel()
        )._extract_job_lifecycle_details(
            "Job run could not be started due to service issues. Please try again later."
        )
        print(message)
        self.assertEqual(
            message,
            "Job run could not be started due to service issues. Please try again later.",
        )

    def test_build_oci_launch_cmd(self):
        dataset_path = "oci://ds_bucket@namespace/prefix/dataset.jsonl"
        report_path = "oci://report_bucket@namespace/prefix/"
        val_set_size = 0.1
        parameters = AquaFineTuningParams(
            batch_size=1,
            epochs=1,
            sample_packing="True",
            learning_rate=0.01,
            sequence_len=2,
            lora_target_modules=["q_proj", "k_proj"],
        )
        finetuning_params = "--trust_remote_code True"
        oci_launch_cmd = self.app._build_oci_launch_cmd(
            dataset_path=dataset_path,
            report_path=report_path,
            val_set_size=val_set_size,
            parameters=parameters,
            finetuning_params=finetuning_params,
        )

        assert (
            oci_launch_cmd
            == f"--training_data {dataset_path} --output_dir {report_path} --val_set_size {val_set_size} --num_epochs {parameters.epochs} --learning_rate {parameters.learning_rate} --sample_packing {parameters.sample_packing} --micro_batch_size {parameters.batch_size} --sequence_len {parameters.sequence_len} --lora_target_modules q_proj,k_proj {finetuning_params}"
        )

    def test_get_finetuning_default_params(self):
        """Test for fetching finetuning config params for a given model."""

        params_dict = {
            "params": {
                "batch_size": 1,
                "sequence_len": 2048,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "learning_rate": 0.0002,
                "lora_r": 32,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "lora_target_modules": ["q_proj", "k_proj"],
            }
        }
        config_json = os.path.join(self.curr_dir, "test_data/finetuning/ft_config.json")
        with open(config_json, "r") as _file:
            config = json.load(_file)

        self.app.get_finetuning_config = MagicMock(return_value=config)
        result = self.app.get_finetuning_default_params(model_id="test_model_id")
        assert result == params_dict

        # check when config json is not available
        self.app.get_finetuning_config = MagicMock(return_value={})
        result = self.app.get_finetuning_default_params(model_id="test_model_id")
        assert result == {}

    @parameterized.expand(
        [
            (
                [
                    "--batch_size 1",
                    "--sequence_len 2048",
                    "--sample_packing true",
                    "--pad_to_sequence_len true",
                    "--learning_rate 0.0002",
                    "--lora_r 32",
                    "--lora_alpha 16",
                    "--lora_dropout 0.05",
                    "--lora_target_linear true",
                    "--lora_target_modules q_proj,k_proj",
                ],
                True,
            ),
            (
                [
                    "--micro_batch_size 1",
                    "--max_sequence_len 2048",
                    "--flash_attention true",
                    "--pad_to_sequence_len true",
                    "--lr_scheduler cosine",
                ],
                False,
            ),
        ]
    )
    def test_validate_finetuning_params(self, params, is_valid):
        """Test for checking if overridden fine-tuning params are valid."""
        if is_valid:
            result = self.app.validate_finetuning_params(params)
            assert result["valid"] is True
        else:
            with pytest.raises(AquaValueError):
                self.app.validate_finetuning_params(params)
