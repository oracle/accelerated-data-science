#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest import TestCase
from unittest.mock import MagicMock
from mock import patch

from notebook.base.handlers import IPythonHandler, APIHandler
from ads.aqua.extension.finetune_handler import (
    AquaFineTuneHandler,
    AquaFineTuneParamsHandler,
)
from ads.aqua.finetune import AquaFineTuningApp, CreateFineTuningDetails


class TestDataset:
    mock_valid_input = dict(
        ft_source_id="ocid1.datasciencemodel.oc1.iad.<OCID>",
        ft_name="test_ft_name",
        dataset_path="oci://ds_bucket@namespace/prefix/dataset.jsonl",
        report_path="oci://report_bucket@namespace/prefix/",
        ft_parameters={"epochs": 1, "learning_rate": 0.02},
        shape_name="VM.GPU.A10.1",
        replica=1,
        validation_set_size=0.2,
        block_storage_size=1,
        experiment_name="test_experiment_name",
    )

    mock_finetuning_config = {
        "shape": {
            "VM.GPU.A10.1": {"batch_size": 1, "replica": 1},
        }
    }


class FineTuningHandlerTestCase(TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.test_instance = AquaFineTuneHandler(MagicMock(), MagicMock())
        self.test_instance.request = MagicMock()
        self.test_instance.finish = MagicMock()

    @patch.object(AquaFineTuneHandler, "get_finetuning_config")
    @patch("ads.aqua.extension.finetune_handler.urlparse")
    def test_get(self, mock_urlparse, mock_get_finetuning_config):
        request_path = MagicMock(path="aqua/finetuning/config")
        mock_urlparse.return_value = request_path

        mock_get_finetuning_config.return_value = TestDataset.mock_finetuning_config

        fineruning_config = self.test_instance.get(id="test_model_id")
        mock_urlparse.assert_called()
        mock_get_finetuning_config.assert_called_with("test_model_id")
        assert fineruning_config == TestDataset.mock_finetuning_config

    @patch.object(AquaFineTuningApp, "create")
    def test_post(self, mock_create):
        self.test_instance.get_json_body = MagicMock(
            return_value=TestDataset.mock_valid_input
        )
        self.test_instance.post()

        self.test_instance.finish.assert_called_with(mock_create.return_value)
        mock_create.assert_called_with(
            CreateFineTuningDetails(**TestDataset.mock_valid_input)
        )

    @patch.object(AquaFineTuningApp, "get_finetuning_config")
    def test_get_finetuning_config(self, mock_get_finetuning_config):
        mock_get_finetuning_config.return_value = TestDataset.mock_finetuning_config

        self.test_instance.get_finetuning_config(model_id="test_model_id")

        self.test_instance.finish.assert_called_with(TestDataset.mock_finetuning_config)
        mock_get_finetuning_config.assert_called_with(model_id="test_model_id")


class AquaFineTuneParamsHandlerTestCase(TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.test_instance = AquaFineTuneParamsHandler(MagicMock(), MagicMock())

    @patch.object(APIHandler, "finish")
    @patch.object(AquaFineTuningApp, "get_finetuning_default_params")
    def test_get_finetuning_default_params(
        self, mock_get_finetuning_default_params, mock_finish
    ):
        default_params = [
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
        ]

        mock_get_finetuning_default_params.return_value = default_params
        mock_finish.side_effect = lambda x: x

        result = self.test_instance.get(model_id="test_model_id")

        assert result["data"] == default_params
        mock_get_finetuning_default_params.assert_called_with(model_id="test_model_id")
