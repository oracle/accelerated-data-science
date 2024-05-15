#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest
from importlib import reload
from unittest.mock import MagicMock, patch

from notebook.base.handlers import IPythonHandler

import ads.aqua
import ads.config
from ads.aqua.extension.deployment_handler import (
    AquaDeploymentHandler,
    AquaDeploymentInferenceHandler,
    AquaDeploymentParamsHandler,
)


class TestDataset:
    USER_COMPARTMENT_ID = "ocid1.compartment.oc1..<USER_COMPARTMENT_OCID>"
    USER_PROJECT_ID = "ocid1.datascienceproject.oc1.iad.<USER_PROJECT_OCID>"
    INSTANCE_SHAPE = "VM.GPU.A10.1"
    deployment_request = {
        "model_id": "ocid1.datasciencemodel.oc1.iad.<OCID>",
        "instance_shape": "VM.GPU.A10.1",
        "display_name": "test-deployment-name",
    }
    inference_request = {
        "prompt": "What is 1+1?",
        "endpoint": "https://modeldeployment.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.<region>.<MD_OCID>/predict",
        "model_params": {
            "model": "odsc-llm",
            "max_tokens": 500,
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 10,
        },
    }


class TestAquaDeploymentHandler(unittest.TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.deployment_handler = AquaDeploymentHandler(MagicMock(), MagicMock())
        self.deployment_handler.request = MagicMock()
        self.deployment_handler.finish = MagicMock()

    @classmethod
    def setUpClass(cls):
        os.environ["PROJECT_COMPARTMENT_OCID"] = TestDataset.USER_COMPARTMENT_ID
        os.environ["PROJECT_OCID"] = TestDataset.USER_PROJECT_ID
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.extension.deployment_handler)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("PROJECT_COMPARTMENT_OCID", None)
        os.environ.pop("PROJECT_OCID", None)
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.extension.deployment_handler)

    @patch("ads.aqua.modeldeployment.AquaDeploymentApp.get_deployment_config")
    def test_get_deployment_config(self, mock_get_deployment_config):
        """Test get method to return deployment config"""
        self.deployment_handler.request.path = "aqua/deployments/config"
        self.deployment_handler.get(id="mock-model-id")
        mock_get_deployment_config.assert_called()

    @unittest.skip("fix this test after exception handler is updated.")
    @patch("ads.aqua.extension.base_handler.AquaAPIhandler.write_error")
    def test_get_deployment_config_without_id(self, mock_error):
        """Test get method to return deployment config"""
        # todo: exception handler needs to be revisited
        self.deployment_handler.request.path = "aqua/deployments/config"
        mock_error.return_value = MagicMock(status=400)
        result = self.deployment_handler.get(id="")
        mock_error.assert_called_once()
        assert result["status"] == 400

    @patch("ads.aqua.modeldeployment.AquaDeploymentApp.get")
    def test_get_deployment(self, mock_get):
        """Test get method to return deployment information."""
        self.deployment_handler.request.path = "aqua/deployments"
        self.deployment_handler.get(id="mock-model-id")
        mock_get.assert_called()

    @patch("ads.aqua.modeldeployment.AquaDeploymentApp.list")
    def test_list_deployment(self, mock_list):
        """Test get method to return a list of model deployments."""
        self.deployment_handler.request.path = "aqua/deployments"
        self.deployment_handler.get(id="")
        mock_list.assert_called_with(
            compartment_id=TestDataset.USER_COMPARTMENT_ID, project_id=None
        )

    @patch("ads.aqua.modeldeployment.AquaDeploymentApp.create")
    def test_post(self, mock_create):
        """Test post method to create a model deployment."""
        self.deployment_handler.get_json_body = MagicMock(
            return_value=TestDataset.deployment_request
        )

        self.deployment_handler.post()
        mock_create.assert_called_with(
            compartment_id=TestDataset.USER_COMPARTMENT_ID,
            project_id=TestDataset.USER_PROJECT_ID,
            model_id=TestDataset.deployment_request["model_id"],
            display_name=TestDataset.deployment_request["display_name"],
            description=None,
            instance_count=None,
            instance_shape=TestDataset.deployment_request["instance_shape"],
            log_group_id=None,
            access_log_id=None,
            predict_log_id=None,
            bandwidth_mbps=None,
            web_concurrency=None,
            server_port=None,
            health_check_port=None,
            env_var=None,
            container_image=None,
        )


class AquaDeploymentParamsHandlerTestCase(unittest.TestCase):
    default_params = ["--seed 42", "--trust-remote-code"]

    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.test_instance = AquaDeploymentParamsHandler(MagicMock(), MagicMock())

    @patch("notebook.base.handlers.APIHandler.finish")
    @patch(
        "ads.aqua.modeldeployment.deployment.AquaDeploymentApp.get_deployment_default_params"
    )
    def test_get_deployment_default_params(
        self, mock_get_deployment_default_params, mock_finish
    ):
        """Test to check the handler get method to return default params for model deployment."""

        mock_get_deployment_default_params.return_value = self.default_params
        mock_finish.side_effect = lambda x: x

        args = {"instance_shape": TestDataset.INSTANCE_SHAPE}
        self.test_instance.get_argument = MagicMock(
            side_effect=lambda arg, default=None: args.get(arg, default)
        )
        result = self.test_instance.get(model_id="test_model_id")
        self.assertCountEqual(result["data"], self.default_params)

        mock_get_deployment_default_params.assert_called_with(
            model_id="test_model_id", instance_shape=TestDataset.INSTANCE_SHAPE
        )

    @patch("notebook.base.handlers.APIHandler.finish")
    @patch(
        "ads.aqua.modeldeployment.deployment.AquaDeploymentApp.validate_deployment_params"
    )
    def test_validate_deployment_params(
        self, mock_validate_deployment_params, mock_finish
    ):
        mock_validate_deployment_params.return_value = dict(valid=True)
        mock_finish.side_effect = lambda x: x

        self.test_instance.get_json_body = MagicMock(
            return_value=dict(model_id="test-model-id", params=self.default_params)
        )
        result = self.test_instance.post()
        assert result["valid"] is True
        mock_validate_deployment_params.assert_called_with(
            model_id="test-model-id", params=self.default_params
        )


class TestAquaDeploymentInferenceHandler(unittest.TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.inference_handler = AquaDeploymentInferenceHandler(
            MagicMock(), MagicMock()
        )
        self.inference_handler.request = MagicMock()
        self.inference_handler.finish = MagicMock()

    @patch("ads.aqua.modeldeployment.MDInferenceResponse.get_model_deployment_response")
    def test_post(self, mock_get_model_deployment_response):
        """Test post method to return model deployment response."""
        self.inference_handler.get_json_body = MagicMock(
            return_value=TestDataset.inference_request
        )
        self.inference_handler.post()
        mock_get_model_deployment_response.assert_called_with(
            TestDataset.inference_request["endpoint"]
        )
