#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest
from importlib import reload
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from ads.aqua.common.enums import PredictEndpoints
from notebook.base.handlers import IPythonHandler
from parameterized import parameterized
import openai

import ads.aqua
import ads.config
from ads.aqua.extension.deployment_handler import (
    AquaDeploymentHandler,
    AquaDeploymentParamsHandler,
    AquaDeploymentStreamingInferenceHandler,
    AquaModelListHandler,
)
from ads.aqua.modeldeployment.entities import AquaDeploymentDetail


class TestDataset:
    USER_COMPARTMENT_ID = "ocid1.compartment.oc1..<USER_COMPARTMENT_OCID>"
    USER_PROJECT_ID = "ocid1.datascienceproject.oc1.iad.<USER_PROJECT_OCID>"
    INSTANCE_SHAPE = "VM.GPU.A10.1"
    deployment_request = {
        "model_id": "ocid1.datasciencemodel.oc1.iad.<OCID>",
        "instance_shape": "VM.GPU.A10.1",
        "display_name": "test-deployment-name",
        "freeform_tags": {"ftag1": "fvalue1", "ftag2": "fvalue2"},
        "defined_tags": {"dtag1": "dvalue1", "dtag2": "dvalue2"},
        "project_id": USER_PROJECT_ID,
        "compartment_id": USER_COMPARTMENT_ID,
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

    @patch("ads.aqua.modeldeployment.AquaDeploymentApp.recommend_shape")
    def test_get_recommend_shape(self, mock_recommend_shape):
        """Test get method to return deployment config"""
        self.deployment_handler.request.path = "aqua/deployments/recommend_shapes"
        self.deployment_handler.get(id="mock-model-id")
        mock_recommend_shape.assert_called()

    @unittest.skip("fix this test after exception handler is updated.")
    @patch("ads.aqua.extension.base_handler.AquaAPIhandler.write_error")
    def test_get_recommend_shape_without_id(self, mock_error):
        """Test get method to return deployment config"""
        # todo: exception handler needs to be revisited
        self.deployment_handler.request.path = "aqua/deployments/recommend_shape"
        mock_error.return_value = MagicMock(status=400)
        result = self.deployment_handler.get(id="")
        mock_error.assert_called_once()
        assert result["status"] == 400

    @patch(
        "ads.aqua.modeldeployment.AquaDeploymentApp.get_multimodel_deployment_config"
    )
    def test_get_multimodel_deployment_config(
        self, mock_get_multimodel_deployment_config
    ):
        """Test get method to return multi model deployment config"""
        self.deployment_handler.request.path = "aqua/deployments/config"
        self.deployment_handler.get(id="mock-model-id-one,mock-model-id-two")
        mock_get_multimodel_deployment_config.assert_called_with(
            model_ids=["mock-model-id-one", "mock-model-id-two"],
            primary_model_id=None,
            compartment_id=TestDataset.USER_COMPARTMENT_ID,
        )

    @patch("ads.aqua.modeldeployment.AquaDeploymentApp.get")
    def test_get_deployment(self, mock_get):
        """Test get method to return deployment information."""
        self.deployment_handler.request.path = "aqua/deployments"
        self.deployment_handler.get(id="mock-model-id")
        mock_get.assert_called()

    @patch("ads.aqua.modeldeployment.AquaDeploymentApp.delete")
    def test_delete_deployment(self, mock_delete):
        self.deployment_handler.request.path = "aqua/deployments"
        self.deployment_handler.delete("mock-model-id")
        mock_delete.assert_called()

    @patch("ads.aqua.modeldeployment.AquaDeploymentApp.activate")
    def test_activate_deployment(self, mock_activate):
        self.deployment_handler.request.path = (
            "aqua/deployments/ocid1.datasciencemodeldeployment.oc1.iad.xxx/activate"
        )
        mock_activate.return_value = {"lifecycle_state": "UPDATING"}
        self.deployment_handler.put()
        mock_activate.assert_called()

    @patch("ads.aqua.modeldeployment.AquaDeploymentApp.deactivate")
    def test_deactivate_deployment(self, mock_deactivate):
        self.deployment_handler.request.path = (
            "aqua/deployments/ocid1.datasciencemodeldeployment.oc1.iad.xxx/deactivate"
        )
        mock_deactivate.return_value = {"lifecycle_state": "UPDATING"}
        self.deployment_handler.put()
        mock_deactivate.assert_called()

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
            instance_shape=TestDataset.deployment_request["instance_shape"],
            freeform_tags=TestDataset.deployment_request["freeform_tags"],
            defined_tags=TestDataset.deployment_request["defined_tags"],
        )


class AquaDeploymentParamsHandlerTestCase(unittest.TestCase):
    default_params = ["--seed 42", "--trust-remote-code"]

    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.test_instance = AquaDeploymentParamsHandler(MagicMock(), MagicMock())

    @patch("notebook.base.handlers.APIHandler.finish")
    @patch("ads.aqua.modeldeployment.AquaDeploymentApp.get_deployment_default_params")
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
            model_id="test_model_id",
            instance_shape=TestDataset.INSTANCE_SHAPE,
            gpu_count=None,
        )

    @parameterized.expand(
        [
            None,
            "container-family-name",
        ]
    )
    @patch("notebook.base.handlers.APIHandler.finish")
    @patch("ads.aqua.modeldeployment.AquaDeploymentApp.validate_deployment_params")
    def test_validate_deployment_params(
        self, container_family_value, mock_validate_deployment_params, mock_finish
    ):
        mock_validate_deployment_params.return_value = dict(valid=True)
        mock_finish.side_effect = lambda x: x

        self.test_instance.get_json_body = MagicMock(
            return_value=dict(
                model_id="test-model-id",
                params=self.default_params,
                container_family=container_family_value,
            )
        )
        result = self.test_instance.post()
        assert result["valid"] is True
        mock_validate_deployment_params.assert_called_with(
            model_id="test-model-id",
            params=self.default_params,
            container_family=container_family_value,
        )


class TestAquaDeploymentStreamingInferenceHandler(unittest.TestCase):

    EXPECTED_OCID = "ocid1.compartment.oc1..aaaaaaaaser65kfcfht7iddoioa4s6xos3vi53d3i7bi3czjkqyluawp2itq"

    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.handler = AquaDeploymentStreamingInferenceHandler(MagicMock(), MagicMock())
        self.handler.request = MagicMock()
        self.handler.set_header = MagicMock()
        self.handler.write = MagicMock()
        self.handler.flush = MagicMock()
        self.handler.finish = MagicMock()

    @patch.object(
        AquaDeploymentStreamingInferenceHandler, "_get_model_deployment_response"
    )
    def test_post(self, mock_get_model_deployment_response):
        """Test post method to return model deployment response."""
        mock_response_gen = iter(["chunk1", "chunk2"])

        mock_get_model_deployment_response.return_value = mock_response_gen

        self.handler.get_json_body = MagicMock(
            return_value={"prompt": "Hello", "model": "some-model"}
        )
        self.handler.request.headers = MagicMock()
        self.handler.request.headers.get.return_value = "test-route"

        self.handler.post("mock-deployment-id")

        mock_get_model_deployment_response.assert_called_with(
            "mock-deployment-id",
            {"prompt": "Hello", "model": "some-model"}
        )
        self.handler.write.assert_any_call("chunk1")
        self.handler.write.assert_any_call("chunk2")
        self.handler.finish.assert_called_once()

    def test_extract_text_from_choice_dict_delta_content(self):
        """Test dict choice with delta.content."""
        choice = {"delta": {"content": "hello"}}
        result = self.handler._extract_text_from_choice(choice)
        self.assertEqual(result, "hello")

    def test_extract_text_from_choice_dict_delta_text(self):
        """Test dict choice with delta.text fallback."""
        choice = {"delta": {"text": "world"}}
        result = self.handler._extract_text_from_choice(choice)
        self.assertEqual(result, "world")

    def test_extract_text_from_choice_dict_message_content(self):
        """Test dict choice with message.content."""
        choice = {"message": {"content": "foo"}}
        result = self.handler._extract_text_from_choice(choice)
        self.assertEqual(result, "foo")

    def test_extract_text_from_choice_dict_top_level_text(self):
        """Test dict choice with top-level text."""
        choice = {"text": "bar"}
        result = self.handler._extract_text_from_choice(choice)
        self.assertEqual(result, "bar")

    def test_extract_text_from_choice_object_delta_content(self):
        """Test object choice with delta.content attribute."""
        choice = MagicMock()
        choice.delta = MagicMock(content="obj-content", text=None)
        result = self.handler._extract_text_from_choice(choice)
        self.assertEqual(result, "obj-content")

    def test_extract_text_from_choice_object_message_str(self):
        """Test object choice with message as string."""
        choice = MagicMock()
        choice.delta = None  # No delta, so message takes precedence
        choice.message = "direct-string"
        result = self.handler._extract_text_from_choice(choice)
        self.assertEqual(result, "direct-string")

    def test_extract_text_from_choice_none_return(self):
        """Test choice with no text content returns None."""
        choice = {}
        result = self.handler._extract_text_from_choice(choice)
        self.assertIsNone(result)

    def test_extract_text_from_chunk_dict_with_choices(self):
        """Test chunk dict with choices list."""
        chunk = {"choices": [{"delta": {"content": "chunk-text"}}]}
        result = self.handler._extract_text_from_chunk(chunk)
        self.assertEqual(result, "chunk-text")

    def test_extract_text_from_chunk_dict_top_level_content(self):
        """Test chunk dict with top-level content (no choices)."""
        chunk = {"content": "direct-content"}
        result = self.handler._extract_text_from_chunk(chunk)
        self.assertEqual(result, "direct-content")

    def test_extract_text_from_chunk_object_choices(self):
        """Test object chunk with choices attribute."""
        chunk = MagicMock()
        chunk.choices = [{"message": {"content": "obj-chunk"}}]
        result = self.handler._extract_text_from_chunk(chunk)
        self.assertEqual(result, "obj-chunk")

    def test_extract_text_from_chunk_empty(self):
        """Test empty/None chunk returns None."""
        result = self.handler._extract_text_from_chunk({})
        self.assertIsNone(result)
        result = self.handler._extract_text_from_chunk(None)
        self.assertIsNone(result)




class AquaModelListHandlerTestCase(unittest.TestCase):
    default_params = {
        "data": [{"id": "id", "object": "object", "owned_by": "openAI", "created": 124}]
    }

    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.aqua_model_list_handler = AquaModelListHandler(MagicMock(), MagicMock())
        self.aqua_model_list_handler._headers = MagicMock()

    @patch("ads.aqua.modeldeployment.AquaDeploymentApp.get")
    @patch("notebook.base.handlers.APIHandler.finish")
    def test_get_model_list(self, mock_get, mock_finish):
        """Test to check the handler get method to return model list."""

        mock_get.return_value = MagicMock(id="test_model_id")
        mock_finish.side_effect = lambda x: x
        result = self.aqua_model_list_handler.get(model_id="test_model_id")
        mock_get.assert_called()
