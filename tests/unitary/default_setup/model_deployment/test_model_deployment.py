#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
import pytest
import numpy as np
from unittest.mock import MagicMock, Mock, patch

from ads.common import auth as authutil
from ads.common import oci_client
from ads.model.deployment.model_deployment import (
    ModelDeployment,
    ModelDeploymentProperties,
)


class ModelDeploymentTestCase(unittest.TestCase):
    MODEL_ID = "<MODEL_OCID>"
    with patch.object(oci_client, "OCIClientFactory"):
        test_model_deployment = ModelDeployment(
            model_deployment_id="test_model_deployment_id", properties={}
        )

    @patch("requests.post")
    def test_predict(self, mock_post):
        """Ensures predict model passes with valid input parameters."""
        mock_post.return_value = Mock(
            status_code=200, json=lambda: {"result": "result"}
        )
        with patch.object(authutil, "default_signer") as mock_auth:
            auth = MagicMock()
            auth["signer"] = MagicMock()
            mock_auth.return_value = auth
            test_result = self.test_model_deployment.predict(json_input="test")
            mock_post.assert_called_with(
                f"{self.test_model_deployment.url}/predict",
                json="test",
                headers={"Content-Type": "application/json"},
                auth=mock_auth.return_value["signer"],
            )
            assert test_result == {"result": "result"}

        with pytest.raises(TypeError):
            self.test_model_deployment.predict(data=np.array([1, 2, 3]))

    @patch("requests.post")
    def test_predict_with_bytes(self, mock_post):
        """Ensures predict model passes with bytes input."""
        byte_data = b"[[1,2,3,4]]"
        with patch.object(authutil, "default_signer") as mock_auth:
            auth = MagicMock()
            auth["signer"] = MagicMock()
            mock_auth.return_value = auth
            self.test_model_deployment.predict(data=byte_data)
            mock_post.assert_called_with(
                f"{self.test_model_deployment.url}/predict",
                data=byte_data,
                auth=mock_auth.return_value["signer"],
                headers={"Content-Type": "application/octet-stream"},
            )

    @patch("requests.post")
    def test_predict_with_auto_serialize_data(self, mock_post):
        """Ensures predict model passes with valid input parameters."""
        mock_post.return_value = Mock(
            status_code=200, json=lambda: {"result": "result"}
        )
        with patch.object(authutil, "default_signer") as mock_auth:
            auth = MagicMock()
            auth["signer"] = MagicMock()
            mock_auth.return_value = auth
            test_result = self.test_model_deployment.predict(
                json_input="test", auto_serialize_data=True
            )
            mock_post.assert_called_with(
                f"{self.test_model_deployment.url}/predict",
                json={"data": "test", "data_type": "<class 'str'>"},
                headers={"Content-Type": "application/json"},
                auth=mock_auth.return_value["signer"],
            )
            assert test_result == {"result": "result"}


class ModelDeploymentPropertiesTestCase(unittest.TestCase):
    MODEL_ID = "<MODEL_OCID>"

    def assert_model_id(self, oci_model):
        """Checks if the model OCID is configured correctly."""
        self.assertIsNotNone(oci_model.model_deployment_configuration_details)
        self.assertEqual(
            oci_model.model_deployment_configuration_details.model_configuration_details.model_id,
            self.MODEL_ID,
        )

    def test_setting_model_deployment_with_model_id(self):
        """Tests setting model deployment with model OCID."""
        # User may pass in the model ID when initializing ModelDeploymentProperties
        properties = ModelDeploymentProperties(model_id=self.MODEL_ID)
        self.assert_model_id(properties.to_update_deployment())

        # User may also update the model_id after initializing the ModelDeploymentProperties
        properties = ModelDeploymentProperties(
            bandwidth_mbps=20, memory_in_gbs=10, ocpus=1
        )
        properties.model_id = self.MODEL_ID
        oci_model = properties.build()
        self.assert_model_id(oci_model)
        self.assertEqual(
            oci_model.model_deployment_configuration_details.model_configuration_details.bandwidth_mbps,
            20,
        )
        self.assertEqual(
            oci_model.model_deployment_configuration_details.model_configuration_details.instance_configuration.model_deployment_instance_shape_config_details.memory_in_gbs,
            10,
        )
        self.assertEqual(
            oci_model.model_deployment_configuration_details.model_configuration_details.instance_configuration.model_deployment_instance_shape_config_details.ocpus,
            1,
        )

        properties = ModelDeploymentProperties(
            bandwidth_mbps=20, memory_in_gbs=20, ocpus=2
        )
        properties.model_id = self.MODEL_ID
        oci_model = properties.to_update_deployment()
        self.assert_model_id(oci_model)
        self.assertEqual(
            oci_model.model_deployment_configuration_details.model_configuration_details.bandwidth_mbps,
            20,
        )
        self.assertEqual(
            oci_model.model_deployment_configuration_details.model_configuration_details.instance_configuration.model_deployment_instance_shape_config_details.memory_in_gbs,
            20,
        )
        self.assertEqual(
            oci_model.model_deployment_configuration_details.model_configuration_details.instance_configuration.model_deployment_instance_shape_config_details.ocpus,
            2,
        )
