#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
from unittest.mock import Mock, patch

from ads.common import auth, oci_client
from ads.model.deployment.model_deployment import ModelDeployment, ModelDeploymentProperties


class ModelDeploymentTestCase(unittest.TestCase):
    MODEL_ID = "<MODEL_OCID>"
    with patch.object(auth, "default_signer"):
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
        test_result = self.test_model_deployment.predict(json_input="test")
        mock_post.assert_called_with(
            f"{self.test_model_deployment.url}/predict",
            json="test",
            headers={"Content-Type": "application/json"},
            auth=self.test_model_deployment.config.get("signer"),
        )
        assert test_result == {"result": "result"}

    @patch("requests.post")
    def test_predict_with_bytes(self, mock_post):
        """Ensures predict model passes with bytes input."""
        byte_data = b"[[1,2,3,4]]"
        self.test_model_deployment.predict(data=byte_data)
        mock_post.assert_called_with(
            f"{self.test_model_deployment.url}/predict",
            data=byte_data,
            auth=self.test_model_deployment.config.get("signer"),
            headers={"Content-Type": "application/octet-stream"},
        )

    @patch("requests.post")
    def test_predict_with_auto_serialize_data(self, mock_post):
        """Ensures predict model passes with valid input parameters."""
        mock_post.return_value = Mock(
            status_code=200, json=lambda: {"result": "result"}
        )
        test_result = self.test_model_deployment.predict(
            json_input="test", auto_serialize_data=True
        )
        mock_post.assert_called_with(
            f"{self.test_model_deployment.url}/predict",
            json={"data": "test", "data_type": "<class 'str'>"},
            headers={"Content-Type": "application/json"},
            auth=self.test_model_deployment.config.get("signer"),
        )
        assert test_result == {"result": "result"}

class ModelDeploymentPropertiesTestCase(unittest.TestCase):
    MODEL_ID = "<MODEL_OCID>"

    # Current unittests running mock for "oci.config.from_file" and has specific requirement for test_config:
    # "tenancy", "user", "fingerprint" must fit the ocid pattern.
    # Add "# must be a real-like ocid" in the same line to pass pre-commit hook validation
    test_config = {
        "tenancy": "ocid1.tenancy.oc1..xxx",  # must be a real-like ocid
        "user": "ocid1.user.oc1..xxx",  # must be a real-like ocid
        "fingerprint": "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
        "key_file": "<path>/<to>/<key_file>",
        "region": "<region>",
    }

    def assert_model_id(self, oci_model):
        """Checks if the model OCID is configured correctly."""
        self.assertIsNotNone(oci_model.model_deployment_configuration_details)
        self.assertEqual(
            oci_model.model_deployment_configuration_details.model_configuration_details.model_id,
            self.MODEL_ID,
        )

    @patch("oci.config.from_file", return_value=test_config)
    @patch("oci.signer.load_private_key_from_file")
    def test_setting_model_deployment_with_model_id(self, mock_load_key_file, mock_config_from_file):
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
