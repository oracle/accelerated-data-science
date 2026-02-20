#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
import random
from unittest.mock import patch

from ads.common import utils
from ads.model.deployment.model_deployment import ModelDeploymentProperties


class TestModelDeploymentProperties:
    random_seed = 42

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_init_with_default_display_name(self, mock_client, mock_signer):
        """Validate that if no display_name specified, generated random default name will be assigned."""
        random.seed(self.random_seed)
        self.test_properties = ModelDeploymentProperties()
        random.seed(self.random_seed)
        assert (
            self.test_properties.display_name[:-9]
            == utils.get_random_name_for_resource()[:-9]
        )

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_init_with_oci_model_deployment_as_dict_with_instance_configuration(
        self, mock_client, mock_signer
    ):
        """Validate ModelDeploymentProperties.__init__ with 'oci_model_deployment' argument provided
        with dict type. Check instance_config values assigned into model configuration details."""
        test_instance_config = {
            "instance_shape": "test_instance_shape",
            "instance_count": 1,
            "bandwidth_mbps": "test_bandwidth_mbps",
            "memory_in_gbs": 10,
            "ocpus": 1,
        }

        self.test_properties = ModelDeploymentProperties(
            oci_model_deployment=test_instance_config
        )
        assert (
            self.test_properties.model_deployment_configuration_details.model_configuration_details.instance_configuration.instance_shape_name
            == "test_instance_shape"
        )
        assert (
            self.test_properties.model_deployment_configuration_details.model_configuration_details.scaling_policy.instance_count
            == 1
        )
        assert (
            self.test_properties.model_deployment_configuration_details.model_configuration_details.bandwidth_mbps
            == "test_bandwidth_mbps"
        )

        assert (
            self.test_properties.model_deployment_configuration_details.model_configuration_details.bandwidth_mbps
            == "test_bandwidth_mbps"
        )

        assert (
            self.test_properties.model_deployment_configuration_details.model_configuration_details.instance_configuration.model_deployment_instance_shape_config_details.memory_in_gbs
            == 10
        )
        assert (
            self.test_properties.model_deployment_configuration_details.model_configuration_details.instance_configuration.model_deployment_instance_shape_config_details.ocpus
            == 1
        )

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_init_with_oci_model_deployment_access_log_id_value_error(
        self, mock_client, mock_signer
    ):
        """Validate ValueError when 'access_log_group_id' provided, but 'access_log_id' not."""
        test_instance_config = {
            "access_log_group_id": "test_access_log_group_id",
        }
        with pytest.raises(ValueError):
            ModelDeploymentProperties(oci_model_deployment=test_instance_config)

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_init_with_oci_model_deployment_predict_log_is_value_error(
        self, mock_client, mock_signer
    ):
        """Validate ValueError when 'predict_log_group_id' provided, but 'predict_log_id' not."""
        test_instance_config = {
            "predict_log_group_id": "test_predict_log_group_id",
        }
        with pytest.raises(ValueError):
            ModelDeploymentProperties(oci_model_deployment=test_instance_config)

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_with_instance_configuration_autoscaling_cpu_utilization(
        self, mock_client, mock_signer
    ):
        """Validate SINGLE_MODEL autoscaling policy is constructed when scaling_type is set."""
        props = ModelDeploymentProperties(model_id="test_model_id")
        props.with_instance_configuration(
            config={
                "instance_shape": "VM.Standard.E4.Flex",
                "memory_in_gbs": 16,
                "ocpus": 1,
                "bandwidth_mbps": 10,
                "scaling_type": "cpu_utilization",
                "minimum_instance_count": 1,
                "maximum_instance_count": 3,
                "initial_instance_count": 1,
                # exercise defaults for thresholds (30/70) by not specifying them
            }
        )

        scaling_policy = (
            props.model_deployment_configuration_details.model_configuration_details.scaling_policy
        )
        assert scaling_policy.policy_type == "AUTOSCALING"
        assert scaling_policy.is_enabled is True
        assert scaling_policy.auto_scaling_policies[0].auto_scaling_policy_type == "THRESHOLD"
        assert scaling_policy.auto_scaling_policies[0].minimum_instance_count == 1
        assert scaling_policy.auto_scaling_policies[0].maximum_instance_count == 3
        assert scaling_policy.auto_scaling_policies[0].initial_instance_count == 1
        rule = scaling_policy.auto_scaling_policies[0].rules[0]
        assert rule.metric_type == "CPU_UTILIZATION"
        assert rule.scale_in_configuration.threshold == 30
        assert rule.scale_out_configuration.threshold == 70
