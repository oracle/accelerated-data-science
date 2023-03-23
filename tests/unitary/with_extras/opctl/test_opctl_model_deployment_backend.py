#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import patch

from ads.opctl.backend.ads_model_deployment import ModelDeploymentBackend
from ads.model import ModelDeployment


class TestModelDeploymentBackend:
    @property
    def config(self):
        return {
            "execution": {
                "backend": "deployment",
                "debug": False,
                "oci_config": "~/.oci/config",
                "oci_profile": "DEFAULT",
                "run_id": "test_model_deployment_id",
                "auth": "api_key",
                "wait_for_completion": False,
                "max_wait_time": 1000,
                "poll_interval": 12,
                "log_type": "predict",
                "log_filter": "test_filter",
                "interval": 3,
            }
        }

    @patch("ads.opctl.backend.ads_model_deployment.ModelDeployment.deploy")
    @patch("ads.opctl.backend.ads_model_deployment.ModelDeployment.from_dict")
    def test_apply(self, mock_from_dict, mock_deploy):
        config = self.config
        mock_from_dict.return_value = ModelDeployment()
        backend = ModelDeploymentBackend(config)
        backend.apply()
        mock_from_dict.assert_called_with(config)
        mock_deploy.assert_called_with(
            wait_for_completion=False,
            max_wait_time=1000,
            poll_interval=12,
        )

    @patch("ads.opctl.backend.ads_model_deployment.ModelDeployment.delete")
    @patch("ads.opctl.backend.ads_model_deployment.ModelDeployment.from_id")
    def test_delete(self, mock_from_id, mock_delete):
        config = self.config
        model_deployment = ModelDeployment()
        model_deployment.set_spec(model_deployment.CONST_LIFECYCLE_STATE, "ACTIVE")
        mock_from_id.return_value = model_deployment
        backend = ModelDeploymentBackend(config)
        backend.delete()
        mock_from_id.assert_called_with("test_model_deployment_id")
        mock_delete.assert_called_with(
            wait_for_completion=False,
            max_wait_time=1000,
            poll_interval=12,
        )

    @patch("ads.opctl.backend.ads_model_deployment.ModelDeployment.activate")
    @patch("ads.opctl.backend.ads_model_deployment.ModelDeployment.from_id")
    def test_activate(self, mock_from_id, mock_activate):
        config = self.config
        model_deployment = ModelDeployment()
        model_deployment.set_spec(model_deployment.CONST_LIFECYCLE_STATE, "INACTIVE")
        mock_from_id.return_value = model_deployment
        backend = ModelDeploymentBackend(config)
        backend.activate()
        mock_from_id.assert_called_with("test_model_deployment_id")
        mock_activate.assert_called_with(
            wait_for_completion=False,
            max_wait_time=1000,
            poll_interval=12,
        )

    @patch("ads.opctl.backend.ads_model_deployment.ModelDeployment.deactivate")
    @patch("ads.opctl.backend.ads_model_deployment.ModelDeployment.from_id")
    def test_deactivate(self, mock_from_id, mock_deactivate):
        config = self.config
        model_deployment = ModelDeployment()
        model_deployment.set_spec(model_deployment.CONST_LIFECYCLE_STATE, "ACTIVE")
        mock_from_id.return_value = model_deployment
        backend = ModelDeploymentBackend(config)
        backend.deactivate()
        mock_from_id.assert_called_with("test_model_deployment_id")
        mock_deactivate.assert_called_with(
            wait_for_completion=False,
            max_wait_time=1000,
            poll_interval=12,
        )

    @patch("ads.opctl.backend.ads_model_deployment.ModelDeployment.watch")
    @patch("ads.opctl.backend.ads_model_deployment.ModelDeployment.from_id")
    def test_watch(self, mock_from_id, mock_watch):
        config = self.config
        mock_from_id.return_value = ModelDeployment()
        backend = ModelDeploymentBackend(config)
        backend.watch()
        mock_from_id.assert_called_with("test_model_deployment_id")
        mock_watch.assert_called_with(
            log_type="predict", interval=3, log_filter="test_filter"
        )
