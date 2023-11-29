#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from ads.model import ModelDeployment
from ads.opctl.backend.ads_model_deployment import ModelDeploymentBackend


class TestModelDeploymentBackend:
    @property
    def curr_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def config(self):
        return {
            "execution": {
                "backend": "deployment",
                "debug": False,
                "oci_config": "~/.oci/config",
                "oci_profile": "DEFAULT",
                "run_id": "test_model_deployment_id",
                "ocid": "fake_model_id",
                "auth": "api_key",
                "wait_for_completion": False,
                "max_wait_time": 1000,
                "poll_interval": 12,
                "log_type": "predict",
                "log_filter": "test_filter",
                "interval": 3,
                "payload": "fake_payload",
                "model_name": "model_name",
                "model_version": "model_version",
            },
            "infrastructure": {
                "compartment_id": "ocid1.compartment.oc1..<unique_id>",
                "project_id": "ocid1.datascienceproject.oc1.<unique_id>",
                "log_group_id": "ocid1.loggroup.oc1.iad.<unique_id>",
                "log_id": "ocid1.log.oc1.iad.<unique_id>",
                "shape_name": "VM.Standard.E4.Flex",
                "bandwidth_mbps": 10,
                "replica": 1,
                "web_concurrency": 10,
            },
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

    @pytest.mark.parametrize(
        "runtime_type",
        ["container", "conda"],
    )
    def test_init(self, runtime_type, monkeypatch):
        """Ensures that starter YAML can be generated for every supported runtime of the Data Flow."""

        # For every supported runtime generate a YAML -> test_files
        # On second iteration remove a temporary code and compare result YAML.
        monkeypatch.delenv("NB_SESSION_OCID", raising=False)

        with tempfile.TemporaryDirectory() as td:
            test_yaml_uri = os.path.join(td, f"modeldeployment_{runtime_type}.yaml")
            expected_yaml_uri = os.path.join(
                self.curr_dir, "test_files", f"modeldeployment_{runtime_type}.yaml"
            )

            ModelDeploymentBackend(self.config).init(
                uri=test_yaml_uri,
                overwrite=False,
                runtime_type=runtime_type,
            )

            with open(test_yaml_uri, "r") as stream:
                test_yaml_dict = yaml.safe_load(stream)
            with open(expected_yaml_uri, "r") as stream:
                expected_yaml_dict = yaml.safe_load(stream)

            assert test_yaml_dict == expected_yaml_dict

    @patch("ads.opctl.backend.ads_model_deployment.ModelDeployment.predict")
    @patch("ads.opctl.backend.ads_model_deployment.ModelDeployment.from_id")
    def test_predict(self, mock_from_id, mock_predict):
        config = self.config
        mock_from_id.return_value = ModelDeployment()
        backend = ModelDeploymentBackend(config)
        backend.predict()
        mock_from_id.assert_called_with("fake_model_id")
        mock_predict.assert_called_with(
            data="fake_payload", model_name="model_name", model_version="model_version"
        )
