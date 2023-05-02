#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

import pytest
from ads.model.runtime.runtime_info import RuntimeInfo
from unittest.mock import patch
from ads.model.runtime.model_deployment_details import ModelDeploymentDetails
from ads.model.runtime.model_provenance_details import ModelProvenanceDetails
from cerberus import DocumentError
from ads.model.runtime.env_info import InferenceEnvInfo, TrainingEnvInfo
import shutil


class TestRuntimeInfo:
    """The class to test RuntimeInfo class."""

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_dict = {
        "MODEL_ARTIFACT_VERSION": None,
        "MODEL_DEPLOYMENT": None,
        "MODEL_PROVENANCE": None,
    }
    runtime_dict_fail = {
        "MODEL_ARTIFACT_VERSION": None,
        "MODEL_DEPLOYMENT": None,
    }

    def test__validate_dict(self):
        assert RuntimeInfo._validate_dict(self.runtime_dict)

    def test__validate_dict_fail(self):
        with pytest.raises(AssertionError):
            RuntimeInfo._validate_dict(self.runtime_dict_fail)

    def test_from_yaml(self):
        runtime_info = RuntimeInfo.from_yaml(
            uri=os.path.join(self.curr_dir, "runtime.yaml")
        )
        assert runtime_info.model_artifact_version == "3.0"
        assert (
            runtime_info.model_deployment.inference_conda_env.inference_env_path
            == "oci://service_conda_packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        )
        assert (
            runtime_info.model_deployment.inference_conda_env.inference_env_slug
            == "mlcpuv1"
        )
        assert (
            runtime_info.model_deployment.inference_conda_env.inference_env_type
            == "data_science"
        )
        assert (
            runtime_info.model_deployment.inference_conda_env.inference_python_version
            == "3.6"
        )

        assert (
            runtime_info.model_provenance.training_conda_env.training_env_path
            == "oci://service_conda_packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        )
        assert (
            runtime_info.model_provenance.training_conda_env.training_env_slug
            == "mlcpuv1"
        )
        assert (
            runtime_info.model_provenance.training_conda_env.training_env_type
            == "data_science"
        )
        assert (
            runtime_info.model_provenance.training_conda_env.training_python_version
            == "3.6"
        )

        assert runtime_info.model_provenance.project_ocid == "fake_project_id"
        assert runtime_info.model_provenance.tenancy_ocid == "fake_tenancy_id"
        assert (
            runtime_info.model_provenance.training_compartment_ocid
            == "fake_training_compartment_id"
        )
        assert runtime_info.model_provenance.training_region == "NOT_FOUND"
        assert (
            runtime_info.model_provenance.training_resource_ocid == "fake_resource_id"
        )
        assert runtime_info.model_provenance.user_ocid == "NOT_FOUND"
        assert runtime_info.model_provenance.vm_image_internal_id == "VMIDNOTSET"

        assert (
            runtime_info.model_provenance.training_code.artifact_directory
            == "fake_artifact_directory"
        )

    @patch.object(InferenceEnvInfo, "_validate", side_effect=DocumentError)
    @patch.object(TrainingEnvInfo, "_validate", side_effect=DocumentError)
    def test_from_yaml_fail(self, mock_inference, mock_training):
        with pytest.raises(DocumentError):
            RuntimeInfo.from_yaml(uri=os.path.join(self.curr_dir, "runtime_fail.yaml"))

    @patch.object(
        ModelDeploymentDetails, "from_dict", return_value=ModelDeploymentDetails()
    )
    @patch.object(
        ModelProvenanceDetails, "from_dict", return_value=ModelProvenanceDetails()
    )
    def test_from_yaml_wrong_format(self, mock_provenance, mock_deployment):
        with pytest.raises(AssertionError):
            RuntimeInfo.from_yaml(uri=os.path.join(self.curr_dir, "index.json"))

    @patch.object(
        ModelDeploymentDetails, "from_dict", return_value=ModelDeploymentDetails()
    )
    @patch.object(
        ModelProvenanceDetails, "from_dict", return_value=ModelProvenanceDetails()
    )
    def test_from_yaml_wrong_format(self, mock_provenance, mock_deployment):
        with pytest.raises(FileNotFoundError):
            RuntimeInfo.from_yaml(uri=os.path.join(self.curr_dir, "fake.yaml"))

    def test_from_and_to_yaml_file(self):
        runtime = RuntimeInfo.from_yaml(uri=os.path.join(self.curr_dir, "runtime.yaml"))
        runtime.to_yaml(uri=os.path.join(self.curr_dir, "runtime_copy.yaml"))
        runtime_copy = RuntimeInfo.from_yaml(
            uri=os.path.join(self.curr_dir, "runtime.yaml")
        )
        assert runtime_copy == runtime

    def teardown_method(self):
        file_name = os.path.join(self.curr_dir, "runtime_copy.yaml")
        if os.path.exists(file_name):
            os.remove(file_name)
