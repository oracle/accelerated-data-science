#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

import yaml
from ads.model.runtime.model_provenance_details import (
    ModelProvenanceDetails,
    TrainingCode,
)
from cerberus import DocumentError

try:
    from yaml import CLoader as loader
except:
    from yaml import Loader as loader

import pytest


class TestTrainingCode:
    """TestTrainingCode class."""

    @classmethod
    def setup_class(cls):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(curr_dir, "runtime.yaml"), encoding="utf-8") as rt:
            cls.runtime_dict = yaml.load(rt, loader)

        with open(os.path.join(curr_dir, "runtime_fail.yaml"), encoding="utf-8") as rt:
            cls.runtime_dict_fail = yaml.load(rt, loader)

    def test_from_dict(self):
        training_code = TrainingCode.from_dict(
            self.runtime_dict["MODEL_PROVENANCE"]["TRAINING_CODE"]
        )
        assert training_code.artifact_directory == "fake_artifact_directory"

    def test_from_dict_fail(self):
        with pytest.raises(AssertionError):
            TrainingCode.from_dict(
                self.runtime_dict_fail["MODEL_PROVENANCE"]["TRAINING_CODE"]
            )


class TestModelProvenanceDetails:
    """TestModelProvenanceDetails class"""

    @classmethod
    def setup_class(cls):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(curr_dir, "runtime.yaml"), encoding="utf-8") as rt:
            cls.runtime_dict = yaml.load(rt, loader)

        with open(os.path.join(curr_dir, "runtime_fail.yaml"), encoding="utf-8") as rt:
            cls.runtime_dict_fail = yaml.load(rt, loader)

    def test_from_dict(self):

        model_provenance = ModelProvenanceDetails.from_dict(
            self.runtime_dict["MODEL_PROVENANCE"]
        )
        assert model_provenance.project_ocid == "fake_project_id"
        assert model_provenance.tenancy_ocid == "fake_tenancy_id"
        assert (
            model_provenance.training_code.artifact_directory
            == "fake_artifact_directory"
        )
        assert (
            model_provenance.training_compartment_ocid == "fake_training_compartment_id"
        )
        assert (
            model_provenance.training_conda_env.training_env_path
            == "oci://service_conda_packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        )
        assert model_provenance.training_conda_env.training_env_slug == "mlcpuv1"
        assert model_provenance.training_conda_env.training_env_type == "data_science"
        assert model_provenance.training_conda_env.training_python_version == "3.6"
        assert model_provenance.training_region == "NOT_FOUND"
        assert model_provenance.training_resource_ocid == "fake_resource_id"
        assert model_provenance.vm_image_internal_id == "VMIDNOTSET"

    def test__validate_dict(self):
        assert ModelProvenanceDetails._validate_dict(
            self.runtime_dict["MODEL_PROVENANCE"]
        )

    def test__validate_dict_fail(self):
        with pytest.raises(DocumentError):
            ModelProvenanceDetails._validate_dict(
                self.runtime_dict_fail["MODEL_PROVENANCE"]
            )
