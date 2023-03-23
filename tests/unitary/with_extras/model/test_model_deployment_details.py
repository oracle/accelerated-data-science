#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from unittest import TestCase

import yaml
from ads.model.runtime.model_deployment_details import ModelDeploymentDetails

try:
    from yaml import CLoader as loader
except:
    from yaml import Loader as loader

import pytest


class TestModelDeploymentDetails(TestCase):
    @classmethod
    def setUpClass(cls):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(curr_dir, "runtime.yaml"), encoding="utf-8") as rt:
            cls.runtime_dict = yaml.load(rt, loader)

        with open(os.path.join(curr_dir, "runtime_fail.yaml"), encoding="utf-8") as rt:
            cls.runtime_dict_fail = yaml.load(rt, loader)

    def test_from_dict(self):
        model_deployment = ModelDeploymentDetails.from_dict(
            self.runtime_dict["MODEL_DEPLOYMENT"]
        )
        assert (
            model_deployment.inference_conda_env.inference_env_path
            == "oci://service_conda_packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        )
        assert model_deployment.inference_conda_env.inference_env_slug == "mlcpuv1"
        assert model_deployment.inference_conda_env.inference_env_type == "data_science"
        assert model_deployment.inference_conda_env.inference_python_version == "3.6"

    def test__validate_dict_fail(self):
        assert ModelDeploymentDetails._validate_dict(
            self.runtime_dict["MODEL_DEPLOYMENT"]
        )

    def test__validate_dict_fail(self):
        with pytest.raises(AssertionError):
            ModelDeploymentDetails._validate_dict({"Key": "value"})
