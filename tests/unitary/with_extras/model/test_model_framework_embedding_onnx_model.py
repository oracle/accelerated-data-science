#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
import yaml
from ads.model.framework.embedding_onnx_model import EmbeddingONNXModel
from ads.model.model_metadata import Framework


class TestEmbeddingONNXModel:
    def setup_class(cls):
        cls.tmp_model_dir = tempfile.mkdtemp()
        os.makedirs(cls.tmp_model_dir, exist_ok=True)
        cls.inference_conda = "oci://fake_bucket@fake_namespace/inference_conda"
        cls.training_conda = "oci://fake_bucket@fake_namespace/training_conda"

    @patch(
        "ads.model.framework.embedding_onnx_model.EmbeddingONNXModel._validate_artifact_directory"
    )
    def test_init(self, mock_validate):
        model = EmbeddingONNXModel(artifact_dir=self.tmp_model_dir)
        assert model.algorithm == "Embedding_ONNX"
        assert model.framework == Framework.EMBEDDING_ONNX
        mock_validate.assert_called()

    @patch("ads.model.generic_model.GenericModel.verify")
    @patch(
        "ads.model.framework.embedding_onnx_model.EmbeddingONNXModel._validate_artifact_directory"
    )
    def test_prepare_and_verify(self, mock_validate, mock_verify):
        mock_verify.return_value = {"results": "successful"}

        model = EmbeddingONNXModel(artifact_dir=self.tmp_model_dir)
        model.prepare(
            model_file_name="test_model_file_name",
            inference_conda_env=self.inference_conda,
            inference_python_version="3.8",
            training_conda_env=self.training_conda,
            training_python_version="3.8",
            force_overwrite=True,
        )

        assert model.model_file_name == "test_model_file_name"
        artifacts = os.listdir(model.artifact_dir)
        assert "score.py" in artifacts
        assert "runtime.yaml" in artifacts
        assert "openapi.json" in artifacts

        runtime_yaml = os.path.join(model.artifact_dir, "runtime.yaml")
        with open(runtime_yaml, "r") as f:
            runtime_dict = yaml.safe_load(f)

            assert (
                runtime_dict["MODEL_DEPLOYMENT"]["INFERENCE_CONDA_ENV"][
                    "INFERENCE_ENV_PATH"
                ]
                == self.inference_conda
            )
            assert (
                runtime_dict["MODEL_DEPLOYMENT"]["INFERENCE_CONDA_ENV"][
                    "INFERENCE_PYTHON_VERSION"
                ]
                == "3.8"
            )
            assert (
                runtime_dict["MODEL_PROVENANCE"]["TRAINING_CONDA_ENV"][
                    "TRAINING_ENV_PATH"
                ]
                == self.training_conda
            )
            assert (
                runtime_dict["MODEL_PROVENANCE"]["TRAINING_CONDA_ENV"][
                    "TRAINING_PYTHON_VERSION"
                ]
                == "3.8"
            )

        with pytest.raises(
            ValueError,
            match="ADS will not auto serialize `data` for embedding onnx model. Input json serializable `data` and set `auto_serialize_data` as False.",
        ):
            model.verify(data="test_data", auto_serialize_data=True)

        model.verify(data="test_data")
        mock_verify.assert_called_with(
            data="test_data",
            reload_artifacts=True,
            auto_serialize_data=False,
        )
        mock_validate.assert_called()

    @patch("ads.model.generic_model.GenericModel.predict")
    @patch("ads.model.generic_model.GenericModel.deploy")
    @patch("ads.model.generic_model.GenericModel.save")
    @patch(
        "ads.model.framework.embedding_onnx_model.EmbeddingONNXModel._validate_artifact_directory"
    )
    def test_prepare_save_deploy_predict(
        self, mock_validate, mock_save, mock_deploy, mock_predict
    ):
        model = EmbeddingONNXModel(artifact_dir=self.tmp_model_dir)
        model.prepare(
            model_file_name="test_model_file_name",
            inference_conda_env=self.inference_conda,
            inference_python_version="3.8",
            training_conda_env=self.training_conda,
            training_python_version="3.8",
            force_overwrite=True,
        )
        model.save(display_name="test_embedding_onne_model")
        model.deploy(
            display_name="test_embedding_onne_model_deployment",
            deployment_instance_shape="VM.Standard.E4.Flex",
            deployment_ocpus=20,
            deployment_memory_in_gbs=256,
        )

        with pytest.raises(
            ValueError,
            match="ADS will not auto serialize `data` for embedding onnx model. Input json serializable `data` and set `auto_serialize_data` as False.",
        ):
            model.verify(data="test_data", auto_serialize_data=True)

        model.predict(data="test_data")
        mock_predict.assert_called_with(
            data="test_data",
            auto_serialize_data=False,
        )
        mock_save.assert_called_with(display_name="test_embedding_onne_model")
        mock_deploy.assert_called_with(
            display_name="test_embedding_onne_model_deployment",
            deployment_instance_shape="VM.Standard.E4.Flex",
            deployment_ocpus=20,
            deployment_memory_in_gbs=256,
        )
        mock_validate.assert_called()

    def teardown_class(cls):
        shutil.rmtree(cls.tmp_model_dir, ignore_errors=True)
