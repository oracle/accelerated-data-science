#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
import tempfile
from unittest.mock import call, patch

import pytest
from ads.model.artifact import (
    AritfactFolderStructureError,
    ArtifactNestedFolderError,
    ArtifactRequiredFilesError,
    ModelArtifact,
    _validate_artifact_dir,
)
from ads.model.runtime.env_info import InferenceEnvInfo, TrainingEnvInfo
from ads.model.runtime.runtime_info import RuntimeInfo
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class TestModelArtifact:
    """Test ModelArtifact class."""

    mlcpu_path_dev = "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
    mlcpu_path_cust = "oci://service-conda-packs@ociodsccust/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
    inference_info_dev = InferenceEnvInfo(
        "mlcpuv1", "data_science", mlcpu_path_dev, "3.6"
    )
    training_info_dev = TrainingEnvInfo(
        "mlcpuv1", "data_science", mlcpu_path_dev, "3.6"
    )
    inference_info_cust = InferenceEnvInfo(
        "mlcpuv1", "data_science", mlcpu_path_cust, "3.6"
    )
    training_info_cust = TrainingEnvInfo(
        "mlcpuv1", "data_science", mlcpu_path_cust, "3.6"
    )

    @classmethod
    def setup_class(cls):
        cls.curdir = os.path.dirname(os.path.abspath(__file__))

    def setup_method(self):
        # called around each method invocation
        self.dirpath = tempfile.mkdtemp()
        self.artifact = ModelArtifact(
            artifact_dir=self.dirpath, model_file_name="fake_model.onnx"
        )

    def teardown_method(self):
        # called around each method invocation
        if os.path.exists(self.dirpath):
            shutil.rmtree(self.dirpath)

    def test_init(self):
        """test init file."""
        artifact = ModelArtifact(
            artifact_dir="fake_folder", model_file_name="fake_name"
        )
        assert artifact.artifact_dir == os.path.abspath(
            os.path.expanduser("fake_folder")
        )
        assert artifact.model_file_name == "fake_name"

    @patch.object(InferenceEnvInfo, "from_slug")
    def test_prepare_runtime_yaml_inference_slug(self, mock_inference_env_info):
        """test generate runtime yaml."""
        file_path = os.path.join(self.dirpath, "runtime.yaml")
        assert not os.path.exists(file_path)
        mock_inference_env_info.return_value = self.inference_info_dev
        self.artifact.prepare_runtime_yaml(inference_conda_env="mlcpuv1")
        assert os.path.exists(file_path)
        runtime_info = RuntimeInfo.from_yaml(uri=file_path)
        assert (
            runtime_info.model_deployment.inference_conda_env == self.inference_info_dev
        )

    @patch.object(InferenceEnvInfo, "from_path")
    def test_prepare_runtime_yaml_inference_path(self, mock_inference_env_info):
        """test generate runtime yaml."""
        file_path = os.path.join(self.dirpath, "runtime.yaml")
        assert not os.path.exists(file_path)
        mock_inference_env_info.return_value = self.inference_info_dev
        self.artifact.prepare_runtime_yaml(inference_conda_env=self.mlcpu_path_dev)
        assert os.path.exists(file_path)
        runtime_info = RuntimeInfo.from_yaml(uri=file_path)
        assert (
            runtime_info.model_deployment.inference_conda_env == self.inference_info_dev
        )

    @patch.object(InferenceEnvInfo, "from_path")
    @patch.object(TrainingEnvInfo, "from_slug")
    def test_prepare_runtime_yaml_inference_training(
        self, mock_training_env_info, mock_inference_env_info
    ):
        """test generate runtime yaml."""
        file_path = os.path.join(self.dirpath, "runtime.yaml")
        assert not os.path.exists(file_path)
        mock_inference_env_info.return_value = self.inference_info_dev
        mock_training_env_info.return_value = self.training_info_dev
        self.artifact.prepare_runtime_yaml(
            inference_conda_env=self.mlcpu_path_dev, training_conda_env="mlcpuv1"
        )
        assert os.path.exists(file_path)
        runtime_info = RuntimeInfo.from_yaml(uri=file_path)
        assert (
            runtime_info.model_deployment.inference_conda_env == self.inference_info_dev
        )
        assert (
            runtime_info.model_provenance.training_conda_env == self.training_info_dev
        )

    @pytest.mark.parametrize(
        "env_info_class, conda_pack, bucketname, namespace, expected_env_info",
        [
            (
                InferenceEnvInfo,
                "mlcpuv1",
                "service-conda-packs",
                "ociodsccust",
                inference_info_cust,
            ),
            (InferenceEnvInfo, mlcpu_path_dev, None, None, inference_info_dev),
            (
                TrainingEnvInfo,
                "mlcpuv1",
                "service-conda-packs",
                "ociodscdev",
                training_info_dev,
            ),
            (TrainingEnvInfo, mlcpu_path_cust, None, None, training_info_cust),
        ],
    )
    def test__populate_env_info_inference(
        self, env_info_class, conda_pack, bucketname, namespace, expected_env_info
    ):
        """test _populate_env_info."""
        env_info = self.artifact._populate_env_info(
            env_info_class,
            conda_pack=conda_pack,
            bucketname=bucketname,
            namespace=namespace,
        )
        assert env_info == expected_env_info

    def test_prepare_score_py(self):
        """test write score.py using local serialization method."""
        template_name = "score"
        self.artifact.prepare_score_py(template_name)
        assert os.path.exists(os.path.join(self.artifact.artifact_dir, "score.py"))
        with open(os.path.join(self.artifact.artifact_dir, "score.py"), "r") as sfl:
            content = sfl.read()
            assert self.artifact.model_file_name in content

    def test_reload_model_not_found(self):
        """test reload function when model is not found."""
        template_name = "score"
        self.artifact.prepare_score_py(template_name)
        with pytest.raises(Exception):
            self.artifact.reload()

    def test_reload(self):
        """test reload function."""
        template_name = "score_onnx"
        self.artifact.prepare_score_py(template_name)

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clr = RandomForestClassifier()
        clr.fit(X_train, y_train)

        # Convert into ONNX format
        initial_type = [("float_input", FloatTensorType([None, 4]))]
        onx = convert_sklearn(clr, initial_types=initial_type)
        with open(
            os.path.join(self.artifact.artifact_dir, "fake_model.onnx"), "wb"
        ) as f:
            f.write(onx.SerializeToString())
        self.artifact.reload()
        assert self.artifact.model is not None
        assert callable(self.artifact.predict)
        assert callable(self.artifact.load_model)

    @patch("os.path.exists")
    def test_from_uri_fail(self, mock_os_path_exists):
        """Ensures `copy_from_uri` fails in case when provided `uri` doesn't exist."""
        mock_os_path_exists.return_value = False
        with pytest.raises(ValueError, match="Provided `uri` doesn't exist."):
            ModelArtifact.from_uri(
                uri="/test/uri",
                artifact_dir="/test/uri",
                model_file_name="test_file_name",
            )

    @pytest.mark.parametrize(
        "test_args",
        [
            {
                "uri": "/src/folder/",
                "artifact_dir": "/dst/folder/",
                "model_file_name": "model.pkl",
            },
            {
                "uri": "/src/folder/",
                "artifact_dir": "/src/folder/",
                "model_file_name": "model.pkl",
            },
            {
                "uri": "/src/folder/",
                "artifact_dir": "/dst/folder/",
                "model_file_name": "model.pkl",
                "force_overwrite": True,
                "auth": {"config": "value"},
            },
        ],
    )
    @patch("ads.common.utils.copy_from_uri")
    @patch("ads.common.auth.default_signer")
    @patch("ads.model.artifact.ModelArtifact.__init__")
    @patch("ads.model.artifact._validate_artifact_dir")
    @patch("shutil.rmtree")
    @patch("ads.common.utils.is_path_exists")
    @patch("tempfile.TemporaryDirectory")
    def test_from_uri(
        self,
        mock_TemporaryDirectory,
        mock_is_path_exists,
        mock_rmtree,
        mock_validate_artifact_dir,
        mock_modelartifact_init,
        mock_default_signer,
        mock_copy_from_uri,
        test_args,
    ):
        """Tests constracting a ModelArtifact object from the existing model artifacts."""
        mock_TemporaryDirectory.return_value.__enter__.return_value = "tmp_folder"
        mock_default_signer.return_value = {"config": "value"}
        mock_modelartifact_init.return_value = None
        mock_is_path_exists.return_value = True
        mock_validate_artifact_dir.side_effect = ArtifactNestedFolderError(
            "nested_folder"
        )

        ModelArtifact.from_uri(**test_args)

        calls = []
        if test_args.get("uri") != test_args.get("artifact_dir"):
            if not test_args.get("auth"):
                mock_default_signer.assert_called()

            calls.append(
                call(
                    uri=test_args.get("uri"),
                    to_path=test_args.get("artifact_dir"),
                    unpack=True,
                    force_overwrite=test_args.get("force_overwrite", False),
                    auth=test_args.get("auth", {"config": "value"}),
                )
            )

        calls.extend(
            [
                call(
                    uri="nested_folder",
                    to_path="tmp_folder",
                    force_overwrite=True,
                ),
                call(
                    uri="tmp_folder",
                    to_path=test_args.get("artifact_dir"),
                    force_overwrite=True,
                ),
            ]
        )

        mock_copy_from_uri.assert_has_calls(calls)
        mock_validate_artifact_dir.assert_called()
        mock_modelartifact_init.assert_called()

    def test__validate_artifact_dir_fail(self):
        """Ensures validating artifacts folder structure fails in case of
        incorrect input parameters."""
        fake_artifact_dir = os.path.abspath(os.path.expanduser("fake_dir"))
        with pytest.raises(ValueError, match="Required artifact files not provided."):
            _validate_artifact_dir(artifact_dir=fake_artifact_dir, required_files=None)
        with pytest.raises(
            ValueError, match=f"The path `{fake_artifact_dir}` not found."
        ):
            _validate_artifact_dir(artifact_dir=fake_artifact_dir)

    def test__validate_artifact_dir_success(self):
        """Tests validating artifacts folder structure."""
        artifact_dir = os.path.join(self.curdir, "./test_files/invalid_model_artifacts")
        with pytest.raises(ArtifactRequiredFilesError):
            _validate_artifact_dir(artifact_dir=artifact_dir)

        artifact_dir = os.path.join(
            self.curdir, "./test_files/invalid_nested_model_artifacts/"
        )
        with pytest.raises(AritfactFolderStructureError):
            _validate_artifact_dir(artifact_dir=artifact_dir)

        artifact_dir = os.path.join(self.curdir, "./test_files/nested_model_artifacts")
        with pytest.raises(ArtifactNestedFolderError) as exc:
            _validate_artifact_dir(artifact_dir=artifact_dir)
            assert exc.folder == os.path.abspath(
                os.path.expanduser(os.path.join(self.curdir, "artifacts"))
            )

        artifact_dir = os.path.join(self.curdir, "./test_files/valid_model_artifacts")
        assert _validate_artifact_dir(artifact_dir=artifact_dir) == True
