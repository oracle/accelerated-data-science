#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for model frameworks. Includes tests for:
 - GenericModel
"""
import os
import random
import shutil
from copy import copy
import glob
import tempfile
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest
import yaml
import numpy as np
from ads.common import utils
from ads.config import (
    JOB_RUN_COMPARTMENT_OCID,
    NB_SESSION_COMPARTMENT_OCID,
)
from ads.model.artifact import ModelArtifact
from ads.model.deployment import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIME,
    ModelDeployer,
    ModelDeployment,
    ModelDeploymentProperties,
)
from ads.model.deployment.common.utils import State as ModelDeploymentState
from ads.model.deployment import (
    ModelDeploymentInfrastructure,
    ModelDeploymentContainerRuntime,
)
from ads.model.generic_model import (
    _ATTRIBUTES_TO_SHOW_,
    GenericModel,
    NotActiveDeploymentError,
    SummaryStatus,
    _prepare_artifact_dir,
)
from ads.model.model_properties import ModelProperties
from ads.model.runtime.runtime_info import RuntimeInfo
from ads.model.datascience_model import DataScienceModel, OCIDataScienceModel
from joblib import dump
from oci.data_science.models.model_provenance import ModelProvenance
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from ads.model.deployment.common.utils import State

try:
    from yaml import CDumper as dumper
except:
    from yaml import Dumper as dumper


_COMPARTMENT_OCID = NB_SESSION_COMPARTMENT_OCID or JOB_RUN_COMPARTMENT_OCID


DSC_MODEL_PAYLOAD = {
    "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
    "projectId": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
    "displayName": "Generic Model With Small Artifact new",
    "description": "The model description",
    "freeformTags": {"key1": "value1"},
    "definedTags": {"key1": {"skey1": "value1"}},
    "inputSchema": {
        "schema": [
            {
                "feature_type": "Integer",
                "dtype": "int64",
                "name": 0,
                "domain": {"values": "", "stats": {}, "constraints": []},
                "required": True,
                "description": "0",
                "order": 0,
            }
        ],
        "version": "1.1",
    },
    "outputSchema": {
        "schema": [
            {
                "dtype": "int64",
                "feature_type": "Integer",
                "name": 0,
                "domain": {"values": "", "stats": {}, "constraints": []},
                "required": True,
                "description": "0",
                "order": 0,
            }
        ],
        "version": "1.1",
    },
    "customMetadataList": {
        "data": [
            {
                "key": "CondaEnvironment",
                "value": "oci://bucket@namespace/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/3.0/dataexpl_p37_cpu_v3",
                "description": "The conda environment where the model was trained.",
                "category": "Training Environment",
            },
        ]
    },
    "definedMetadataList": {
        "data": [
            {"key": "Algorithm", "value": "test"},
            {"key": "Framework", "value": None},
            {"key": "FrameworkVersion", "value": None},
            {"key": "UseCaseType", "value": "multinomial_classification"},
            {"key": "Hyperparameters", "value": None},
            {"key": "ArtifactTestResults", "value": None},
        ]
    },
    "provenanceMetadata": {
        "git_branch": "master",
        "git_commit": "7c8c8502896ba36837f15037b67e05a3cf9722c7",
        "repository_url": "file:///home/datascience",
        "training_script_path": None,
        "training_id": None,
        "artifact_dir": "test_script_dir",
    },
    "artifact": "ocid1.datasciencemodel.oc1.iad.<unique_ocid>.zip",
}

OCI_MODEL_PAYLOAD = {
    "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
    "project_id": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
    "display_name": "Generic Model With Small Artifact new",
    "description": "The model description",
    "lifecycle_state": "ACTIVE",
    "created_by": "ocid1.user.oc1..<unique_ocid>",
    "freeform_tags": {"key1": "value1"},
    "defined_tags": {"key1": {"skey1": "value1"}},
    "time_created": "2022-08-24T17:07:39.200000Z",
    "custom_metadata_list": [
        {
            "key": "CondaEnvironment",
            "value": "oci://bucket@namespace/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/3.0/dataexpl_p37_cpu_v3",
            "description": "The conda environment where the model was trained.",
            "category": "Training Environment",
        },
    ],
    "defined_metadata_list": [
        {"key": "Algorithm", "value": "test"},
        {"key": "Framework"},
        {"key": "FrameworkVersion"},
        {"key": "UseCaseType", "value": "multinomial_classification"},
        {"key": "Hyperparameters"},
        {"key": "ArtifactTestResults"},
    ],
    "input_schema": '{"schema": [{"dtype": "int64", "feature_type": "Integer", "name": 0, "domain": {"values": "", "stats": {}, "constraints": []}, "required": true, "description": "0", "order": 0}], "version": "1.1"}',
    "output_schema": '{"schema": [{"dtype": "int64", "feature_type": "Integer", "name": 0, "domain": {"values": "", "stats": {}, "constraints": []}, "required": true, "description": "0", "order": 0}], "version": "1.1"}',
}

OCI_MODEL_PROVENANCE_PAYLOAD = {
    "git_branch": "master",
    "git_commit": "7c8c8502896ba36837f15037b67e05a3cf9722c7",
    "repository_url": "file:///home/datascience",
    "script_dir": "test_script_dir",
    "training_id": None,
    "training_script": None,
}

INFERENCE_CONDA_ENV = "oci://bucket@namespace/<path_to_service_pack>"
TRAINING_CONDA_ENV = "oci://bucket@namespace/<path_to_service_pack>"
DEFAULT_PYTHON_VERSION = "3.8"
MODEL_FILE_NAME = "fake_model_name"
FAKE_MD_URL = "http://<model-deployment-url>"


def _prepare(model):
    model.prepare(
        inference_conda_env=INFERENCE_CONDA_ENV,
        inference_python_version=DEFAULT_PYTHON_VERSION,
        training_conda_env=TRAINING_CONDA_ENV,
        training_python_version=DEFAULT_PYTHON_VERSION,
        model_file_name=MODEL_FILE_NAME,
        force_overwrite=True,
    )


class TestEstimator:
    def predict(self, x):
        return x**2


class TestGenericModel:
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    random_seed = 42

    @pytest.fixture(autouse=True, scope="module")
    def conda_file(self, tmpdir_factory):
        conda_file = tmpdir_factory.mktemp("conda")
        manifest = {
            "manifest": {
                "pack_path": "oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/pyspark/1.0/pyspv10",
                "python": "3.6",
                "slug": "pyspv10",
                "type": "data_science",
                "version": "1.0",
                "arch_type": "CPU",
                "manifest_version": "1.0",
                "name": "pyspark",
            }
        }
        with open(os.path.join(conda_file.strpath, "test_manifest.yaml"), "w") as mfile:
            yaml.dump(manifest, mfile)

        conda_prefix = os.environ["CONDA_PREFIX"]
        os.environ["CONDA_PREFIX"] = conda_file.strpath
        yield conda_file
        os.environ["CONDA_PREFIX"] = conda_prefix

    def setup_class(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))
        cls.clr = RandomForestClassifier()
        cls.clr.fit(cls.X_train, cls.y_train)

    def setup_method(self):
        self.generic_model = GenericModel(
            estimator=self.clr, artifact_dir="fake_folder"
        )
        self.mock_dsc_model = DataScienceModel(**DSC_MODEL_PAYLOAD)

    def teardown_method(self):
        if self.generic_model.artifact_dir and os.path.exists(
            self.generic_model.artifact_dir
        ):
            shutil.rmtree(self.generic_model.artifact_dir, ignore_errors=True)

    def test_init(self):
        """test the init function"""
        assert self.generic_model.artifact_dir == os.path.abspath(
            os.path.expanduser("fake_folder")
        )
        assert self.generic_model.estimator == self.clr

    def test__handle_model_file_name(self):
        """test the model file name."""
        self.generic_model.model_file_name = self.generic_model._handle_model_file_name(
            as_onnx=True, model_file_name="fake_name.onnx"
        )
        assert self.generic_model.model_file_name == "fake_name.onnx"

    def test__handle_model_file_name_raise_error(self):
        with pytest.raises(NotImplementedError):
            self.generic_model._serialize = False
            self.generic_model._handle_model_file_name(
                as_onnx=False, model_file_name=None
            )

    @patch("ads.common.auth.default_signer")
    def test_prepare(self, mock_signer):
        """prepare a trained model."""
        self.generic_model.prepare(
            "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1",
            model_file_name="fake_model_name",
        )

        assert os.path.exists(os.path.join("fake_folder", "runtime.yaml"))
        assert os.path.exists(os.path.join("fake_folder", "score.py"))

    @patch.object(GenericModel, "_handle_model_file_name", return_value=None)
    def test_prepare_fail(self, mock_handle_model_file_name):
        """Ensures that prepare method fails in case if model_file_name not provided."""
        with pytest.raises(
            ValueError, match="The `model_file_name` needs to be provided."
        ):
            self.generic_model.prepare(
                "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
            )

    @patch("ads.common.auth.default_signer")
    def test_prepare_both_conda_env(self, mock_signer):
        """prepare a model by only providing inference conda env."""
        self.generic_model.prepare(
            inference_conda_env="oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1",
            inference_python_version="3.6",
            training_conda_env="oci://service-conda-packs@ociodscdev/service_pack/cpu/Oracle_Database_for_CPU_Python_3.7/1.0/database_p37_cpu_v1",
            training_python_version="3.7",
            model_file_name="fake_model_name",
            force_overwrite=True,
        )
        assert os.path.exists(os.path.join("fake_folder", "runtime.yaml"))
        assert os.path.exists(os.path.join("fake_folder", "score.py"))
        runtime_yaml_file = os.path.join(
            self.generic_model.artifact_dir, "runtime.yaml"
        )
        runtime_info = RuntimeInfo.from_yaml(uri=runtime_yaml_file)
        assert (
            runtime_info.model_deployment.inference_conda_env.inference_env_path
            == "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        )
        assert (
            runtime_info.model_deployment.inference_conda_env.inference_python_version
            == "3.6"
        )
        assert (
            runtime_info.model_provenance.training_conda_env.training_env_path
            == "oci://service-conda-packs@ociodscdev/service_pack/cpu/Oracle_Database_for_CPU_Python_3.7/1.0/database_p37_cpu_v1"
        )
        assert (
            runtime_info.model_provenance.training_conda_env.training_python_version
            == "3.7"
        )

    @patch("ads.common.auth.default_signer")
    def test_prepare_with_custom_scorepy(self, mock_signer):
        """Test prepare a trained model with custom score.py."""
        self.generic_model.prepare(
            INFERENCE_CONDA_ENV,
            model_file_name="fake_model_name",
            score_py_uri=f"{os.path.dirname(os.path.abspath(__file__))}/test_files/custom_score.py",
        )
        assert os.path.exists(os.path.join("fake_folder", "score.py"))

        prediction = self.generic_model.verify(data="test")["prediction"]
        assert prediction == "This is a custom score.py."

    @patch("ads.common.auth.default_signer")
    def test_verify_without_reload(self, mock_signer):
        """Test verify input data without reload artifacts."""
        _prepare(self.generic_model)
        self.generic_model.verify(self.X_test.tolist())

        with patch("ads.model.artifact.ModelArtifact.reload") as mock_reload:
            self.generic_model.verify(self.X_test.tolist(), reload_artifacts=False)
            mock_reload.assert_not_called()

    @patch("ads.common.auth.default_signer")
    def test_verify(self, mock_signer):
        """Test verify input data"""
        _prepare(self.generic_model)
        prediction_1 = self.generic_model.verify(self.X_test.tolist())
        assert isinstance(prediction_1, dict), "Failed to verify json payload."

    def test_reload(self):
        """test the reload."""
        pass

    @patch.object(GenericModel, "_random_display_name", return_value="test_name")
    @patch.object(DataScienceModel, "create")
    def test_save(self, mock_dsc_model_create, mock__random_display_name):
        """test saving a model to artifact."""
        mock_dsc_model_create.return_value = MagicMock(id="fake_id")
        self.generic_model.prepare(
            inference_conda_env="dataexpl_p37_cpu_v3",
            namespace="ociodscdev",
            inference_python_version="3.7",
            model_file_name="model.joblib",
            force_overwrite=True,
            training_id=None,
        )
        self.generic_model.save()
        assert self.generic_model.model_id is not None and isinstance(
            self.generic_model.model_id, str
        )
        mock_dsc_model_create.assert_called_with(
            bucket_uri=None,
            overwrite_existing_artifact=True,
            remove_existing_artifact=True,
        )

    def test_save_not_implemented_error(self):
        """test saving a model to artifact."""
        self.generic_model._serialize = False
        self.generic_model.prepare(
            inference_conda_env="dataexpl_p37_cpu_v3",
            namespace="ociodscdev",
            inference_python_version="3.7",
            model_file_name="model.joblib",
            force_overwrite=True,
            training_id=None,
        )

        dump(
            self.clr,
            os.path.join(
                self.generic_model.artifact_dir, self.generic_model.model_file_name
            ),
        )
        with pytest.raises(NotImplementedError):
            self.generic_model.save()

    def test_set_model_input_serializer(self):
        """Tests set_model_input_serializer() with different input types."""
        from ads.model.serde.model_input import (
            CloudpickleModelInputSERDE,
            JsonModelInputSERDE,
        )
        from ads.model.serde.common import SERDE

        generic_model = GenericModel(estimator=self.clr, artifact_dir="fake_folder")
        # set by passing str
        generic_model.set_model_input_serializer("cloudpickle")
        assert isinstance(
            generic_model.get_data_serializer(), CloudpickleModelInputSERDE
        )

        # set by passing ModelInputSerializerType
        generic_model.set_model_input_serializer("json")
        assert isinstance(generic_model.get_data_serializer(), JsonModelInputSERDE)

        # set customized serialize by inheriting from SERDE
        class MySERDEA(SERDE):
            def __init__(self):
                super().__init__()

            def serialize(self, data):
                return 1

            def deserialize(self, data):
                return 2

        generic_model.set_model_input_serializer(model_input_serializer=MySERDEA())
        assert generic_model.get_data_serializer().serialize(2) == 1

        # set customized serialize without inheritance
        class MySERDEB:
            def __init__(self):
                super().__init__()

            def serialize(self, data):
                return 2

            def deserialize(self, data):
                return 1

        generic_model.set_model_input_serializer(model_input_serializer=MySERDEB())
        assert generic_model.get_data_serializer().serialize(1) == 2
        assert generic_model.model_input_serializer.name == "customized"

    @patch("ads.model.serde.model_input.JsonModelInputSerializer.serialize")
    def test_handle_input(self, mock_serializer):
        """Test validate input data from verify/predict."""
        fake_path = "/tmp/fake_path"
        with patch.object(GenericModel, "_handle_image_input") as mock_handle_image:
            self.generic_model._handle_input_data(**{"image": fake_path})
            mock_handle_image.assert_called_with(image=fake_path)

        with pytest.raises(TypeError):
            self.generic_model._handle_input_data(
                data=self.X_test, auto_serialize_data=False
            )

        with pytest.raises(ValueError):
            self.generic_model._handle_input_data(json=self.X_test.tolist())

    @pytest.mark.parametrize(
        "test_data",
        [
            pd.Series([1, 2, 3]),
            [1, 2, 3],
            np.array([1, 2, 3]),
            pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4]}),
            "I have an apple",
            {"a": [1], "b": [2], "c": [3]},
        ],
    )
    @patch("ads.model.serde.model_input.JsonModelInputSerializer.serialize")
    def test_handle_different_input_data(self, mock_serializer, test_data):
        """Test validate input data from verify/predict."""
        self.generic_model._handle_input_data(data=test_data)
        mock_serializer.assert_called_with(data=test_data)

    def test_handle_image_input_fail(self):
        """Test extracting an invalid image arg."""
        invalid_image_path = os.path.join(
            f"{os.path.dirname(os.path.abspath(__file__))}/test_files/invalid_model_artifacts/runtime.yaml",
        )
        with pytest.raises(ValueError):
            self.generic_model._handle_image_input(image=invalid_image_path)

    def test_generic_model_serialize(self):
        self.generic_model.prepare(
            inference_conda_env="dataexpl_p37_cpu_v3",
            namespace="ociodscdev",
            inference_python_version="3.7",
            force_overwrite=True,
            training_id=None,
        )
        assert self.generic_model.model_file_name == "model.pkl"

    @patch.object(ModelDeployer, "deploy")
    def test_deploy_fail(self, mock_deploy):
        """Ensures model deployment fails with invalid input paramters."""
        with pytest.raises(
            ValueError,
            match=(
                "The model needs to be saved to the Model Catalog "
                "before it can be deployed."
            ),
        ):
            self.generic_model.deploy()

        with pytest.raises(
            ValueError, match="`deployment_log_group_id` needs to be specified."
        ):
            self.generic_model.dsc_model = MagicMock(id="test_model")
            self.generic_model.deploy(
                deployment_access_log_id="log_id", deployment_predict_log_id="log_id"
            )

        mock_deploy.assert_not_called()

    @patch.object(ModelDeployment, "deploy")
    def test_deploy_success(self, mock_deploy):
        test_model_id = "ocid.test_model_id"
        self.generic_model.dsc_model = MagicMock(id=test_model_id)
        self.generic_model.ignore_conda_error = True
        infrastructure = ModelDeploymentInfrastructure(
            **{
                "shape_name": "test_deployment_instance_shape",
                "subnet_id": "test_deployment_subnet_id",
                "replica": 10,
                "bandwidth_mbps": 100,
                "shape_config_details": {"memory_in_gbs": 10, "ocpus": 1},
                "access_log": {
                    "log_group_id": "test_deployment_log_group_id",
                    "log_id": "test_deployment_access_log_id",
                },
                "predict_log": {
                    "log_group_id": "test_deployment_log_group_id",
                    "log_id": "test_deployment_predict_log_id",
                },
                "project_id": "test_project_id",
                "compartment_id": "test_compartment_id",
            }
        )
        runtime = ModelDeploymentContainerRuntime(
            **{
                "image": "test_docker_image",
                "image_digest": "test_image_digest",
                "cmd": ["test_cmd"],
                "entrypoint": ["test_entrypoint"],
                "server_port": 8080,
                "health_check_port": 8080,
                "env": {"test_key": "test_value"},
                "deployment_mode": "HTTPS_ONLY",
            }
        )
        mock_deploy.return_value = ModelDeployment(
            display_name="test_display_name",
            description="test_description",
            infrastructure=infrastructure,
            runtime=runtime,
            model_deployment_url="test_model_deployment_url",
            model_deployment_id="test_model_deployment_id",
        )
        input_dict = {
            "wait_for_completion": True,
            "display_name": "test_display_name",
            "description": "test_description",
            "deployment_instance_shape": "test_deployment_instance_shape",
            "deployment_instance_subnet_id": "test_deployment_subnet_id",
            "deployment_instance_count": 10,
            "deployment_bandwidth_mbps": 100,
            "deployment_memory_in_gbs": 10,
            "deployment_ocpus": 1,
            "deployment_log_group_id": "test_deployment_log_group_id",
            "deployment_access_log_id": "test_deployment_access_log_id",
            "deployment_predict_log_id": "test_deployment_predict_log_id",
            "deployment_image": "test_docker_image",
            "cmd": ["test_cmd"],
            "entrypoint": ["test_entrypoint"],
            "server_port": 8080,
            "health_check_port": 8080,
            "environment_variables": {"test_key": "test_value"},
            "project_id": "test_project_id",
            "compartment_id": "test_compartment_id",
            "max_wait_time": 100,
            "poll_interval": 200,
        }

        result = self.generic_model.deploy(
            **input_dict,
        )
        assert result == mock_deploy.return_value
        assert result.infrastructure.access_log == {
            "log_id": input_dict["deployment_access_log_id"],
            "log_group_id": input_dict["deployment_log_group_id"],
        }
        assert result.infrastructure.predict_log == {
            "log_id": input_dict["deployment_predict_log_id"],
            "log_group_id": input_dict["deployment_log_group_id"],
        }
        assert (
            result.infrastructure.bandwidth_mbps
            == input_dict["deployment_bandwidth_mbps"]
        )
        assert result.infrastructure.compartment_id == input_dict["compartment_id"]
        assert result.infrastructure.project_id == input_dict["project_id"]
        assert (
            result.infrastructure.shape_name == input_dict["deployment_instance_shape"]
        )
        assert result.infrastructure.shape_config_details == {
            "ocpus": input_dict["deployment_ocpus"],
            "memory_in_gbs": input_dict["deployment_memory_in_gbs"],
        }
        assert result.infrastructure.subnet_id == input_dict["deployment_instance_subnet_id"]
        assert result.runtime.image == input_dict["deployment_image"]
        assert result.runtime.entrypoint == input_dict["entrypoint"]
        assert result.runtime.server_port == input_dict["server_port"]
        assert result.runtime.health_check_port == input_dict["health_check_port"]
        assert result.runtime.env == {"test_key": "test_value"}
        assert result.runtime.deployment_mode == "HTTPS_ONLY"
        mock_deploy.assert_called_with(
            wait_for_completion=input_dict["wait_for_completion"],
            max_wait_time=input_dict["max_wait_time"],
            poll_interval=input_dict["poll_interval"],
        )

    @patch.object(ModelDeployment, "deploy")
    def test_deploy_with_default_display_name(self, mock_deploy):
        """Ensure that a randomly generated easy to remember name will be generated,
        when display_name not specified (default)."""
        test_model_id = "ocid.test_model_id"
        self.generic_model.dsc_model = MagicMock(id=test_model_id)
        random_name = utils.get_random_name_for_resource()
        mock_deploy.return_value = ModelDeployment(display_name=random_name)
        random.seed(self.random_seed)
        self.generic_model.deploy(
            compartment_id="test_compartment_id", project_id="test_project_id"
        )
        random.seed(self.random_seed)
        assert (
            self.generic_model.model_deployment.properties.display_name[:-9]
            == random_name[:-9]
        )

    @pytest.mark.parametrize("input_data", [(X_test.tolist())])
    @patch("ads.common.auth.default_signer")
    def test_predict_locally(self, mock_signer, input_data):
        _prepare(self.generic_model)
        test_result = self.generic_model.predict(data=input_data, local=True)
        expected_result = self.generic_model.estimator.predict(input_data).tolist()
        assert (
            test_result["prediction"] == expected_result
        ), "Failed to verify input data."

        with patch("ads.model.artifact.ModelArtifact.reload") as mock_reload:
            self.generic_model.predict(
                data=input_data, local=True, reload_artifacts=False
            )
            mock_reload.assert_not_called()

    @patch.object(ModelDeployment, "predict")
    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    @patch(
        "ads.model.deployment.model_deployment.ModelDeployment.url",
        return_value=FAKE_MD_URL,
    )
    def test_predict_with_not_active_deployment_fail(
        self, mock_url, mock_client, mock_signer, mock_predict
    ):
        """Ensures predict model fails in case of model deployment is not in an active state."""
        with pytest.raises(NotActiveDeploymentError):
            self.generic_model._as_onnx = False
            self.generic_model.predict(data="test")

        with patch.object(
            ModelDeployment, "state", new_callable=PropertyMock
        ) as mock_state:
            mock_state.return_value = ModelDeploymentState.FAILED
            with pytest.raises(NotActiveDeploymentError):
                self.generic_model.model_deployment = ModelDeployment(
                    model_deployment_id="test"
                )
                self.generic_model._as_onnx = False
                self.generic_model.predict(data="test")

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    @patch(
        "ads.model.deployment.model_deployment.ModelDeployment.url",
        return_value=FAKE_MD_URL,
    )
    def test_predict_bytes_success(self, mock_url, mock_client, mock_signer):
        """Ensures predict model passes with bytes input."""
        with patch.object(
            ModelDeployment, "state", new_callable=PropertyMock
        ) as mock_state:
            mock_state.return_value = ModelDeploymentState.ACTIVE
            with patch.object(ModelDeployment, "predict") as mock_predict:
                mock_predict.return_value = {"result": "result"}
                self.generic_model.model_deployment = ModelDeployment(
                    model_deployment_id="test",
                )
                # self.generic_model.model_deployment.current_state = ModelDeploymentState.ACTIVE
                self.generic_model._as_onnx = False
                byte_data = b"[[1,2,3,4]]"
                test_result = self.generic_model.predict(
                    data=byte_data, auto_serialize_data=True
                )
                # mock_predict.assert_called_with(data=byte_data)
                assert test_result == {"result": "result"}

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    @patch(
        "ads.model.deployment.model_deployment.ModelDeployment.url",
        return_value=FAKE_MD_URL,
    )
    def test_predict_success(self, mock_url, mock_client, mock_signer):
        """Ensures predict model passes with valid input parameters."""
        with patch.object(
            ModelDeployment, "state", new_callable=PropertyMock
        ) as mock_state:
            mock_state.return_value = ModelDeploymentState.ACTIVE
            with patch.object(ModelDeployment, "predict") as mock_predict:
                mock_predict.return_value = {"result": "result"}
                self.generic_model.model_deployment = ModelDeployment(
                    model_deployment_id="test"
                )
                # self.generic_model.model_deployment.current_state = ModelDeploymentState.ACTIVE
                self.generic_model._as_onnx = False
                test_result = self.generic_model.predict(data="test")
                # mock_predict.assert_called_with(
                #     json_input={"data": "test", "data_type": "<class 'str'>"}
                # )
                assert test_result == {"result": "result"}

    # @pytest.mark.skip(reason="need to fix later.")
    @pytest.mark.parametrize(
        "test_args",
        [
            {
                "uri": "/src/folder",
                "artifact_dir": "/dst/folder",
                "model_file_name": "model.pkl",
            },
            {
                "uri": "/src/folder",
                "artifact_dir": "/src/folder",
                "model_file_name": "model.pkl",
            },
            {
                "uri": "/src/folder",
                "artifact_dir": "/dst/folder",
                "model_file_name": "model.pkl",
                "as_onnx": True,
            },
            {
                "uri": "/src/folder",
                "artifact_dir": "/dst/folder",
                "model_file_name": "model.onnx",
            },
            {
                "uri": "src/folder",
                "artifact_dir": "/dst/folder",
                "model_file_name": "model.pkl",
                "force_overwrite": True,
                "auth": {"config": "value"},
            },
            {
                "uri": "src/folder",
                "model_file_name": "model.pkl",
                "force_overwrite": True,
                "auth": {"config": "value"},
            },
            {
                "uri": "src/folder",
                "force_overwrite": True,
                "auth": {"config": "value"},
            },
        ],
    )
    @patch.object(ModelArtifact, "from_uri")
    @patch.object(GenericModel, "reload_runtime_info")
    @patch.object(GenericModel, "_summary_status")
    @patch.object(GenericModel, "__init__", return_value=None)
    @patch("ads.common.auth.default_signer")
    @patch("tempfile.mkdtemp", return_value="tmp_artifact_dir")
    def test_from_model_artifact(
        self,
        mock_mkdtemp,
        mock_default_signer,
        mock_genericmodel_init,
        mock_genericmodel_summary_status,
        mock_reload_runtime_info,
        mock_from_uri,
        test_args,
    ):
        """Tests loading model from a folder, or zip/tar archive."""
        mock_genericmodel_summary_status.return_value = SummaryStatus()
        mock_default_signer.return_value = {"config": "value"}
        mock_artifact_instance = MagicMock(model="test_model")
        mock_from_uri.return_value = mock_artifact_instance
        GenericModel.from_model_artifact(**test_args)

        test_result = mock_from_uri.assert_called_with(
            uri=test_args.get("uri"),
            artifact_dir=test_args.get("artifact_dir", "tmp_artifact_dir"),
            force_overwrite=test_args.get("force_overwrite", False),
            auth=test_args.get("auth", {"config": "value"}),
            model_file_name=test_args.get("model_file_name"),
            ignore_conda_error=False,
        )

        if not test_args.get("as_onnx"):
            mock_genericmodel_init.assert_called_with(
                estimator=mock_artifact_instance.model,
                artifact_dir=test_args.get("artifact_dir", "tmp_artifact_dir"),
                auth=test_args.get("auth", {"config": "value"}),
                properties=ModelProperties(),
            )
        else:
            mock_genericmodel_init.assert_called_with(
                estimator=mock_artifact_instance.model,
                artifact_dir=test_args.get("artifact_dir", "tmp_artifact_dir"),
                auth=test_args.get("auth", {"config": "value"}),
                properties=ModelProperties(),
                as_onnx=True,
            )

        mock_reload_runtime_info.assert_called()
        assert test_result == None

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    @patch(
        "ads.model.deployment.model_deployment.ModelDeployment.url",
        return_value=FAKE_MD_URL,
    )
    def test_predict_success__serialize_input(self, mock_url, mock_client, mock_signer):
        """Ensures predict model passes with valid input parameters."""

        df = pd.DataFrame([1, 2, 3])
        with patch.object(
            ModelDeployment, "state", new_callable=PropertyMock
        ) as mock_state:
            with patch.object(
                GenericModel, "get_data_serializer"
            ) as mock_get_data_serializer:
                mock_get_data_serializer.return_value.data = df.to_json()
                mock_state.return_value = ModelDeploymentState.ACTIVE
                with patch.object(ModelDeployment, "predict") as mock_predict:
                    mock_predict.return_value = {"result": "result"}
                    self.generic_model.model_deployment = ModelDeployment(
                        model_deployment_id="test"
                    )
                    self.generic_model._as_onnx = False
                    test_result = self.generic_model.predict(
                        data="test", auto_serialize_data=True
                    )
                    # mock_predict.assert_called_with(json_input=df.to_json())
                    assert test_result == {"result": "result"}

    @pytest.mark.parametrize(
        "test_args",
        [
            {
                "model_id": "test_model_id",
                "model_file_name": "test_model_file_name",
                "artifact_dir": "/test_artifact_dir",
                "bucket_uri": "test_bucket_uri",
                "remove_existing_artifact": False,
                "ignore_conda_error": True,
            },
            {
                "model_id": "test_model_id",
                "model_file_name": "test_model_file_name",
                "artifact_dir": "/test_artifact_dir",
                "auth": {"config": "value"},
                "force_overwrite": True,
                "compartment_id": "test_compartment_id",
                "timeout": 100,
                "ignore_conda_error": False,
            },
        ],
    )
    @patch.object(GenericModel, "from_model_artifact")
    @patch("ads.common.auth.default_signer")
    @patch("ads.model.generic_model.GenericModel.__init__")
    @patch.object(DataScienceModel, "download_artifact")
    @patch.object(DataScienceModel, "from_id")
    def test_from_model_catalog(
        self,
        mock_dsc_model_from_id,
        mock_download_artifact,
        mock_genericmodel_init,
        mock_default_signer,
        mock_from_model_artifact,
        test_args,
    ):
        """Tests loading model from model catalog."""
        mock_default_signer.return_value = {"config": "value"}
        mock_genericmodel_init.return_value = None
        mock_dsc_model_from_id.return_value = self.mock_dsc_model

        mock_from_model_artifact.return_value = self.generic_model

        test_result = GenericModel.from_model_catalog(**test_args)

        mock_download_artifact.assert_called_with(
            target_dir=test_args.get("artifact_dir"),
            force_overwrite=test_args.get("force_overwrite", False),
            bucket_uri=test_args.get("bucket_uri"),
            remove_existing_artifact=test_args.get("remove_existing_artifact", True),
            auth={"config": "value"},
            region=None,
            timeout=test_args.get("timeout"),
        )

        expected_model_properties = ModelProperties(
            compartment_id=test_args.get("compartment_id", _COMPARTMENT_OCID),
            bucket_uri=test_args.get("bucket_uri"),
            remove_existing_artifact=test_args.get("remove_existing_artifact", True),
        )

        if not test_args.get("compartment_id"):
            mock_from_model_artifact.assert_called_with(
                uri=test_args.get("artifact_dir"),
                model_file_name=test_args.get("model_file_name"),
                artifact_dir=test_args.get("artifact_dir"),
                auth=test_args.get("auth", {"config": "value"}),
                force_overwrite=test_args.get("force_overwrite", False),
                properties=expected_model_properties,
                ignore_conda_error=test_args.get("ignore_conda_error", False),
            )
        else:
            mock_from_model_artifact.assert_called_with(
                uri=test_args.get("artifact_dir"),
                model_file_name=test_args.get("model_file_name"),
                artifact_dir=test_args.get("artifact_dir"),
                auth=test_args.get("auth", {"config": "value"}),
                force_overwrite=test_args.get("force_overwrite", False),
                properties=expected_model_properties,
                ignore_conda_error=test_args.get("ignore_conda_error", False),
                compartment_id=test_args.get("compartment_id"),
            )

        assert (
            test_result.metadata_taxonomy == self.mock_dsc_model.defined_metadata_list
        )
        assert test_result.metadata_custom == self.mock_dsc_model.custom_metadata_list
        assert test_result.schema_input == self.mock_dsc_model.input_schema
        assert test_result.schema_output == self.mock_dsc_model.output_schema
        assert (
            test_result.metadata_provenance == self.mock_dsc_model.provenance_metadata
        )

    @patch.object(SummaryStatus, "update_status")
    @patch.object(
        ModelDeployment,
        "state",
        new_callable=PropertyMock,
        return_value=ModelDeploymentState.ACTIVE,
    )
    @patch.object(GenericModel, "from_model_catalog")
    @patch.object(ModelDeployment, "from_id")
    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_from_model_deployment(
        self,
        mock_client,
        mock_default_signer,
        mock_from_id,
        mock_from_model_catalog,
        mock_model_deployment_state,
        mock_update_status,
    ):
        """Tests loading model from model deployment."""
        test_auth_config = {"signer": {"config": "value"}}
        mock_default_signer.return_value = test_auth_config
        test_model_deployment_id = "md_ocid"
        test_model_id = "model_ocid"
        md_props = ModelDeploymentProperties(model_id=test_model_id)
        md = ModelDeployment(properties=md_props)
        mock_from_id.return_value = md

        test_model = MagicMock(model_deployment=md, _summary_status=SummaryStatus())
        mock_from_model_catalog.return_value = test_model

        test_result = GenericModel.from_model_deployment(
            model_deployment_id=test_model_deployment_id,
            model_file_name="test.pkl",
            artifact_dir="test_dir",
            auth=test_auth_config,
            force_overwrite=True,
            properties=None,
            bucket_uri="test_bucket_uri",
            remove_existing_artifact=True,
            compartment_id="test_compartment_id",
        )

        mock_from_id.assert_called_with(
            test_model_deployment_id
        )
        mock_from_model_catalog.assert_called_with(
            model_id=test_model_id,
            model_file_name="test.pkl",
            artifact_dir="test_dir",
            auth=test_auth_config,
            force_overwrite=True,
            properties=None,
            bucket_uri="test_bucket_uri",
            remove_existing_artifact=True,
            compartment_id="test_compartment_id",
            ignore_conda_error=False,
        )

        assert test_result == test_model

    @patch.object(
        ModelDeployment,
        "state",
        new_callable=PropertyMock,
        return_value=ModelDeploymentState.FAILED,
    )
    @patch.object(ModelDeployment, "from_id")
    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_from_model_deployment_fail(
        self,
        mock_client,
        mock_default_signer,
        mock_from_id,
        mock_model_deployment_state,
    ):
        """Tests loading model from model deployment."""
        test_auth_config = {"signer": {"config": "value"}}
        mock_default_signer.return_value = test_auth_config
        test_model_deployment_id = "md_ocid"
        test_model_id = "model_ocid"
        md_props = ModelDeploymentProperties(model_id=test_model_id)
        md = ModelDeployment(properties=md_props)
        mock_from_id.return_value = md

        with pytest.raises(NotActiveDeploymentError):
            GenericModel.from_model_deployment(
                model_deployment_id=test_model_deployment_id,
                model_file_name="test.pkl",
                artifact_dir="test_dir",
                auth=test_auth_config,
                force_overwrite=True,
                properties=None,
                bucket_uri="test_bucket_uri",
                remove_existing_artifact=True,
                compartment_id="test_compartment_id",
            )
            mock_from_id.assert_called_with(
                test_model_deployment_id
            )

    @patch.object(ModelDeployment, "update")
    @patch.object(ModelDeployment, "from_id")
    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_update_deployment_class_level(
        self, mock_client, mock_signer, mock_from_id, mock_update
    ):
        test_model_deployment_id = "xxxx.datasciencemodeldeployment.xxxx"
        md_props = ModelDeploymentProperties(model_id=test_model_deployment_id)
        md = ModelDeployment(properties=md_props)
        mock_from_id.return_value = md

        test_model = MagicMock(model_deployment=md, _summary_status=SummaryStatus())
        mock_update.return_value = test_model

        with pytest.raises(
            ValueError, match="Parameter `model_deployment_id` must be provided."
        ):
            GenericModel.update_deployment(
                properties=None,
                wait_for_completion=True,
                max_wait_time=100,
                poll_interval=200,
            )

        test_result = GenericModel.update_deployment(
            test_model_deployment_id,
            properties=None,
            wait_for_completion=True,
            max_wait_time=100,
            poll_interval=200,
        )

        mock_from_id.assert_called_with(
            test_model_deployment_id
        )

        mock_update.assert_called_with(
            properties=None,
            wait_for_completion=True,
            max_wait_time=100,
            poll_interval=200,
        )

        assert test_result == test_model

    @patch.object(ModelDeployment, "update")
    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_update_deployment_instance_level_with_id(
        self, mock_client, mock_signer, mock_update
    ):
        test_model_deployment_id = "xxxx.datasciencemodeldeployment.xxxx"
        md_props = ModelDeploymentProperties(model_id=test_model_deployment_id)
        md = ModelDeployment(properties=md_props)

        test_model = MagicMock(model_deployment=md, _summary_status=SummaryStatus())
        mock_update.return_value = test_model

        generic_model = GenericModel(estimator=TestEstimator())
        test_result = generic_model.update_deployment(
            model_deployment_id=test_model_deployment_id,
            properties=None,
            wait_for_completion=True,
            max_wait_time=100,
            poll_interval=200,
        )

        mock_update.assert_called_with(
            properties=None,
            wait_for_completion=True,
            max_wait_time=100,
            poll_interval=200,
        )

        assert test_result == test_model

    @patch.object(GenericModel, "from_model_deployment")
    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_from_id_model_deployment(
        self, mock_client, mock_signer, mock_from_model_deployment
    ):
        test_model_deployment_id = "xxxx.datasciencemodeldeployment.xxxx"
        md_props = ModelDeploymentProperties(model_id=test_model_deployment_id)
        md = ModelDeployment(properties=md_props)

        test_model = MagicMock(model_deployment=md, _summary_status=SummaryStatus())
        mock_from_model_deployment.return_value = test_model

        test_model_deployment_result = GenericModel.from_id(
            ocid=test_model_deployment_id,
            model_file_name="test.pkl",
            artifact_dir="test_dir",
            auth=None,
            force_overwrite=True,
            properties=None,
            bucket_uri="test_bucket_uri",
            remove_existing_artifact=True,
            compartment_id="test_compartment_id",
            ignore_conda_error=True,
        )

        mock_from_model_deployment.assert_called_with(
            test_model_deployment_id,
            model_file_name="test.pkl",
            artifact_dir="test_dir",
            auth=None,
            force_overwrite=True,
            properties=None,
            bucket_uri="test_bucket_uri",
            remove_existing_artifact=True,
            compartment_id="test_compartment_id",
            ignore_conda_error=True,
        )

        assert test_model_deployment_result == test_model

    @patch.object(GenericModel, "from_model_catalog")
    def test_from_id_model(self, mock_from_model_catalog):
        test_model_id = "xxxx.datasciencemodel.xxxx"
        mock_oci_model = MagicMock(
            metadata_custom="test_metadata_custom",
            metadata_taxonomy="test_metadata_taxonomy",
            schema_input="test_schema_input",
            schema_output="test_schema_output",
            provenance_metadata=ModelProvenance(
                repository_url="test_rep_url",
                git_branch="test_git_branch",
                git_commit="test_git_commit",
                script_dir="test_script_dir",
                training_script="test_training_script",
                training_id="test_training_id",
            ),
        )

        mock_from_model_catalog.return_value = mock_oci_model

        test_model_result = GenericModel.from_id(
            ocid=test_model_id,
            model_file_name="test.pkl",
            artifact_dir="test_dir",
            auth=None,
            force_overwrite=True,
            properties=None,
            bucket_uri="test_bucket_uri",
            remove_existing_artifact=True,
            compartment_id="test_compartment_id",
            ignore_conda_error=True,
        )

        mock_from_model_catalog.assert_called_with(
            test_model_id,
            model_file_name="test.pkl",
            artifact_dir="test_dir",
            auth=None,
            force_overwrite=True,
            properties=None,
            bucket_uri="test_bucket_uri",
            remove_existing_artifact=True,
            compartment_id="test_compartment_id",
            ignore_conda_error=True,
        )

        assert test_model_result == mock_oci_model

    def test_from_id_fail(self):
        test_id = "test_invalid_ocid"

        with pytest.raises(
            ValueError,
            match="Invalid OCID: test_invalid_ocid. Please provide valid model OCID or model deployment OCID.",
        ):
            GenericModel.from_id(
                test_id,
                model_file_name="test.pkl",
                artifact_dir="test_dir",
                auth=None,
                force_overwrite=True,
                properties=None,
                bucket_uri="test_bucket_uri",
                remove_existing_artifact=True,
                compartment_id="test_compartment_id",
                ignore_conda_error=True,
            )

    @patch.object(ModelDeployment, "activate")
    @patch.object(ModelDeployment, "deactivate")
    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_restart_deployment(
        self, mock_client, mock_signer, mock_deactivate, mock_activate
    ):
        test_model_deployment_id = "xxxx.datasciencemodeldeployment.xxxx"
        md_props = ModelDeploymentProperties(model_id=test_model_deployment_id)
        md = ModelDeployment(properties=md_props)
        generic_model = GenericModel(estimator=TestEstimator())
        generic_model.model_deployment = md
        mock_deactivate.return_value = md
        mock_activate.return_value = md
        generic_model.restart_deployment(max_wait_time=2000, poll_interval=50)
        mock_deactivate.assert_called_with(max_wait_time=2000, poll_interval=50)
        mock_activate.assert_called_with(max_wait_time=2000, poll_interval=50)

    def test__to_dict_not_prepared(self):
        dictionary = self.generic_model._to_dict()
        for key in _ATTRIBUTES_TO_SHOW_:
            assert key in dictionary
        assert dictionary["model_deployment_id"] is None
        assert dictionary["model_id"] is None
        assert dictionary["artifact_dir"] is not None

    @patch("ads.common.auth.default_signer")
    def test__to_dict_prepared(self, moxk_signer):
        self.generic_model.prepare(
            "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        )
        dictionary = self.generic_model._to_dict()
        for key in _ATTRIBUTES_TO_SHOW_:
            assert key in dictionary

        assert dictionary["model_deployment_id"] is None
        assert dictionary["artifact_dir"][self.generic_model.artifact_dir] is not None

    def test__to_yaml(self):
        dictionary = self.generic_model._to_dict()
        assert self.generic_model._to_yaml() == yaml.dump(dictionary, Dumper=dumper)

    @pytest.mark.parametrize(
        "test_args, expected_prepare, expected_save, expected_deploy",
        [
            (
                {
                    "model_display_name": "fake_display_name",
                    "deployment_display_name": "fake_deployment_display_name",
                    "ignore_conda_error": False,
                },
                {
                    "inference_conda_env": None,
                    "inference_python_version": None,
                    "training_conda_env": None,
                    "training_python_version": None,
                    "model_file_name": None,
                    "as_onnx": False,
                    "initial_types": None,
                    "force_overwrite": False,
                    "namespace": "id19sfcrra6z",
                    "use_case_type": None,
                    "X_sample": None,
                    "y_sample": None,
                    "training_script_path": None,
                    "training_id": None,
                    "ignore_pending_changes": True,
                    "ignore_conda_error": False,
                    "max_col_num": 2000,
                    "impute_values": None,
                },
                {
                    "display_name": "fake_display_name",
                    "description": None,
                    "freeform_tags": None,
                    "defined_tags": None,
                    "ignore_introspection": False,
                    "compartment_id": os.getenv("NB_SESSION_COMPARTMENT_OCID", None),
                    "project_id": os.getenv("PROJECT_OCID", None),
                    "timeout": None,
                    "bucket_uri": None,
                    "overwrite_existing_artifact": True,
                    "remove_existing_artifact": True,
                    "region": None,
                    "version_label": None,
                    "model_version_set": None,
                },
                {
                    "wait_for_completion": True,
                    "display_name": "fake_deployment_display_name",
                    "description": None,
                    "deployment_instance_shape": None,
                    "deployment_instance_subnet_id": None,
                    "deployment_instance_count": None,
                    "deployment_bandwidth_mbps": None,
                    "deployment_log_group_id": None,
                    "deployment_access_log_id": None,
                    "deployment_predict_log_id": None,
                    "deployment_memory_in_gbs": None,
                    "deployment_ocpus": None,
                    "deployment_image": None,
                    "kwargs": {},
                },
            ),
            (
                {
                    "inference_conda_env": None,
                    "inference_python_version": None,
                    "training_conda_env": None,
                    "training_python_version": None,
                    "model_file_name": None,
                    "as_onnx": False,
                    "initial_types": None,
                    "force_overwrite": False,
                    "namespace": "id19sfcrra6z",
                    "use_case_type": None,
                    "X_sample": None,
                    "y_sample": None,
                    "training_script_path": None,
                    "training_id": None,
                    "ignore_pending_changes": True,
                    "ignore_conda_error": False,
                    "max_col_num": 2000,
                    "model_display_name": "fake_display_name",
                    "model_description": None,
                    "model_freeform_tags": None,
                    "model_defined_tags": None,
                    "ignore_introspection": False,
                    "wait_for_completion": True,
                    "deployment_display_name": "fake_deployment_display_name",
                    "deployment_description": None,
                    "deployment_instance_shape": None,
                    "deployment_instance_subnet_id": None,
                    "deployment_instance_count": None,
                    "deployment_bandwidth_mbps": None,
                    "deployment_log_group_id": None,
                    "deployment_access_log_id": None,
                    "deployment_predict_log_id": None,
                    "deployment_memory_in_gbs": None,
                    "deployment_ocpus": None,
                    "impute_values": None,
                },
                {
                    "inference_conda_env": None,
                    "inference_python_version": None,
                    "training_conda_env": None,
                    "training_python_version": None,
                    "model_file_name": None,
                    "as_onnx": False,
                    "initial_types": None,
                    "force_overwrite": False,
                    "namespace": "id19sfcrra6z",
                    "use_case_type": None,
                    "X_sample": None,
                    "y_sample": None,
                    "training_script_path": None,
                    "training_id": None,
                    "ignore_pending_changes": True,
                    "ignore_conda_error": False,
                    "max_col_num": 2000,
                    "impute_values": None,
                },
                {
                    "display_name": "fake_display_name",
                    "description": None,
                    "freeform_tags": None,
                    "defined_tags": None,
                    "ignore_introspection": False,
                    "compartment_id": os.getenv("NB_SESSION_COMPARTMENT_OCID", None),
                    "project_id": os.getenv("PROJECT_OCID", None),
                    "timeout": None,
                    "bucket_uri": None,
                    "overwrite_existing_artifact": True,
                    "remove_existing_artifact": True,
                    "model_version_set": None,
                    "version_label": None,
                    "region": None,
                },
                {
                    "wait_for_completion": True,
                    "display_name": "fake_deployment_display_name",
                    "description": None,
                    "deployment_instance_shape": None,
                    "deployment_instance_subnet_id": None,
                    "deployment_instance_count": None,
                    "deployment_bandwidth_mbps": None,
                    "deployment_log_group_id": None,
                    "deployment_access_log_id": None,
                    "deployment_predict_log_id": None,
                    "deployment_memory_in_gbs": None,
                    "deployment_ocpus": None,
                    "deployment_image": None,
                    "kwargs": {},
                },
            ),
            (
                {
                    "inference_conda_env": "fake_inference_conda_env",
                    "inference_python_version": "fake_inference_python_version",
                    "training_conda_env": "fake_training_conda_env",
                    "training_python_version": "fake_training_python_version",
                    "model_file_name": "fake_model_file_name",
                    "as_onnx": True,
                    "initial_types": None,
                    "force_overwrite": False,
                    "namespace": "fake_namespace",
                    "use_case_type": "fake_use_case_type",
                    "X_sample": None,
                    "y_sample": None,
                    "training_script_path": None,
                    "training_id": None,
                    "ignore_pending_changes": True,
                    "ignore_conda_error": False,
                    "max_col_num": 3000,
                    "model_display_name": "fake_display_name",
                    "model_description": "fake_description",
                    "model_freeform_tags": {"fake_key": "fake_vale"},
                    "model_defined_tags": {"fake_key": "fake_vale"},
                    "ignore_introspection": True,
                    "wait_for_completion": True,
                    "deployment_display_name": "fake_deployment_display_name",
                    "deployment_description": "fake_deployment_description",
                    "deployment_instance_shape": "2.1",
                    "deployment_instance_subnet_id": "ocid1.subnet.oc1.iad.<unique_ocid>",
                    "deployment_instance_count": 1,
                    "deployment_bandwidth_mbps": 10,
                    "deployment_log_group_id": "ocid1.loggroup.oc1.iad.<unique_ocid>",
                    "deployment_access_log_id": "ocid1.log.oc1.iad.<unique_ocid>",
                    "deployment_predict_log_id": "ocid1.log.oc1.iad.<unique_ocid>",
                    "deployment_memory_in_gbs": 10,
                    "deployment_ocpus": 1,
                    "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
                    "project_id": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
                    "timeout": 10,
                    "freeform_tags": {"test": "value"},
                    "impute_values": None,
                },
                {
                    "inference_conda_env": "fake_inference_conda_env",
                    "inference_python_version": "fake_inference_python_version",
                    "training_conda_env": "fake_training_conda_env",
                    "training_python_version": "fake_training_python_version",
                    "model_file_name": "fake_model_file_name",
                    "as_onnx": True,
                    "initial_types": None,
                    "force_overwrite": False,
                    "namespace": "fake_namespace",
                    "use_case_type": "fake_use_case_type",
                    "X_sample": None,
                    "y_sample": None,
                    "training_script_path": None,
                    "training_id": None,
                    "ignore_pending_changes": True,
                    "ignore_conda_error": False,
                    "max_col_num": 3000,
                    "impute_values": None,
                },
                {
                    "display_name": "fake_display_name",
                    "description": "fake_description",
                    "freeform_tags": {"fake_key": "fake_vale"},
                    "defined_tags": {"fake_key": "fake_vale"},
                    "ignore_introspection": True,
                    "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
                    "project_id": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
                    "timeout": 10,
                    "bucket_uri": None,
                    "overwrite_existing_artifact": True,
                    "remove_existing_artifact": True,
                    "model_version_set": None,
                    "version_label": None,
                    "region": None,
                },
                {
                    "wait_for_completion": True,
                    "display_name": "fake_deployment_display_name",
                    "description": "fake_deployment_description",
                    "deployment_instance_shape": "2.1",
                    "deployment_instance_count": 1,
                    "deployment_bandwidth_mbps": 10,
                    "deployment_instance_subnet_id": "ocid1.subnet.oc1.iad.<unique_ocid>",
                    "deployment_log_group_id": "ocid1.loggroup.oc1.iad.<unique_ocid>",
                    "deployment_access_log_id": "ocid1.log.oc1.iad.<unique_ocid>",
                    "deployment_predict_log_id": "ocid1.log.oc1.iad.<unique_ocid>",
                    "deployment_memory_in_gbs": 10,
                    "deployment_ocpus": 1,
                    "deployment_image": None,
                    "kwargs": {
                        "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
                        "project_id": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
                        "freeform_tags": {"test": "value"},
                    },
                },
            ),
            (
                {
                    "inference_conda_env": None,
                    "inference_python_version": None,
                    "training_conda_env": None,
                    "training_python_version": None,
                    "model_file_name": "fake_model_file_name",
                    "as_onnx": True,
                    "initial_types": None,
                    "force_overwrite": False,
                    "namespace": "fake_namespace",
                    "use_case_type": "fake_use_case_type",
                    "X_sample": None,
                    "y_sample": None,
                    "training_script_path": None,
                    "training_id": None,
                    "ignore_pending_changes": True,
                    "ignore_conda_error": True,
                    "max_col_num": 3000,
                    "model_display_name": "fake_display_name",
                    "model_description": "fake_description",
                    "model_freeform_tags": {"fake_key": "fake_vale"},
                    "model_defined_tags": {"fake_key": "fake_vale"},
                    "ignore_introspection": True,
                    "wait_for_completion": True,
                    "deployment_display_name": "fake_deployment_display_name",
                    "deployment_description": "fake_deployment_description",
                    "deployment_instance_shape": "2.1",
                    "deployment_instance_subnet_id": "ocid1.subnet.oc1.iad.<unique_ocid>",
                    "deployment_instance_count": 1,
                    "deployment_bandwidth_mbps": 10,
                    "deployment_log_group_id": "ocid",
                    "deployment_access_log_id": "ocid",
                    "deployment_predict_log_id": "ocid",
                    "deployment_memory_in_gbs": 10,
                    "deployment_ocpus": 1,
                    "deployment_image": "test_docker_image",
                    "cmd": ["test_cmd"],
                    "entrypoint": ["test_entrypoint"],
                    "server_port": 8080,
                    "health_check_port": 8080,
                    "environment_variables": {"test_key": "test_value"},
                    "compartment_id": "ocid..",
                    "project_id": "ocid..",
                    "timeout": 10,
                    "freeform_tags": {"test": "value"},
                    "impute_values": None,
                },
                {
                    "inference_conda_env": None,
                    "inference_python_version": None,
                    "training_conda_env": None,
                    "training_python_version": None,
                    "model_file_name": "fake_model_file_name",
                    "as_onnx": True,
                    "initial_types": None,
                    "force_overwrite": False,
                    "namespace": "fake_namespace",
                    "use_case_type": "fake_use_case_type",
                    "X_sample": None,
                    "y_sample": None,
                    "training_script_path": None,
                    "training_id": None,
                    "ignore_pending_changes": True,
                    "ignore_conda_error": True,
                    "max_col_num": 3000,
                    "impute_values": None,
                },
                {
                    "display_name": "fake_display_name",
                    "description": "fake_description",
                    "freeform_tags": {"fake_key": "fake_vale"},
                    "defined_tags": {"fake_key": "fake_vale"},
                    "ignore_introspection": True,
                    "compartment_id": "ocid..",
                    "project_id": "ocid..",
                    "timeout": 10,
                    "bucket_uri": None,
                    "overwrite_existing_artifact": True,
                    "remove_existing_artifact": True,
                    "model_version_set": None,
                    "version_label": None,
                    "region": None,
                },
                {
                    "wait_for_completion": True,
                    "display_name": "fake_deployment_display_name",
                    "description": "fake_deployment_description",
                    "deployment_instance_shape": "2.1",
                    "deployment_instance_count": 1,
                    "deployment_bandwidth_mbps": 10,
                    "deployment_instance_subnet_id": "ocid1.subnet.oc1.iad.<unique_ocid>",
                    "deployment_log_group_id": "ocid",
                    "deployment_access_log_id": "ocid",
                    "deployment_predict_log_id": "ocid",
                    "deployment_memory_in_gbs": 10,
                    "deployment_ocpus": 1,
                    "deployment_image": "test_docker_image",
                    "kwargs": {
                        "compartment_id": "ocid..",
                        "project_id": "ocid..",
                        "freeform_tags": {"test": "value"},
                        "cmd": ["test_cmd"],
                        "entrypoint": ["test_entrypoint"],
                        "server_port": 8080,
                        "health_check_port": 8080,
                        "environment_variables": {"test_key": "test_value"},
                    },
                },
            ),
        ],
    )
    @patch.object(GenericModel, "prepare")
    @patch.object(GenericModel, "save")
    @patch.object(GenericModel, "deploy")
    def test_prepare_save_deploy(
        self,
        mock_deploy,
        mock_save,
        mock_prepare,
        test_args,
        expected_prepare,
        expected_save,
        expected_deploy,
    ):
        self.generic_model.prepare_save_deploy(**test_args)
        mock_prepare.assert_called_with(**expected_prepare)
        mock_save.assert_called_with(**expected_save)
        mock_deploy.assert_called_with(**expected_deploy)

    @patch.object(GenericModel, "prepare")
    @patch.object(GenericModel, "save")
    @patch.object(GenericModel, "deploy")
    def test_prepare_save_deploy_with_default_display_name(
        self,
        mock_deploy,
        mock_save,
        mock_prepare,
    ):
        """Validate that prepare_sve_deploy() with no display name specified for model and model deployment will have generated names for this resources."""
        random.seed(self.random_seed)
        expected_save = {
            "display_name": utils.get_random_name_for_resource(),
            "description": None,
            "freeform_tags": None,
            "defined_tags": None,
            "ignore_introspection": False,
            "compartment_id": os.getenv("NB_SESSION_COMPARTMENT_OCID", None),
            "project_id": os.getenv("PROJECT_OCID", None),
            "timeout": None,
            "bucket_uri": None,
            "overwrite_existing_artifact": True,
            "remove_existing_artifact": True,
            "version_label": None,
            "model_version_set": None,
            "region": None,
        }
        expected_deploy = {
            "wait_for_completion": True,
            "display_name": utils.get_random_name_for_resource(),
            "description": None,
            "deployment_instance_shape": None,
            "deployment_instance_subnet_id": None,
            "deployment_instance_count": None,
            "deployment_bandwidth_mbps": None,
            "deployment_memory_in_gbs": None,
            "deployment_ocpus": None,
            "deployment_log_group_id": None,
            "deployment_access_log_id": None,
            "deployment_predict_log_id": None,
            "deployment_image": None,
            "kwargs": {},
        }
        random.seed(self.random_seed)
        self.generic_model.prepare_save_deploy()
        mock_save.assert_called_with(**expected_save)
        mock_deploy.assert_called_with(**expected_deploy)

    @pytest.mark.parametrize(
        "test_args",
        [
            {
                "model_id": "test_model_id",
                "delete_associated_model_deployment": False,
                "delete_model_artifact": False,
                "artifact_dir": "test_artifact_dir",
            },
            {
                "model_id": "test_model_id",
                "delete_associated_model_deployment": True,
                "auth": {"config": "value"},
                "timeout": 100,
                "compartment_id": "test_compartment_id",
                "delete_model_artifact": False,
                "artifact_dir": "test_artifact_dir",
            },
        ],
    )
    @patch("ads.common.auth.default_signer")
    @patch("shutil.rmtree")
    @patch.object(DataScienceModel, "from_id")
    def test_delete_cls(
        self,
        mock_dsc_model_from_id,
        mock_rmtree,
        mock_default_signer,
        test_args,
    ):
        """Tests deleting model from Model Catalog."""
        mock_dsc_model_delete = MagicMock()
        mock_dsc_model_from_id.return_value = MagicMock(delete=mock_dsc_model_delete)
        mock_default_signer.return_value = {"config": "value"}
        GenericModel.delete(**test_args)

        mock_dsc_model_delete.assert_called_with(
            delete_associated_model_deployment=test_args.get(
                "delete_associated_model_deployment"
            ),
        )

        if test_args.get("delete_model_artifact"):
            mock_rmtree.assert_called_with(
                test_args.get("artifact_dir"), ignore_errors=True
            )

    @pytest.mark.parametrize(
        "test_args",
        [
            {
                "delete_associated_model_deployment": False,
                "delete_model_artifact": False,
                "artifact_dir": "test_artifact_dir",
            },
            {
                "delete_associated_model_deployment": False,
                "delete_model_artifact": True,
            },
            {
                "model_id": "test_model_id",
                "delete_associated_model_deployment": True,
                "auth": {"config": "value"},
                "timeout": 100,
                "compartment_id": "test_compartment_id",
                "delete_model_artifact": False,
                "artifact_dir": "test_artifact_dir",
            },
        ],
    )
    @patch("ads.common.auth.default_signer")
    @patch("shutil.rmtree")
    @patch.object(DataScienceModel, "from_id")
    def test_delete_instance(
        self,
        mock_dsc_model_from_id,
        mock_rmtree,
        mock_default_signer,
        test_args,
    ):
        """Tests deleting model from Model Catalog."""
        mock_dsc_model_delete = MagicMock()
        mock_dsc_model_from_id.return_value = MagicMock(
            id="test_model", delete=mock_dsc_model_delete
        )
        mock_default_signer.return_value = {"config": "value"}
        self.generic_model.dsc_model = MagicMock(id="test_model")

        self.generic_model.delete(**test_args)
        mock_dsc_model_delete.assert_called_with(
            delete_associated_model_deployment=test_args.get(
                "delete_associated_model_deployment"
            ),
        )

        if test_args.get("delete_model_artifact"):
            mock_rmtree.assert_called_with(
                test_args.get("artifact_dir") or self.generic_model.artifact_dir,
                ignore_errors=True,
            )

    def test_delete_fail(self):
        """Ensures that deleting model fails in case of wrong input parameters."""
        with pytest.raises(ValueError, match="The `model_id` must be provided."):
            GenericModel.delete()
        with pytest.raises(ValueError, match="The `artifact_dir` must be provided."):
            GenericModel.delete(model_id="test_model_id", delete_model_artifact=True)

    @patch("ads.common.utils.get_random_name_for_resource", return_value="test_name")
    def test_random_display_name(self, mock_get_random_name_for_resource):
        """Ensures the random disply name for model can be successfully generated."""
        self.generic_model._PREFIX = "test_prefix"
        assert self.generic_model._random_display_name() == "test_prefix-test_name"

    def test_upload_artifact_fail(self):
        """Ensures uploading model artifacts to the provided `uri` fails in case of wrong input data."""
        with pytest.raises(ValueError, match="The `uri` must be provided."):
            self.generic_model.upload_artifact(uri="", auth={"config": "value"})
        with pytest.raises(ValueError, match=r"The model artifacts not found.*"):
            self.generic_model.artifact_dir = None
            self.generic_model.upload_artifact(uri="test_uri", auth={"config": "value"})

    def test_upload_artifact_success(self):
        """Tests uploading model artifacts to the provided `uri`."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # copy test artifacts to the temp folder
            shutil.copytree(
                os.path.join(self.curr_dir, "test_files/valid_model_artifacts"),
                os.path.join(tmp_dir, "model_artifacts"),
            )
            self.generic_model.artifact_dir = os.path.join(tmp_dir, "model_artifacts/")
            self.generic_model.dsc_model = MagicMock(id="test_model_id")
            expected_artifact_path = os.path.join(
                tmp_dir, f"{self.generic_model.model_id}.zip"
            )

            self.generic_model.upload_artifact(
                uri=tmp_dir + "/", auth={"config": "value"}
            )

            assert os.path.exists(expected_artifact_path)

            target_dir = os.path.join(tmp_dir, "model_artifacts_new/")
            with ZipFile(expected_artifact_path) as zip_file:
                zip_file.extractall(target_dir)

            test_files = list(
                glob.iglob(os.path.join(target_dir, "**"), recursive=True)
            )
            expected_files = [
                os.path.join(target_dir, file_name)
                for file_name in ["", "runtime.yaml", "score.py"]
            ]
            assert sorted(test_files) == sorted(expected_files)

    def test_update_fail(self):
        """Ensures that saving model metadata fails in case of the wrong input attributes."""
        with pytest.raises(ValueError):
            self.generic_model.update()

    @patch.object(DataScienceModel, "update")
    def test_update(
        self,
        mock_dsc_update,
    ):
        """Tests updating model metadata in the Model Catalog."""

        # Prepare DataScienceModel
        mock_oci_dsc_model = OCIDataScienceModel(**OCI_MODEL_PAYLOAD)
        mock_oci_dsc_model.id = "model_id"
        self.mock_dsc_model.dsc_model = mock_oci_dsc_model

        # Prepare GenericModel
        self.mock_dsc_model._spec["id"] = "model_id"
        self.generic_model.dsc_model = self.mock_dsc_model

        # Modify GenericModel
        self.generic_model.metadata_custom.get(
            "CondaEnvironment"
        ).value = "test_conda_environment"
        self.generic_model.metadata_taxonomy.get(
            "Algorithm"
        ).value = "new_algorithm_value"
        self.generic_model.metadata_provenance.git_branch = "develop"

        test_name = "New Model Name"
        test_description = "New Model Description"
        test_freeform_tags = {"tag": "value"}
        test_defined_tags = {"tag": {"key": "value"}}

        # Call GenericModel.update
        self.generic_model.update(
            display_name=test_name,
            description=test_description,
            freeform_tags=test_freeform_tags,
            defined_tags=test_defined_tags,
        )

        mock_dsc_update.assert_called()

        self.generic_model.dsc_model.metadata_custom == self.mock_dsc_model.custom_metadata_list
        self.generic_model.dsc_model.metadata_taxonomy == self.mock_dsc_model.defined_metadata_list
        self.generic_model.dsc_model.schema_input == self.mock_dsc_model.input_schema
        self.generic_model.dsc_model.schema_output == self.mock_dsc_model.output_schema
        self.generic_model.dsc_model.metadata_provenance == self.mock_dsc_model.provenance_metadata

    @patch.object(ModelDeployment, "deploy")
    def test_deploy_combined_initialization(self, mock_deploy):
        self.generic_model.properties = ModelProperties(
            deployment_image="default_test_docker_image",
            compartment_id="default_test_compartment_id",
            project_id="default_test_project_id",
        )
        test_model_id = "ocid.test_model_id"
        self.generic_model.dsc_model = MagicMock(id=test_model_id)
        self.generic_model.ignore_conda_error = True
        infrastructure = ModelDeploymentInfrastructure(
            **{
                "shape_name": "test_deployment_instance_shape",
                "replica": 10,
                "bandwidth_mbps": 100,
                "shape_config_details": {"memory_in_gbs": 10, "ocpus": 1},
                "access_log": {
                    "log_group_id": "test_deployment_log_group_id",
                    "log_id": "test_deployment_access_log_id",
                },
                "predict_log": {
                    "log_group_id": "test_deployment_log_group_id",
                    "log_id": "test_deployment_predict_log_id",
                },
                "project_id": "project_id_passed_using_with",
                "compartment_id": "compartment_id_passed_using_with",
            }
        )
        runtime = ModelDeploymentContainerRuntime(
            **{
                "image": "image_passed_using_with",
                "image_digest": "test_image_digest",
                "cmd": ["test_cmd"],
                "entrypoint": ["test_entrypoint"],
                "server_port": 8080,
                "health_check_port": 8080,
                "env": {"test_key": "test_value"},
                "deployment_mode": "HTTPS_ONLY",
            }
        )
        mock_deploy.return_value = ModelDeployment(
            display_name="test_display_name",
            description="test_description",
            infrastructure=infrastructure,
            runtime=runtime,
            model_deployment_url="test_model_deployment_url",
            model_deployment_id="test_model_deployment_id",
        )
        input_dict = {
            "wait_for_completion": True,
            "display_name": "test_display_name",
            "description": "test_description",
            "deployment_instance_shape": "test_deployment_instance_shape",
            "deployment_instance_count": 10,
            "deployment_bandwidth_mbps": 100,
            "deployment_memory_in_gbs": 10,
            "deployment_ocpus": 1,
            "deployment_log_group_id": "test_deployment_log_group_id",
            "deployment_access_log_id": "test_deployment_access_log_id",
            "deployment_predict_log_id": "test_deployment_predict_log_id",
            "cmd": ["test_cmd"],
            "entrypoint": ["test_entrypoint"],
            "server_port": 8080,
            "health_check_port": 8080,
            "environment_variables": {"test_key": "test_value"},
            "max_wait_time": 100,
            "poll_interval": 200,
        }

        self.generic_model.model_deployment.infrastructure.with_compartment_id(
            "compartment_id_passed_using_with"
        ).with_project_id("project_id_passed_using_with")
        self.generic_model.model_deployment.runtime.with_image(
            "image_passed_using_with"
        )

        result = self.generic_model.deploy(
            **input_dict,
        )
        assert result == mock_deploy.return_value
        assert result.infrastructure.access_log == {
            "log_id": input_dict["deployment_access_log_id"],
            "log_group_id": input_dict["deployment_log_group_id"],
        }
        assert result.infrastructure.predict_log == {
            "log_id": input_dict["deployment_predict_log_id"],
            "log_group_id": input_dict["deployment_log_group_id"],
        }
        assert (
            result.infrastructure.bandwidth_mbps
            == input_dict["deployment_bandwidth_mbps"]
        )
        assert (
            result.infrastructure.compartment_id == "compartment_id_passed_using_with"
        )
        assert result.infrastructure.project_id == "project_id_passed_using_with"
        assert (
            result.infrastructure.shape_name == input_dict["deployment_instance_shape"]
        )
        assert result.infrastructure.shape_config_details == {
            "ocpus": input_dict["deployment_ocpus"],
            "memory_in_gbs": input_dict["deployment_memory_in_gbs"],
        }
        assert result.runtime.image == "image_passed_using_with"
        assert result.runtime.entrypoint == input_dict["entrypoint"]
        assert result.runtime.server_port == input_dict["server_port"]
        assert result.runtime.health_check_port == input_dict["health_check_port"]
        assert result.runtime.env == {"test_key": "test_value"}
        assert result.runtime.deployment_mode == "HTTPS_ONLY"
        mock_deploy.assert_called_with(
            wait_for_completion=input_dict["wait_for_completion"],
            max_wait_time=input_dict["max_wait_time"],
            poll_interval=input_dict["poll_interval"],
        )

        assert (
            self.generic_model.properties.compartment_id
            == "compartment_id_passed_using_with"
        )
        assert (
            self.generic_model.properties.project_id == "project_id_passed_using_with"
        )
        assert (
            self.generic_model.properties.deployment_image == "image_passed_using_with"
        )


class TestCommonMethods:
    """Tests common methods presented in the generic_model module."""

    @pytest.mark.parametrize(
        "test_value, expected_value",
        [
            ("/artifact_dir", "/artifact_dir"),
            ("", "tmp_artifact_dir"),
            (None, "tmp_artifact_dir"),
            ("artifact_dir", "artifact_dir"),
        ],
    )
    @patch("tempfile.mkdtemp", return_value="tmp_artifact_dir")
    @patch("ads.common.auth.default_signer")
    def test__prepare_artifact_dir(
        self, mock_signer, mock_mkdtemp, test_value, expected_value
    ):
        """Ensures that artifact dir name can be benerated propertly."""

        assert (
            _prepare_artifact_dir(test_value) == expected_value
            if expected_value == "tmp_artifact_dir"
            else os.path.abspath(os.path.expanduser(expected_value))
        )
