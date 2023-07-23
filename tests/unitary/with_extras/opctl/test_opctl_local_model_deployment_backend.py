#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from mock import ANY, patch, MagicMock
import pytest
from ads.opctl.backend.local import (
    LocalModelDeploymentBackend,
    ModelCustomMetadata,
    os,
    create_signer,
    OCIClientFactory,
    is_in_notebook_session,
    run_command,
)


class TestLocalModelDeploymentBackend:
    @property
    def config(self):
        return {
            "execution": {
                "backend": "local",
                "use_conda": True,
                "debug": False,
                "env_var": ["TEST_ENV=test_env"],
                "oci_config": "~/.oci/config",
                "oci_profile": "DEFAULT",
                "image": "ml-job",
                "env_vars": {"TEST_ENV": "test_env"},
                "job_name": "hello-world",
                "auth": "api_key",
                "ocid": "fake_id",
                "compartment_id": "fake_id",
                "project_id": "fake_id",
                "payload": "fake_data",
                "bucket_uri": "fake_bucket",
            },
            "infrastructure": {},
        }

    @property
    def backend(self):
        return LocalModelDeploymentBackend(config=self.config)

    @property
    def custom_metadata(self):
        custom_metadata = ModelCustomMetadata()
        custom_metadata.add(key="CondaEnvironmentPath", value="fake_path")
        custom_metadata.add(key="SlugName", value="fake_slug")
        return custom_metadata

    @patch.object(ModelCustomMetadata, "_from_oci_metadata")
    def test__get_conda_info_from_custom_metadata(
        self,
        mock_custom_metadata,
    ):
        mock_custom_metadata.return_value = self.custom_metadata

        backend = LocalModelDeploymentBackend(config=self.config)
        backend.client.get_model = MagicMock()
        conda_slug, conda_path = backend._get_conda_info_from_custom_metadata("fake_id")
        assert conda_slug == "fake_slug"
        assert conda_path == "fake_path"

    def test__get_conda_info_from_runtime(self):
        yaml_str = """
MODEL_ARTIFACT_VERSION: '3.0'
MODEL_DEPLOYMENT:
  INFERENCE_CONDA_ENV:
    INFERENCE_ENV_PATH: fake_path
    INFERENCE_ENV_SLUG: fake_slug
    INFERENCE_ENV_TYPE: data_science
    INFERENCE_PYTHON_VERSION: '3.8'
MODEL_PROVENANCE:
  PROJECT_OCID: ''
  TENANCY_OCID: ''
  TRAINING_CODE:
    ARTIFACT_DIRECTORY: fake_dir
  TRAINING_COMPARTMENT_OCID: ''
  TRAINING_CONDA_ENV:
    TRAINING_ENV_PATH: ''
    TRAINING_ENV_SLUG: ''
    TRAINING_ENV_TYPE: ''
    TRAINING_PYTHON_VERSION: ''
  TRAINING_REGION: ''
  TRAINING_RESOURCE_OCID: ''
  USER_OCID: ''
  VM_IMAGE_INTERNAL_ID: ''
"""
        with open("./runtime.yaml", "w") as f:
            f.write(yaml_str)

        (
            conda_slug,
            conda_path,
        ) = LocalModelDeploymentBackend._get_conda_info_from_runtime("./")

        assert conda_slug == "fake_slug"
        assert conda_path == "fake_path"

    @patch("ads.opctl.backend.local.os.listdir", return_value=["path"])
    @patch("ads.opctl.backend.local.os.path.exists", return_value=True)
    def test_predict(self, mock_path_exists, mock_list_dir):
        with patch("ads.opctl.backend.local._download_model") as mock__download:
            with patch.object(
                LocalModelDeploymentBackend,
                "_get_conda_info_from_custom_metadata",
                return_value=("fake_slug", "fake_path"),
            ):
                with patch.object(
                    LocalModelDeploymentBackend, "_get_conda_info_from_runtime"
                ):
                    with patch.object(
                        LocalModelDeploymentBackend,
                        "_run_with_conda_pack",
                        return_value=0,
                    ) as mock__run_with_conda_pack:
                        backend = LocalModelDeploymentBackend(self.config)
                        backend.predict()
                        mock__download.assert_not_called()
                        mock__run_with_conda_pack.assert_called_once_with(
                            {
                                os.path.expanduser("~/.oci"): {
                                    "bind": "/home/datascience/.oci"
                                },
                                os.path.expanduser("~/.ads_ops/models/fake_id"): {
                                    "bind": "/opt/ds/model/deployed_model/"
                                },
                            },
                            "--artifact-directory /opt/ds/model/deployed_model/ --payload 'fake_data' --auth api_key --profile DEFAULT",
                            install=True,
                            conda_uri="fake_path",
                        )

    @patch("ads.opctl.backend.local.os.listdir", return_value=["path"])
    @patch("ads.opctl.backend.local.os.path.exists", return_value=False)
    def test_predict_download(self, mock_path_exists, mock_list_dir):
        with patch("ads.opctl.backend.local._download_model") as mock__download:
            with patch.object(
                LocalModelDeploymentBackend,
                "_get_conda_info_from_custom_metadata",
                return_value=("fake_slug", "fake_path"),
            ):
                with patch.object(
                    LocalModelDeploymentBackend, "_get_conda_info_from_runtime"
                ):
                    with patch.object(
                        LocalModelDeploymentBackend,
                        "_run_with_conda_pack",
                        return_value=0,
                    ) as mock__run_with_conda_pack:
                        backend = LocalModelDeploymentBackend(self.config)

                        backend.predict()
                        mock__download.assert_called_once_with(
                            auth="api_key",
                            profile="DEFAULT",
                            ocid="fake_id",
                            artifact_directory=os.path.expanduser(
                                "~/.ads_ops/models/fake_id"
                            ),
                            region=None,
                            bucket_uri="fake_bucket",
                            timeout=None,
                            force_overwrite=True,
                        )
                        mock__run_with_conda_pack.assert_called_once_with(
                            {
                                os.path.expanduser("~/.oci"): {
                                    "bind": "/home/datascience/.oci"
                                },
                                os.path.expanduser("~/.ads_ops/models/fake_id"): {
                                    "bind": "/opt/ds/model/deployed_model/"
                                },
                            },
                            "--artifact-directory /opt/ds/model/deployed_model/ --payload 'fake_data' --auth api_key --profile DEFAULT",
                            install=True,
                            conda_uri="fake_path",
                        )

    @patch("ads.opctl.backend.local.is_in_notebook_session", return_value=True)
    @patch("ads.opctl.backend.local.os.listdir", return_value=["path"])
    @patch("ads.opctl.backend.local.os.path.exists", return_value=False)
    def test_predict_download_in_notebook(
        self, mock_path_exists, mock_list_dir, mock_is_in_notebook
    ):
        with patch("ads.opctl.backend.local._download_model") as mock__download:
            with patch.object(
                LocalModelDeploymentBackend,
                "_get_conda_info_from_custom_metadata",
                return_value=("fake_slug", "fake_path"),
            ):
                with patch.object(
                    LocalModelDeploymentBackend, "_get_conda_info_from_runtime"
                ):
                    with patch(
                        "ads.opctl.backend.local.run_command"
                    ) as mock_run_command:
                        backend = LocalModelDeploymentBackend(self.config)

                        backend.predict()
                        mock__download.assert_called_once_with(
                            auth="api_key",
                            profile="DEFAULT",
                            ocid="fake_id",
                            artifact_directory=os.path.expanduser(
                                "~/.ads_ops/models/fake_id"
                            ),
                            region=None,
                            bucket_uri="fake_bucket",
                            timeout=None,
                            force_overwrite=True,
                        )
                        mock_run_command.assert_called_once()
