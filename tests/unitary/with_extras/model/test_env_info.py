#!/usr/bin/env python

# Copyright (c) 2022, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
from unittest.mock import ANY, mock_open, patch

import pytest
import requests
import yaml
from ads.model.runtime.env_info import InferenceEnvInfo, TrainingEnvInfo
from ads.model.runtime import utils
from ads.model.runtime.utils import get_service_packs
from cerberus import DocumentError

try:
    from yaml import CLoader as loader
except:
    from yaml import Loader as loader


def mocked_requests_request(*args, **kwargs):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.status_code = status_code
            self.json_data = json_data

        def json(self):
            return self.json_data

        @property
        def ok(self):
            return True

    if (
        args[0] == "GET"
        and args[1]
        == "https://objectstorage.us-ashburn-1.oraclecloud.com/p/Ri7zFc_h91sxMdgnza9Qnqw3Ina8hf8wzDvEpAnUXMDOnUR1U1fpsaBUjUfgPgIq/n/ociodscdev/b/service-conda-packs/o/service_pack/index.json"
    ):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(curr_dir, "index.json"), encoding="utf-8") as f:
            data = json.load(f)
        return MockResponse(data, 200)
    return MockResponse(None, 404)


def mocked_requests_request_bad_response(*args, **kwargs):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.status_code = status_code
            self.json_data = json_data

        def json(self):
            return self.json_data

        @property
        def ok(self):
            return False

    return MockResponse(None, 404)


def mocked_requests_request_no_internet(*args, **kwargs):
    raise Exception


@patch("requests.request", mocked_requests_request)
def test_get_service_packs():
    service_pack_path_mapping, service_pack_slug_mapping = get_service_packs(
        "ociodscdev", "service-conda-packs", None
    )
    assert (
        isinstance(service_pack_path_mapping, dict)
        and service_pack_path_mapping is not None
    )
    assert (
        isinstance(service_pack_slug_mapping, dict)
        and service_pack_slug_mapping is not None
    )
    assert all(
        path.startswith("oci://") for path, _ in service_pack_path_mapping.items()
    )
    slugs = [slug for slug, _ in service_pack_path_mapping.values()]
    assert all(slug in slugs for slug, _ in service_pack_slug_mapping.items())


@patch("requests.request", mocked_requests_request_no_internet)
def test__get_index_json_through_bucket():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(curr_dir, "index.json"), encoding="utf-8") as f:
        data = json.load(f)
    open_mock = mock_open(read_data=json.dumps(data))
    with patch("fsspec.open", open_mock):
        service_pack_list = utils._get_index_json_through_bucket(
            "ociodscdev", "service-conda-packs"
        )

    assert isinstance(service_pack_list, list) and service_pack_list is not None


@patch("requests.request", mocked_requests_request_bad_response)
def test_get_service_packs_bad_response():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(curr_dir, "index.json"), encoding="utf-8") as f:
        data = json.load(f)
    open_mock = mock_open(read_data=json.dumps(data))
    with patch("fsspec.open", open_mock):
        service_pack_path_mapping, service_pack_slug_mapping = get_service_packs(
            "ociodscdev", "service-conda-packs", None
        )
    open_mock.assert_called_with(
        "oci://service-conda-packs@ociodscdev/service_pack/index.json", "r"
    )
    assert (
        isinstance(service_pack_path_mapping, dict)
        and service_pack_path_mapping is not None
    )
    assert (
        isinstance(service_pack_slug_mapping, dict)
        and service_pack_slug_mapping is not None
    )
    assert all(
        path.startswith("oci://") for path, _ in service_pack_path_mapping.items()
    )
    slugs = [slug for slug, _ in service_pack_path_mapping.values()]
    assert all(slug in slugs for slug, _ in service_pack_slug_mapping.items())


class TestTrainingEnvInfo:
    """Test the TrainingEnvInfo class."""

    @classmethod
    def setup_class(cls):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(curr_dir, "runtime.yaml"), encoding="utf-8") as rt:
            cls.runtime_dict = yaml.load(rt, loader)

        with open(os.path.join(curr_dir, "runtime_fail.yaml"), encoding="utf-8") as rt:
            cls.runtime_dict_fail = yaml.load(rt, loader)

    def test_init(self):
        info = TrainingEnvInfo()
        assert info.training_env_slug == ""
        assert info.training_env_type == ""
        assert info.training_env_path == ""
        assert info.training_python_version == ""

    @patch("ads.model.runtime.env_info.get_service_packs")
    def test_from_slug_dev_sp(self, mock_get_service_packs):
        env_path = "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        mock_get_service_packs.return_value = ({}, {"mlcpuv1": (env_path, "3.6")})
        info = TrainingEnvInfo.from_slug(
            "mlcpuv1", namespace="ociodscdev", bucketname="service-conda-packs"
        )
        assert info.training_env_slug == "mlcpuv1"
        assert info.training_env_path == env_path
        assert info.training_env_type == "data_science"
        assert info.training_python_version == "3.6"

    @patch("ads.model.runtime.env_info.get_service_packs")
    def test_from_slug_prod_sp(self, mock_get_service_packs):
        env_path = "oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        mock_get_service_packs.return_value = ({}, {"mlcpuv1": (env_path, "3.6")})
        info = TrainingEnvInfo.from_slug(
            "mlcpuv1", namespace="id19sfcrra6z", bucketname="service-conda-packs"
        )
        assert info.training_env_slug == "mlcpuv1"
        assert info.training_env_path == env_path
        assert info.training_env_type == "data_science"
        assert info.training_python_version == "3.6"

    @patch("ads.model.runtime.env_info.get_service_packs")
    def test_from_slug_not_exist(self, mock_get_service_packs):
        mock_get_service_packs.return_value = (
            {
                "test_path": ("mlcpuv1", "3.6"),
            },
            {
                "mlcpuv1": ("test_path", "3.6"),
            },
        )
        with pytest.raises(ValueError) as exc_info:
            TrainingEnvInfo.from_slug(
                "not_exist", namespace="ociodscdev", bucketname="service-conda-packs"
            )
        error_message = str(exc_info.value)
        assert (
            "conda environment slug `not_exist` could not be resolved" in error_message
        )
        assert "full OCI path from Environment Explorer" in error_message

    @patch("ads.model.runtime.env_info.get_service_packs")
    def test_from_slug_service_pack_list_not_extracted(self, mock_get_service_packs):
        mock_get_service_packs.return_value = ({}, {})
        with pytest.raises(ValueError) as exc_info:
            TrainingEnvInfo.from_slug(
                "mlcpuv1", namespace="ociodscdev", bucketname="service-conda-packs"
            )
        error_message = str(exc_info.value)
        assert (
            "service conda environment list could not be extracted" in error_message
        )
        assert (
            "full conda environment path from Environment Explorer" in error_message
        )

    @patch("ads.model.runtime.env_info.get_service_packs")
    @patch("ads.model.runtime.env_info.utils.is_path_exists", return_value=True)
    @patch(
        "ads.model.runtime.env_info.ObjectStorageDetails.fetch_metadata_of_object",
        return_value={"slug": "py37_250", "python": "3.7"},
    )
    def test_from_path_cp(
        self, mock_fetch_metadata, mock_is_path_exists, mock_get_service_packs
    ):
        env_path = "oci://bucket-Condas@llerda/conda_environments/cpu/Beat Python 3.12/1.0/beatpython3_12v1_0"
        info = TrainingEnvInfo.from_path(
            env_path,
            auth={"signer": object()},
        )
        assert info.training_env_path == env_path
        assert info.training_env_type == "published"
        assert info.training_env_slug == "py37_250"
        assert info.training_python_version == "3.7"
        mock_get_service_packs.assert_not_called()
        mock_is_path_exists.assert_called_once_with(
            env_path,
            auth={"signer": ANY},
        )
        mock_fetch_metadata.assert_called_once()

    @patch("ads.model.runtime.env_info.get_service_packs")
    @patch("ads.model.runtime.env_info.utils.is_path_exists", return_value=True)
    @patch(
        "ads.model.runtime.env_info.ObjectStorageDetails.fetch_metadata_of_object",
        side_effect=Exception("missing metadata"),
    )
    def test_from_path(
        self, mock_fetch_metadata, _mock_is_path_exists, mock_get_service_packs
    ):
        info = TrainingEnvInfo.from_path(
            "oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1",
            auth={"signer": object()},
        )
        assert info.training_env_path == "oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        assert info.training_env_slug == "mlcpuv1"
        assert info.training_env_type == "data_science"
        assert info.training_python_version == ""
        mock_get_service_packs.assert_not_called()
        mock_fetch_metadata.assert_called_once()

    @patch("ads.model.runtime.env_info.get_service_packs")
    @patch("ads.model.runtime.env_info.utils.is_path_exists", return_value=False)
    def test_from_path_not_accessible(self, _mock_is_path_exists, mock_get_service_packs):
        env_path = "oci://license_checker@ociodscdev/conda/missing"
        with pytest.raises(ValueError) as exc_info:
            TrainingEnvInfo.from_path(env_path, auth={"signer": object()})
        error_message = str(exc_info.value)
        assert f"conda environment path `{env_path}`" in error_message
        assert "does not exist or is not accessible" in error_message
        assert (
            "valid full conda environment path from Environment Explorer"
            in error_message
        )
        assert "service conda environment list" not in error_message
        assert "conda environment slug" not in error_message
        mock_get_service_packs.assert_not_called()

    def test_from_dict(self):
        info = TrainingEnvInfo.from_dict(
            self.runtime_dict["MODEL_PROVENANCE"]["TRAINING_CONDA_ENV"]
        )
        info.training_env_slug = "mlcpuv1"
        info.training_env_type = "service_pack"
        info.training_env_path = "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        info.training_python_version = "3.7"

    def test__validate_dict(self):
        assert TrainingEnvInfo._validate_dict(
            self.runtime_dict["MODEL_PROVENANCE"]["TRAINING_CONDA_ENV"]
        )

    def test__validate_dict_fail(self):
        with pytest.raises(DocumentError):
            TrainingEnvInfo._validate_dict(
                self.runtime_dict_fail["MODEL_PROVENANCE"]["TRAINING_CONDA_ENV"]
            )


class TestInferenceEnvInfo:
    """Test the InferenceEnvInfo class."""

    @classmethod
    def setup_class(cls):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(curr_dir, "runtime.yaml"), encoding="utf-8") as rt:
            cls.runtime_dict = yaml.load(rt, loader)

        with open(os.path.join(curr_dir, "runtime_fail.yaml"), encoding="utf-8") as rt:
            cls.runtime_dict_fail = yaml.load(rt, loader)

    def test_init(self):
        info = InferenceEnvInfo()
        assert info.inference_env_slug == ""
        assert info.inference_env_type == ""
        assert info.inference_env_path == ""
        assert info.inference_python_version == ""

    @patch("ads.model.runtime.env_info.get_service_packs")
    def test_from_slug_dev_sp(self, mock_get_service_packs):
        env_path = "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        mock_get_service_packs.return_value = ({}, {"mlcpuv1": (env_path, "3.6")})
        info = InferenceEnvInfo.from_slug(
            "mlcpuv1", namespace="ociodscdev", bucketname="service-conda-packs"
        )
        assert info.inference_env_slug == "mlcpuv1"
        assert info.inference_env_path == env_path
        assert info.inference_env_type == "data_science"
        assert info.inference_python_version == "3.6"

    @patch("ads.model.runtime.env_info.get_service_packs")
    def test_from_slug_prod_sp(self, mock_get_service_packs):
        env_path = "oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        mock_get_service_packs.return_value = ({}, {"mlcpuv1": (env_path, "3.6")})
        info = InferenceEnvInfo.from_slug(
            "mlcpuv1", namespace="id19sfcrra6z", bucketname="service-conda-packs"
        )
        assert info.inference_env_slug == "mlcpuv1"
        assert info.inference_env_path == env_path
        assert info.inference_env_type == "data_science"
        assert info.inference_python_version == "3.6"

    @patch("ads.model.runtime.env_info.get_service_packs")
    def test_from_slug_not_exist(self, mock_get_service_packs):
        mock_get_service_packs.return_value = (
            {
                "test_path": ("mlcpuv1", "3.6"),
            },
            {
                "mlcpuv1": ("test_path", "3.6"),
            },
        )
        with pytest.raises(ValueError) as exc_info:
            InferenceEnvInfo.from_slug(
                "not_exist", namespace="ociodscdev", bucketname="service-conda-packs"
            )
        error_message = str(exc_info.value)
        assert (
            "conda environment slug `not_exist` could not be resolved" in error_message
        )
        assert "full OCI path from Environment Explorer" in error_message

    @patch("ads.model.runtime.env_info.get_service_packs")
    def test_from_slug_service_pack_list_not_extracted(self, mock_get_service_packs):
        mock_get_service_packs.return_value = ({}, {})
        with pytest.raises(ValueError) as exc_info:
            InferenceEnvInfo.from_slug(
                "mlcpuv1", namespace="ociodscdev", bucketname="service-conda-packs"
            )
        error_message = str(exc_info.value)
        assert (
            "service conda environment list could not be extracted" in error_message
        )
        assert (
            "full conda environment path from Environment Explorer" in error_message
        )

    @patch("ads.model.runtime.env_info.get_service_packs")
    @patch("ads.model.runtime.env_info.utils.is_path_exists", return_value=True)
    @patch(
        "ads.model.runtime.env_info.ObjectStorageDetails.fetch_metadata_of_object",
        return_value={"slug": "py37_250", "python": "3.7"},
    )
    def test_from_path_custom_pack(
        self, mock_fetch_metadata, _mock_is_path_exists, mock_get_service_packs
    ):
        env_path = "oci://bucket-Condas@llerda/conda_environments/cpu/Beat Python 3.12/1.0/beatpython3_12v1_0"
        info = InferenceEnvInfo.from_path(
            env_path,
            auth={"signer": object()},
        )
        assert info.inference_env_path == env_path
        assert info.inference_env_type == "published"
        assert info.inference_env_slug == "py37_250"
        assert info.inference_python_version == "3.7"
        mock_get_service_packs.assert_not_called()
        mock_fetch_metadata.assert_called_once()

    @patch("ads.model.runtime.env_info.get_service_packs")
    @patch("ads.model.runtime.env_info.utils.is_path_exists", return_value=True)
    @patch(
        "ads.model.runtime.env_info.ObjectStorageDetails.fetch_metadata_of_object",
        side_effect=Exception("missing metadata"),
    )
    def test_from_path(
        self, mock_fetch_metadata, _mock_is_path_exists, mock_get_service_packs
    ):
        info = InferenceEnvInfo.from_path(
            "oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1",
            auth={"signer": object()},
        )
        assert info.inference_env_path == "oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        assert info.inference_env_slug == "mlcpuv1"
        assert info.inference_env_type == "data_science"
        assert info.inference_python_version == ""
        mock_get_service_packs.assert_not_called()
        mock_fetch_metadata.assert_called_once()

    @patch("ads.model.runtime.env_info.get_service_packs")
    @patch("ads.model.runtime.env_info.utils.is_path_exists", return_value=False)
    def test_from_path_not_accessible(self, _mock_is_path_exists, mock_get_service_packs):
        env_path = "oci://license_checker@ociodscdev/conda/missing"
        with pytest.raises(ValueError) as exc_info:
            InferenceEnvInfo.from_path(env_path, auth={"signer": object()})
        error_message = str(exc_info.value)
        assert f"conda environment path `{env_path}`" in error_message
        assert "does not exist or is not accessible" in error_message
        assert (
            "valid full conda environment path from Environment Explorer"
            in error_message
        )
        assert "service conda environment list" not in error_message
        assert "conda environment slug" not in error_message
        mock_get_service_packs.assert_not_called()

    def test_from_dict(self):
        info = InferenceEnvInfo.from_dict(
            self.runtime_dict["MODEL_DEPLOYMENT"]["INFERENCE_CONDA_ENV"]
        )
        info.inference_env_slug = "mlcpuv1"
        info.inference_env_type = "service_pack"
        info.inference_env_path = "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        info.inference_python_version = "3.7"

    def test__validate_dict(self):
        assert InferenceEnvInfo.from_dict(
            self.runtime_dict["MODEL_DEPLOYMENT"]["INFERENCE_CONDA_ENV"]
        )

    def test__validate_dict_fail(self):
        with pytest.raises(DocumentError):
            InferenceEnvInfo.from_dict(
                self.runtime_dict_fail["MODEL_DEPLOYMENT"]["INFERENCE_CONDA_ENV"]
            )
