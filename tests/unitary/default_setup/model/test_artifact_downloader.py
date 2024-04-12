#!/usr/bin/env python

# Copyright (c) 2022, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import glob
import os
import shutil
import tempfile
import json
from unittest.mock import MagicMock, patch

import pytest
from ads.model.artifact_downloader import (
    LargeArtifactDownloader,
    SmallArtifactDownloader,
)

MODEL_OCID = "ocid1.datasciencemodel.oc1.xxx"


class TestArtifactDownloader:
    def setup_class(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))
        cls.mock_artifact_path = os.path.join(
            cls.curr_dir, "test_files/model_artifacts/"
        )
        cls.mock_artifact_file_path = os.path.join(
            cls.curr_dir, "test_files/model_artifacts.zip"
        )

    def teardown_class(cls):
        # if os.path.exists(cls.mock_artifact_path):
        #     shutil.rmtree(cls.mock_artifact_path, ignore_errors=True)
        pass

    def setup_method(self):
        self.mock_dsc_model = MagicMock(
            get_artifact_info=MagicMock(),
            get_model_artifact_content=MagicMock(),
            import_model_artifact=MagicMock(),
            id=MODEL_OCID,
        )
        self.mock_auth = {"config": {}}
        self.mock_region = "test_region"

    def test__init__(self):
        # SmallArtifactDownloader initialization
        sm_downloader = SmallArtifactDownloader(
            dsc_model=self.mock_dsc_model,
            target_dir="test_target_dir",
            force_overwrite=True,
        )
        assert sm_downloader.dsc_model == self.mock_dsc_model
        assert sm_downloader.target_dir == "test_target_dir"
        assert sm_downloader.force_overwrite == True
        assert sm_downloader.progress == None

        # LargeArtifactDownloader initialization
        lg_downloader = LargeArtifactDownloader(
            dsc_model=self.mock_dsc_model,
            target_dir="test_target_dir",
            force_overwrite=True,
            region=self.mock_region,
            bucket_uri="test_bucket_uri",
            overwrite_existing_artifact=False,
            remove_existing_artifact=False,
            auth=self.mock_auth,
        )
        assert lg_downloader.dsc_model == self.mock_dsc_model
        assert lg_downloader.target_dir == "test_target_dir"
        assert lg_downloader.force_overwrite == True
        assert lg_downloader.progress == None
        assert lg_downloader.auth == self.mock_auth
        assert lg_downloader.region == self.mock_region
        assert lg_downloader.overwrite_existing_artifact == False
        assert lg_downloader.remove_existing_artifact == False

    def test_downaload_fail(self):
        with patch("os.path.exists", return_value=True):
            with patch("os.listdir", return_value=["test.txt"]):
                with pytest.raises(ValueError):
                    SmallArtifactDownloader(
                        dsc_model=self.mock_dsc_model,
                        target_dir="test_target_dir",
                        force_overwrite=False,
                    ).download()

    def test_downaload(self):
        with patch.object(SmallArtifactDownloader, "_download") as mock_download:
            SmallArtifactDownloader(
                dsc_model=self.mock_dsc_model,
                target_dir="test_target_dir",
                force_overwrite=False,
            ).download()
            mock_download.assert_called()

    def test_downaload_small_artifact(self):
        with open(self.mock_artifact_file_path, "rb") as file_data:
            expected_artifact_bytes_content = file_data.read()

        self.mock_dsc_model.get_artifact_info.return_value = {
            "Content-Disposition": "attachment; filename=artifact.zip",
        }
        self.mock_dsc_model.get_model_artifact_content.return_value = (
            expected_artifact_bytes_content
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            SmallArtifactDownloader(
                dsc_model=self.mock_dsc_model,
                target_dir=tmp_dir,
                force_overwrite=True,
            ).download()

            self.mock_dsc_model.get_model_artifact_content.assert_called()

            test_files = list(glob.iglob(os.path.join(tmp_dir, "**"), recursive=True))
            expected_files = [
                os.path.join(tmp_dir, file_name)
                for file_name in ["", "runtime.yaml", "score.py"]
            ]
            assert sorted(test_files) == sorted(expected_files)

    def test_downaload_large_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = os.path.join(tmp_dir, "model_artifacts/")
            bucket_uri = os.path.join(tmp_dir, "model_artifacts.zip")
            shutil.copyfile(self.mock_artifact_file_path, bucket_uri)

            LargeArtifactDownloader(
                dsc_model=self.mock_dsc_model,
                target_dir=target_dir,
                force_overwrite=True,
                region=self.mock_region,
                bucket_uri=bucket_uri,
                overwrite_existing_artifact=True,
                remove_existing_artifact=True,
                auth=self.mock_auth,
            ).download()

            self.mock_dsc_model.import_model_artifact.assert_called_with(
                bucket_uri=bucket_uri, region=self.mock_region
            )

            test_files = list(
                glob.iglob(os.path.join(target_dir, "**"), recursive=True)
            )
            expected_files = [
                os.path.join(target_dir, file_name)
                for file_name in ["", "runtime.yaml", "score.py"]
            ]
            assert sorted(test_files) == sorted(expected_files)

    def test_download_small_artifact_json(self):
        """Test to check if Small Artifact downloader is called,
        artifact is saved as json to target directory
        """
        self.mock_artifact_file_path = os.path.join(
            self.curr_dir, "test_files/model_description.json"
        )
        with open(self.mock_artifact_file_path, "rb") as file_data:
            expected_artifact_bytes_content = file_data.read()

        self.mock_dsc_model.get_artifact_info.return_value = {
            "Content-Disposition": "attachment; filename=artifact.json",
        }
        self.mock_dsc_model.get_model_artifact_content.return_value = (
            expected_artifact_bytes_content
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            SmallArtifactDownloader(
                dsc_model=self.mock_dsc_model,
                target_dir=tmp_dir,
                force_overwrite=True,
            ).download()
            self.mock_dsc_model.get_model_artifact_content.assert_called()

            test_files = list(glob.iglob(os.path.join(tmp_dir, "**"), recursive=True))
            expected_files = [
                os.path.join(tmp_dir, file_name)
                for file_name in ["", "model_description.json"]
            ]
            assert sorted(test_files) == sorted(expected_files)

    @patch(
        "ads.common.object_storage_details.ObjectStorageDetails.bulk_download_from_object_storage"
    )
    def test_download_large_artifact_from_model_file_description(self, mock_download):
        """Tests whether model_file_description is loaded within downloader and is parsed, and also if
        #  download_from_model_file_description is appropriately called."""

        self.mock_dsc_model.is_model_by_reference.return_value = True
        self.mock_artifact_file_path = os.path.join(
            self.curr_dir, "test_files/model_description.json"
        )
        with open(self.mock_artifact_file_path, "r") as file_data:
            model_file_description = json.load(file_data)

        bucket_uri = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = os.path.join(tmp_dir, "model_artifacts/")

            LargeArtifactDownloader(
                dsc_model=self.mock_dsc_model,
                target_dir=target_dir,
                force_overwrite=True,
                region=self.mock_region,
                bucket_uri=bucket_uri,
                overwrite_existing_artifact=True,
                remove_existing_artifact=True,
                auth=self.mock_auth,
                model_file_description=model_file_description,
            ).download()

            mock_download.assert_called()
