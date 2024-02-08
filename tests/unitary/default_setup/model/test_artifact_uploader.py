#!/usr/bin/env python

# Copyright (c) 2022, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import glob
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest
from ads.model.artifact_uploader import LargeArtifactUploader, SmallArtifactUploader
from ads.model.common.utils import zip_artifact
from ads.common.auth import default_signer
from ads.common.utils import DEFAULT_PARALLEL_PROCESS_COUNT
from oci import object_storage

MODEL_OCID = "ocid1.datasciencemodel.oc1.xxx"


class TestArtifactUploader:
    def setup_class(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))
        cls.mock_artifact_path = os.path.join(
            cls.curr_dir, "test_files/model_artifacts/"
        )
        cls.mock_artifact_file_path = os.path.join(
            cls.curr_dir, "test_files/model_artifacts.zip"
        )
        cls.mock_oci_artifact_path = "oci://bucket-name@namespace/model_artifacts.zip"

    def teardown_class(cls):
        # if os.path.exists(cls.mock_artifact_path):
        #     shutil.rmtree(cls.mock_artifact_path, ignore_errors=True)
        pass

    def setup_method(self):
        self.mock_dsc_model = MagicMock(
            create_model_artifact=MagicMock(),
            export_model_artifact=MagicMock(),
            id=MODEL_OCID,
        )
        self.mock_auth = {"config": {}}
        self.mock_region = "test_region"

    def test__init__(self):
        # Ensures that initialization of the uploader fails in case of incorrect artifact path
        with pytest.raises(ValueError, match=f"The `not_existing_path` does not exist"):
            SmallArtifactUploader(
                dsc_model=self.mock_dsc_model, artifact_path="not_existing_path"
            )

        # Ensures the SmallArtifactUploader can be successfully initialized
        with patch("os.path.exists", return_value=True):
            sm_artifact_uploader = SmallArtifactUploader(
                dsc_model=self.mock_dsc_model, artifact_path="existing_path"
            )
            assert sm_artifact_uploader.dsc_model == self.mock_dsc_model
            assert sm_artifact_uploader.artifact_path == "existing_path"
            assert sm_artifact_uploader.artifact_file_path == None
            assert sm_artifact_uploader.progress == None

        # Ensures the LargeArtifactUploader can be successfully initialized
        with patch("os.path.exists", return_value=True):
            with pytest.raises(ValueError, match="The `bucket_uri` must be provided."):
                lg_artifact_uploader = LargeArtifactUploader(
                    dsc_model=self.mock_dsc_model,
                    artifact_path="existing_path",
                    auth=self.mock_auth,
                    region=self.mock_region,
                    bucket_uri="",
                    overwrite_existing_artifact=False,
                    remove_existing_artifact=False,
                )

            invalid_artifact_path = "oci://my-bucket@my-tenancy/mymodel"
            with pytest.raises(ValueError):
                lg_artifact_uploader = LargeArtifactUploader(
                    dsc_model=self.mock_dsc_model,
                    artifact_path=invalid_artifact_path,
                    auth=self.mock_auth,
                    region=self.mock_region,
                    overwrite_existing_artifact=False,
                    remove_existing_artifact=False,
                )

            with patch("ads.common.utils.is_path_exists", return_value=False):
                with pytest.raises(
                    ValueError,
                    match=f"The `{self.mock_oci_artifact_path}` does not exist",
                ):
                    lg_artifact_uploader = LargeArtifactUploader(
                        dsc_model=self.mock_dsc_model,
                        artifact_path=self.mock_oci_artifact_path,
                        auth=self.mock_auth,
                        region=self.mock_region,
                        overwrite_existing_artifact=False,
                        remove_existing_artifact=False,
                    )

            auth = default_signer()
            lg_artifact_uploader = LargeArtifactUploader(
                dsc_model=self.mock_dsc_model,
                artifact_path="existing_path",
                auth=auth,
                region=self.mock_region,
                bucket_uri="test_bucket_uri",
                overwrite_existing_artifact=False,
                remove_existing_artifact=False,
            )
            assert lg_artifact_uploader.dsc_model == self.mock_dsc_model
            assert lg_artifact_uploader.artifact_path == "existing_path"
            assert lg_artifact_uploader.artifact_file_path == None
            assert lg_artifact_uploader.progress == None
            assert lg_artifact_uploader.auth == auth
            assert lg_artifact_uploader.region == self.mock_region
            assert lg_artifact_uploader.bucket_uri == "test_bucket_uri"
            assert lg_artifact_uploader.overwrite_existing_artifact == False
            assert lg_artifact_uploader.remove_existing_artifact == False
            assert (
                lg_artifact_uploader._parallel_process_count
                == DEFAULT_PARALLEL_PROCESS_COUNT
            )

            with patch("ads.common.utils.is_path_exists", return_value=True):
                uploader = LargeArtifactUploader(
                    dsc_model=self.mock_dsc_model,
                    artifact_path=self.mock_oci_artifact_path,
                    overwrite_existing_artifact=False,
                    remove_existing_artifact=False,
                )
                assert uploader.artifact_path == self.mock_oci_artifact_path
                assert uploader.bucket_uri == self.mock_oci_artifact_path
                assert uploader.artifact_file_path == None

    def test_prepare_artifact_tmp_zip(self):
        # Tests case when a folder provided as artifacts location
        with patch("ads.model.common.utils.zip_artifact") as mock_zip_artifact:
            mock_zip_artifact.return_value = "test_artifact.zip"
            artifact_uploader = SmallArtifactUploader(
                dsc_model=self.mock_dsc_model, artifact_path=self.mock_artifact_path
            )
            test_result = artifact_uploader._prepare_artifact_tmp_file()
            assert test_result == "test_artifact.zip"

        # Tests case when a zip file provided as artifacts location
        with patch("os.path.exists", return_value=True):
            with patch("os.path.isfile", return_value=True):
                artifact_uploader = SmallArtifactUploader(
                    dsc_model=self.mock_dsc_model,
                    artifact_path=self.mock_artifact_path + ".zip",
                )
                test_result = artifact_uploader._prepare_artifact_tmp_file()
                assert test_result == self.mock_artifact_path + ".zip"

        # Tests case when a zip file provided as object storage path
        with patch("ads.common.utils.is_path_exists", return_value=True):
            artifact_uploader = LargeArtifactUploader(
                dsc_model=self.mock_dsc_model,
                artifact_path=self.mock_oci_artifact_path,
            )
            test_result = artifact_uploader._prepare_artifact_tmp_file()
            assert test_result == self.mock_oci_artifact_path

    def test_remove_artifact_tmp_zip(self):
        artifact_uploader = SmallArtifactUploader(
            dsc_model=self.mock_dsc_model, artifact_path=self.mock_artifact_path
        )
        with patch("shutil.rmtree") as mock_rmtree:
            # Tests case when tmp artifact needs to be removed
            artifact_uploader.artifact_file_path = "artifacts.zip"
            artifact_uploader._prepare_artifact_tmp_file()
            mock_rmtree.assert_called_with("artifacts.zip", ignore_errors=True)

        with patch("os.path.exists", return_value=True):
            artifact_uploader = SmallArtifactUploader(
                dsc_model=self.mock_dsc_model, artifact_path="artifacts.zip"
            )
            with patch("shutil.rmtree") as mock_rmtree:
                # Tests case when tmp artifact shouldn't be removed
                artifact_uploader.artifact_file_path = "artifacts.zip"
                artifact_uploader.artifact_path = "artifacts.zip"
                artifact_uploader._prepare_artifact_tmp_file()
                mock_rmtree.assert_not_called()

    @patch.object(SmallArtifactUploader, "_upload")
    @patch.object(SmallArtifactUploader, "_prepare_artifact_tmp_zip")
    @patch.object(SmallArtifactUploader, "_remove_artifact_tmp_zip")
    def test_upload(
        self,
        mock__remove_artifact_tmp_zip,
        mock__prepare_artifact_tmp_zip,
        mock__upload,
    ):
        artifact_uploader = SmallArtifactUploader(
            dsc_model=self.mock_dsc_model, artifact_path=self.mock_artifact_path
        )
        artifact_uploader.upload()
        mock__remove_artifact_tmp_zip.assert_called()
        mock__prepare_artifact_tmp_zip.assert_called()
        mock__upload.assert_called()

    def test_upload_small_artifact(self):
        with open(self.mock_artifact_file_path, "rb") as file_data:
            with patch.object(
                SmallArtifactUploader,
                "_prepare_artifact_tmp_zip",
                return_value=self.mock_artifact_file_path,
            ) as mock_prepare_artifact_tmp_zip:
                with patch.object(
                    SmallArtifactUploader, "_remove_artifact_tmp_zip"
                ) as mock_remove_artifact_tmp_zip:
                    artifact_uploader = SmallArtifactUploader(
                        dsc_model=self.mock_dsc_model,
                        artifact_path=self.mock_artifact_path,
                    )
                    artifact_uploader.artifact_file_path = self.mock_artifact_file_path
                    artifact_uploader.upload()
                    mock_prepare_artifact_tmp_zip.assert_called()
                    mock_remove_artifact_tmp_zip.assert_called()
                    self.mock_dsc_model.create_model_artifact.assert_called()

    @patch("ads.common.utils.is_path_exists")
    @patch("ads.common.utils.upload_to_os")
    def test_upload_large_artifact(self, mock_upload, mock_is_path_exists):
        # Case when artifact already exists and overwrite_existing_artifact==True
        dest_path = "oci://my-bucket@my-namespace/my-artifact-path"
        test_bucket_file_name = os.path.join(dest_path, f"{MODEL_OCID}.zip")
        mock_is_path_exists.return_value = True
        auth = default_signer()
        artifact_uploader = LargeArtifactUploader(
            dsc_model=self.mock_dsc_model,
            artifact_path=self.mock_artifact_file_path,
            bucket_uri=dest_path + "/",
            auth=auth,
            region=self.mock_region,
            overwrite_existing_artifact=True,
            remove_existing_artifact=False,
        )
        artifact_uploader.upload()
        mock_upload.assert_called_with(
            src_uri=self.mock_artifact_file_path,
            dst_uri=test_bucket_file_name,
            auth=auth,
            parallel_process_count=DEFAULT_PARALLEL_PROCESS_COUNT,
            force_overwrite=True,
            progressbar_description="Copying model artifact to the Object Storage bucket.",
        )
        self.mock_dsc_model.export_model_artifact.assert_called_with(
            bucket_uri=test_bucket_file_name, region=self.mock_region
        )

        # Case when artifact already exists and overwrite_existing_artifact==False
        with pytest.raises(FileExistsError):
            artifact_uploader = LargeArtifactUploader(
                dsc_model=self.mock_dsc_model,
                artifact_path=self.mock_artifact_file_path,
                bucket_uri=dest_path + "/",
                auth=default_signer(),
                region=self.mock_region,
                overwrite_existing_artifact=False,
                remove_existing_artifact=False,
            )
            artifact_uploader.upload()

    @patch("ads.common.utils.is_path_exists", return_value=True)
    @patch("ads.common.utils.upload_to_os")
    def test_skip_upload(self, mock_upload, mock_is_path_exists):
        """Tests case when provided artifact is object storage path."""
        artifact_uploader = LargeArtifactUploader(
            dsc_model=self.mock_dsc_model,
            artifact_path=self.mock_oci_artifact_path,
            auth=default_signer(),
            region=self.mock_region,
            overwrite_existing_artifact=False,
            remove_existing_artifact=False,
        )
        artifact_uploader.upload()
        mock_upload.assert_not_called()
        self.mock_dsc_model.export_model_artifact.assert_called_with(
            bucket_uri=self.mock_oci_artifact_path, region=self.mock_region
        )

    def test_zip_artifact_fail(self):
        with pytest.raises(ValueError, match="The `artifact_dir` must be provided."):
            zip_artifact(None)
        with pytest.raises(ValueError, match=f"The not_existing_artifact not exists."):
            zip_artifact("not_existing_artifact")

    def test_zip_artifact_success(self):
        test_result = zip_artifact(self.mock_artifact_path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with ZipFile(test_result) as zip_file:
                zip_file.extractall(tmp_dir)

            test_files = list(glob.iglob(os.path.join(tmp_dir, "**"), recursive=True))
            expected_files = [
                os.path.join(tmp_dir, file_name)
                for file_name in ["", "runtime.yaml", "score.py"]
            ]
            assert sorted(test_files) == sorted(expected_files)

        shutil.rmtree(test_result, ignore_errors=True)
