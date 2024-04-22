#!/usr/bin/env python
# -*- coding: utf-8; -*-
import logging

# Copyright (c) 2022, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict, Optional

from ads.common import utils
from ads.common.object_storage_details import ObjectStorageDetails
from ads.model.common import utils as model_utils
from ads.model.service.oci_datascience_model import OCIDataScienceModel


class ArtifactUploader(ABC):
    """The abstract class to upload model artifacts."""

    PROGRESS_STEPS_COUNT = 3

    def __init__(self, dsc_model: OCIDataScienceModel, artifact_path: str):
        """Initializes `ArtifactUploader` instance.

        Parameters
        ----------
        dsc_model: OCIDataScienceModel
            The data scince model instance.
        artifact_path: str
            The model artifact location.
        """
        if not (
            ObjectStorageDetails.is_oci_path(artifact_path)
            or os.path.exists(artifact_path)
        ):
            raise ValueError(f"The `{artifact_path}` does not exist")

        self.dsc_model = dsc_model
        self.artifact_path = artifact_path
        self.artifact_file_path = None
        self.progress = None

    def upload(self):
        """Uploads model artifacts."""
        try:
            with utils.get_progress_bar(
                ArtifactUploader.PROGRESS_STEPS_COUNT + self.PROGRESS_STEPS_COUNT
            ) as progress:
                self.progress = progress
                self.progress.update("Preparing model artifacts file.")
                self._prepare_artifact_tmp_file()
                self.progress.update("Uploading model artifacts.")
                self._upload()
                self.progress.update(
                    "Uploading model artifacts has been successfully completed."
                )
                self.progress.update("Done.")
        except Exception:
            raise
        finally:
            self._remove_artifact_tmp_file()

    def _prepare_artifact_tmp_file(self) -> str:
        """Prepares model artifacts file.

        Returns
        -------
        str
            Path to the model artifact file.
        """
        if ObjectStorageDetails.is_oci_path(self.artifact_path):
            self.artifact_file_path = self.artifact_path
        elif os.path.isfile(self.artifact_path) and self.artifact_path.lower().endswith(
            (".zip", ".json")
        ):
            self.artifact_file_path = self.artifact_path
        else:
            self.artifact_file_path = model_utils.zip_artifact(
                artifact_dir=self.artifact_path
            )
        return self.artifact_file_path

    def _remove_artifact_tmp_file(self):
        """Removes temporary created artifact file."""
        if (
            self.artifact_file_path
            and self.artifact_file_path.lower() != self.artifact_path.lower()
        ):
            shutil.rmtree(self.artifact_file_path, ignore_errors=True)

    @abstractmethod
    def _upload(self):
        """Uploads model artifacts. Needs to be implemented in a child class."""


class SmallArtifactUploader(ArtifactUploader):
    """The class helper to upload small model artifacts."""

    PROGRESS_STEPS_COUNT = 1

    def _upload(self):
        """Uploads model artifacts to the model catalog."""
        _, ext = os.path.splitext(self.artifact_file_path)
        self.progress.update("Uploading model artifacts to the catalog")
        with open(self.artifact_file_path, "rb") as file_data:
            self.dsc_model.create_model_artifact(bytes_content=file_data, extension=ext)


class LargeArtifactUploader(ArtifactUploader):
    """
    The class helper to upload large model artifacts.

    Attributes
    ----------
    artifact_path: str
        The model artifact location. Possible values are:
            - object storage path to zip archive. Example: `oci://<bucket_name>@<namespace>/prefix/mymodel.zip`.
            - local path to zip archive. Example: `./mymodel.zip`.
            - local path to folder with artifacts. Example: `./mymodel`.
    artifact_file_path: str
        The uri of the zip of model artifact.
    auth: dict
        The default authetication is set using `ads.set_auth` API.
        If you need to override the default, use the `ads.common.auth.api_keys` or
        `ads.common.auth.resource_principal` to create appropriate authentication signer
        and kwargs required to instantiate IdentityClient object.
    bucket_uri: str
        The OCI Object Storage URI where model artifacts will be copied to.
        The `bucket_uri` is only necessary for uploading large artifacts which
        size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.

        .. versionadded:: 2.8.10

            If artifact_path is object storage path to a zip archive, bucket_uri will be ignored.

    dsc_model: OCIDataScienceModel
        The data scince model instance.
    overwrite_existing_artifact: bool
        Overwrite target bucket artifact if exists.
    progress: TqdmProgressBar
        An instance of the TqdmProgressBar.
    region: str
        The destination Object Storage bucket region.
        By default the value will be extracted from the `OCI_REGION_METADATA` environment variables.
    remove_existing_artifact: bool
        Wether artifacts uploaded to object storage bucket need to be removed or not.
    upload_manager: UploadManager
        The uploadManager simplifies interaction with the Object Storage service.
    """

    PROGRESS_STEPS_COUNT = 4

    def __init__(
        self,
        dsc_model: OCIDataScienceModel,
        artifact_path: str,
        bucket_uri: str = None,
        auth: Optional[Dict] = None,
        region: Optional[str] = None,
        overwrite_existing_artifact: Optional[bool] = True,
        remove_existing_artifact: Optional[bool] = True,
        parallel_process_count: int = utils.DEFAULT_PARALLEL_PROCESS_COUNT,
    ):
        """Initializes `LargeArtifactUploader` instance.

        Parameters
        ----------
        dsc_model: OCIDataScienceModel
            The data scince model instance.
        artifact_path: str
            The model artifact location. Possible values are:
                - object storage path to zip archive. Example: `oci://<bucket_name>@<namespace>/prefix/mymodel.zip`.
                - local path to zip archive. Example: `./mymodel.zip`.
                - local path to folder with artifacts. Example: `./mymodel`.
        bucket_uri: (str, optional). Defaults to `None`.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for uploading large artifacts from local which
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.

            .. versionadded:: 2.8.10

                If `artifact_path` is object storage path to a zip archive, `bucket_uri` will be ignored.

        auth: (Dict, optional). Defaults to `None`.
            The default authetication is set using `ads.set_auth` API.
            If you need to override the default, use the `ads.common.auth.api_keys` or
            `ads.common.auth.resource_principal` to create appropriate authentication signer
            and kwargs required to instantiate IdentityClient object.
        region: (str, optional). Defaults to `None`.
            The destination Object Storage bucket region.
            By default the value will be extracted from the `OCI_REGION_METADATA` environment variables.
        overwrite_existing_artifact: (bool, optional). Defaults to `True`.
            Overwrite target bucket artifact if exists.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Whether artifacts uploaded to object storage bucket need to be removed or not.
        parallel_process_count: (int, optional).
            The number of worker processes to use in parallel for uploading individual parts of a multipart upload.
        """
        self.auth = auth or dsc_model.auth
        if ObjectStorageDetails.is_oci_path(artifact_path):
            if not artifact_path.endswith(".zip"):
                raise ValueError(
                    f"The `artifact_path={artifact_path}` is invalid."
                    "The remote path for model artifact should be a zip archive, "
                    "e.g. `oci://<bucket_name>@<namespace>/prefix/mymodel.zip`."
                )
            if not utils.is_path_exists(uri=artifact_path, auth=self.auth):
                raise ValueError(f"The `{artifact_path}` does not exist.")
            bucket_uri = artifact_path

        if not bucket_uri:
            raise ValueError("The `bucket_uri` must be provided.")

        super().__init__(dsc_model=dsc_model, artifact_path=artifact_path)
        self.region = region or utils.extract_region(self.auth)
        self.bucket_uri = bucket_uri
        self.overwrite_existing_artifact = overwrite_existing_artifact
        self.remove_existing_artifact = remove_existing_artifact
        self._parallel_process_count = parallel_process_count

    def _upload(self):
        """Uploads model artifacts to the model catalog."""
        bucket_uri = self.bucket_uri
        self.progress.update("Copying model artifact to the Object Storage bucket")
        if not bucket_uri == self.artifact_file_path:
            bucket_uri_file_name = os.path.basename(bucket_uri)

            if not bucket_uri_file_name:
                bucket_uri = os.path.join(bucket_uri, f"{self.dsc_model.id}.zip")
            elif not bucket_uri.lower().endswith(".zip"):
                bucket_uri = f"{bucket_uri}.zip"

            if not self.overwrite_existing_artifact and utils.is_path_exists(
                uri=bucket_uri, auth=self.auth
            ):
                raise FileExistsError(
                    f"The bucket_uri=`{self.bucket_uri}` exists. Please use a new file name or "
                    "set `overwrite_existing_artifact` to `True` if you wish to overwrite."
                )

            try:
                utils.upload_to_os(
                    src_uri=self.artifact_file_path,
                    dst_uri=bucket_uri,
                    auth=self.auth,
                    parallel_process_count=self._parallel_process_count,
                    force_overwrite=self.overwrite_existing_artifact,
                    progressbar_description="Copying model artifact to the Object Storage bucket.",
                )
            except Exception as ex:
                raise RuntimeError(
                    f"Failed to upload model artifact to the given Object Storage path `{self.bucket_uri}`."
                    f"See Exception: {ex}"
                )

        self.progress.update("Exporting model artifact to the model catalog")
        self.dsc_model.export_model_artifact(bucket_uri=bucket_uri, region=self.region)

        if self.remove_existing_artifact:
            self.progress.update(
                "Removing temporary artifacts from the Object Storage bucket"
            )
            utils.remove_file(bucket_uri)
        else:
            self.progress.update()
