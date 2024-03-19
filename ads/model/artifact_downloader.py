#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Optional
from zipfile import ZipFile

from ads.common import utils
from ads.common.utils import extract_region
from ads.model.service.oci_datascience_model import OCIDataScienceModel
from ads.common.object_storage_details import ObjectStorageDetails


class ArtifactDownloader(ABC):
    """The abstract class to download model artifacts."""

    PROGRESS_STEPS_COUNT = 1

    def __init__(
        self,
        dsc_model: OCIDataScienceModel,
        target_dir: str,
        force_overwrite: Optional[bool] = False,
    ):
        """Initializes `ArtifactDownloader` instance.

        Parameters
        ----------
        dsc_model: OCIDataScienceModel
            The data scince model instance.
        target_dir: str
            The target location of model after download.
        force_overwrite: bool
            Overwrite target_dir if exists.
        """
        self.dsc_model = dsc_model
        self.target_dir = target_dir
        self.force_overwrite = force_overwrite
        self.progress = None

    def download(self):
        """Downloads model artifacts.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If target directory does not exist.
        """
        if os.path.exists(self.target_dir) and len(os.listdir(self.target_dir)) > 0:
            if not self.force_overwrite:
                raise ValueError(
                    f"The `{self.target_dir}` directory already exists. "
                    "Set `force_overwrite` to `True` if you wish to overwrite."
                )
            shutil.rmtree(self.target_dir)
        os.makedirs(self.target_dir, exist_ok=True)
        with utils.get_progress_bar(
            ArtifactDownloader.PROGRESS_STEPS_COUNT + self.PROGRESS_STEPS_COUNT
        ) as progress:
            self.progress = progress
            self._download()
            self.progress.update(
                "Downloading model artifacts has been successfully completed."
            )
            self.progress.update("Done.")

    @abstractmethod
    def _download(self):
        """Downloads model artifacts."""


class SmallArtifactDownloader(ArtifactDownloader):
    PROGRESS_STEPS_COUNT = 3

    def _download(self):
        """Downloads model artifacts."""
        self.progress.update("Importing model artifacts from catalog")

        artifact_info = self.dsc_model.get_artifact_info()
        artifact_name = artifact_info["Content-Disposition"].replace(
            "attachment; filename=", ""
        )
        _, file_extension = os.path.splitext(artifact_name)
        file_extension = file_extension.lower() if file_extension else ".zip"

        file_content = self.dsc_model.get_model_artifact_content()
        self.progress.update("Copying model artifacts to the artifact directory")

        file_name = (
            "model_description" if file_extension == ".json" else str(uuid.uuid4())
        )
        artifact_file_path = os.path.join(
            self.target_dir, f"{file_name}{file_extension}"
        )
        with open(artifact_file_path, "wb") as _file:
            _file.write(file_content)

        if file_extension == ".zip":
            self.progress.update("Extracting model artifacts")
            with ZipFile(artifact_file_path) as _file:
                _file.extractall(self.target_dir)
            utils.remove_file(artifact_file_path)


class LargeArtifactDownloader(ArtifactDownloader):
    PROGRESS_STEPS_COUNT = 4

    def __init__(
        self,
        dsc_model: OCIDataScienceModel,
        target_dir: str,
        auth: Optional[Dict] = None,
        force_overwrite: Optional[bool] = False,
        region: Optional[str] = None,
        bucket_uri: Optional[str] = None,
        overwrite_existing_artifact: Optional[bool] = True,
        remove_existing_artifact: Optional[bool] = True,
        model_file_description: Optional[dict] = None,
    ):
        """Initializes `LargeArtifactDownloader` instance.

        Parameters
        ----------
        dsc_model: OCIDataScienceModel
            The data scince model instance.
        target_dir: str
            The target location of model after download.
        auth: (Dict, optional). Defaults to `None`.
            The default authetication is set using `ads.set_auth` API.
            If you need to override the default, use the `ads.common.auth.api_keys` or
            `ads.common.auth.resource_principal` to create appropriate authentication signer
            and kwargs required to instantiate IdentityClient object.
        force_overwrite: (bool, optional). Defaults to `False`.
            Overwrite target_dir if exists.
        region: (str, optional). Defaults to `None`.
            The destination Object Storage bucket region.
            By default the value will be extracted from the `OCI_REGION_METADATA` environment variables.
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for uploading large artifacts which
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
        overwrite_existing_artifact: (bool, optional). Defaults to `True`.
            Overwrite target bucket artifact if exists.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Wether artifacts uploaded to object storage bucket need to be removed or not.
        model_file_description: (dict, optional). Defaults to None.
            Contains object path details for models created by reference.

        """
        super().__init__(
            dsc_model=dsc_model, target_dir=target_dir, force_overwrite=force_overwrite
        )
        self.auth = auth or dsc_model.auth
        self.region = region or extract_region(self.auth)
        self.bucket_uri = bucket_uri
        self.overwrite_existing_artifact = overwrite_existing_artifact
        self.remove_existing_artifact = remove_existing_artifact
        self.model_file_description = model_file_description

    def _download(self):
        """Downloads model artifacts."""
        self.progress.update(f"Importing model artifacts from catalog")

        if self.dsc_model.is_model_by_reference() and self.model_file_description:
            self.download_from_model_file_description()
            self.progress.update()
            return

        bucket_uri = self.bucket_uri

        if not os.path.basename(bucket_uri):
            bucket_uri = os.path.join(bucket_uri, f"{self.dsc_model.id}.zip")
        elif not bucket_uri.lower().endswith(".zip"):
            bucket_uri = f"{bucket_uri}.zip"

        self.dsc_model.import_model_artifact(bucket_uri=bucket_uri, region=self.region)
        self.progress.update("Copying model artifacts to the artifact directory")
        zip_file_path = os.path.join(self.target_dir, f"{str(uuid.uuid4())}.zip")
        zip_file_path = utils.copy_file(
            uri_src=bucket_uri,
            uri_dst=zip_file_path,
            auth=self.auth,
            progressbar_description="Copying model artifacts to the artifact directory",
        )
        self.progress.update("Extracting model artifacts")
        with ZipFile(zip_file_path) as zip_file:
            zip_file.extractall(self.target_dir)

        utils.remove_file(zip_file_path)
        if self.remove_existing_artifact:
            self.progress.update(
                "Removing temporary artifacts from the Object Storage bucket"
            )
            utils.remove_file(bucket_uri)
        else:
            self.progress.update()

    def download_from_model_file_description(self):
        """Helper function to download the objects using model file description content to the target directory."""

        models = self.model_file_description["models"]
        total_size = 0

        for model in models:
            namespace, bucket_name, prefix = (
                model["namespace"],
                model["bucketName"],
                model["prefix"],
            )
            bucket_uri = f"oci://{bucket_name}@{namespace}/{prefix}"

            message = f"Copying model artifacts by reference from {bucket_uri} to {self.target_dir}"
            self.progress.update(message)

            objects = model["objects"]
            os_details_list = list()

            for obj in objects:
                name = obj["name"]
                version = None if obj["version"] == "" else obj["version"]
                size = obj["sizeInBytes"]
                if size == 0:
                    continue
                total_size += size
                object_uri = f"oci://{bucket_name}@{namespace}/{name}"

                os_details = ObjectStorageDetails.from_path(object_uri)
                os_details.version = version
                os_details_list.append(os_details)
            try:
                ObjectStorageDetails.from_path(
                    bucket_uri
                ).bulk_download_from_object_storage(
                    paths=os_details_list,
                    target_dir=self.target_dir,
                    progress_bar=self.progress,
                )
            except Exception as ex:
                raise RuntimeError(
                    f"Failed to download model artifact by reference from the given Object Storage path `{bucket_uri}`."
                    f"See Exception: {ex}"
                )
