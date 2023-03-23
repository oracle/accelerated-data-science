#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict, Optional

from ads.common import utils
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
        if not os.path.exists(artifact_path):
            raise ValueError(f"The `{artifact_path}` does not exist")

        self.dsc_model = dsc_model
        self.artifact_path = artifact_path
        self.artifact_zip_path = None
        self.progress = None

    def upload(self):
        """Uploads model artifacts."""
        try:
            with utils.get_progress_bar(
                ArtifactUploader.PROGRESS_STEPS_COUNT + self.PROGRESS_STEPS_COUNT
            ) as progress:
                self.progress = progress
                self.progress.update("Preparing model artifacts ZIP archive.")
                self._prepare_artiact_tmp_zip()
                self.progress.update("Uploading model artifacts.")
                self._upload()
                self.progress.update(
                    "Uploading model artifacts has been successfully completed."
                )
                self.progress.update("Done.")
        except Exception:
            raise
        finally:
            self._remove_artiact_tmp_zip()

    def _prepare_artiact_tmp_zip(self) -> str:
        """Prepares model artifacts ZIP archive.

        Parameters
        ----------
        progress: (TqdmProgressBar, optional). Defaults to `None`.
            The progress indicator.

        Returns
        -------
        str
            Path to the model artifact ZIP archive.
        """
        if os.path.isfile(self.artifact_path) and self.artifact_path.lower().endswith(
            ".zip"
        ):
            self.artifact_zip_path = self.artifact_path
        else:
            self.artifact_zip_path = model_utils.zip_artifact(
                artifact_dir=self.artifact_path
            )
        return self.artifact_zip_path

    def _remove_artiact_tmp_zip(self):
        """Removes temporary created artifact zip archive."""
        if (
            self.artifact_zip_path
            and self.artifact_zip_path.lower() != self.artifact_path.lower()
        ):
            shutil.rmtree(self.artifact_zip_path, ignore_errors=True)

    @abstractmethod
    def _upload(self):
        """Uploads model artifacts. Needs to be implemented in a child class."""


class SmallArtifactUploader(ArtifactUploader):
    PROGRESS_STEPS_COUNT = 1

    def _upload(self):
        """Uploads model artifacts to the model catalog."""
        self.progress.update("Uploading model artifacts to the catalog")
        with open(self.artifact_zip_path, "rb") as file_data:
            self.dsc_model.create_model_artifact(file_data)


class LargeArtifactUploader(ArtifactUploader):
    PROGRESS_STEPS_COUNT = 4

    def __init__(
        self,
        dsc_model: OCIDataScienceModel,
        artifact_path: str,
        bucket_uri: str,
        auth: Optional[Dict] = None,
        region: Optional[str] = None,
        overwrite_existing_artifact: Optional[bool] = True,
        remove_existing_artifact: Optional[bool] = True,
    ):
        """Initializes `LargeArtifactUploader` instance.

        Parameters
        ----------
        dsc_model: OCIDataScienceModel
            The data scince model instance.
        artifact_path: str
            The model artifact location.
        bucket_uri: str
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for uploading large artifacts which
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
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
            Wether artifacts uploaded to object storage bucket need to be removed or not.
        """
        if not bucket_uri:
            raise ValueError("The `bucket_uri` must be provided.")

        super().__init__(dsc_model=dsc_model, artifact_path=artifact_path)
        self.auth = auth or dsc_model.auth
        self.region = region or utils.extract_region(self.auth)
        self.bucket_uri = bucket_uri
        self.overwrite_existing_artifact = overwrite_existing_artifact
        self.remove_existing_artifact = remove_existing_artifact

    def _upload(self):
        """Uploads model artifacts to the model catalog."""
        self.progress.update("Copying model artifact to the Object Storage bucket")

        try:
            bucket_uri = self.bucket_uri
            bucket_uri_file_name = os.path.basename(bucket_uri)

            if not bucket_uri_file_name:
                bucket_uri = os.path.join(bucket_uri, f"{self.dsc_model.id}.zip")
            elif not bucket_uri.lower().endswith(".zip"):
                bucket_uri = f"{bucket_uri}.zip"

            bucket_file_name = utils.copy_file(
                self.artifact_zip_path,
                bucket_uri,
                force_overwrite=self.overwrite_existing_artifact,
                auth=self.auth,
                progressbar_description="Copying model artifact to the Object Storage bucket",
            )
        except FileExistsError:
            raise FileExistsError(
                f"The `{self.bucket_uri}` exists. Please use a new file name or "
                "set `overwrite_existing_artifact` to `True` if you wish to overwrite."
            )
        self.progress.update("Exporting model artifact to the model catalog")
        self.dsc_model.export_model_artifact(
            bucket_uri=bucket_file_name, region=self.region
        )

        if self.remove_existing_artifact:
            self.progress.update(
                "Removing temporary artifacts from the Object Storage bucket"
            )
            utils.remove_file(bucket_uri)
        else:
            self.progress.update()
