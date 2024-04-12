#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import time
from functools import wraps
from io import BytesIO
from typing import Callable, Dict, List, Optional

import oci.data_science
from ads.common import utils
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.oci_datascience import OCIDataScienceMixin
from ads.common.oci_mixin import OCIWorkRequestMixin
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.common.utils import extract_region
from ads.common.work_request import DataScienceWorkRequest
from ads.model.deployment import ModelDeployment
from oci.data_science.models import (
    ArtifactExportDetailsObjectStorage,
    ArtifactImportDetailsObjectStorage,
    CreateModelDetails,
    ExportModelArtifactDetails,
    ImportModelArtifactDetails,
    UpdateModelDetails,
    WorkRequest,
)
from oci.exceptions import ServiceError

logger = logging.getLogger(__name__)

_REQUEST_INTERVAL_IN_SEC = 3

MODEL_NEEDS_TO_BE_SAVED = (
    "Model needs to be saved to the Model Catalog before it can be accessed."
)

MODEL_BY_REFERENCE_DESC = "modelDescription"


class ModelProvenanceNotFoundError(Exception):  # pragma: no cover
    pass


class ModelArtifactNotFoundError(Exception):  # pragma: no cover
    pass


class ModelNotSavedError(Exception):  # pragma: no cover
    pass


class ModelWithActiveDeploymentError(Exception):  # pragma: no cover
    pass


def check_for_model_id(msg: str = MODEL_NEEDS_TO_BE_SAVED):
    """The decorator helping to check if the ID attribute sepcified for a datascience model.

    Parameters
    ----------
    msg: str
        The message that will be thrown.

    Raises
    ------
    ModelNotSavedError
        In case if the ID attribute not specified.

    Examples
    --------
    >>> @check_for_id(msg="Some message.")
    ... def test_function(self, name: str, last_name: str)
    ...     pass
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.id:
                raise ModelNotSavedError(msg)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class OCIDataScienceModel(
    OCIDataScienceMixin,
    OCIWorkRequestMixin,
    oci.data_science.models.Model,
):
    """Represents an OCI Data Science Model.
    This class contains all attributes of the `oci.data_science.models.Model`.
    The main purpose of this class is to link the `oci.data_science.models.Model`
    and the related client methods.
    Linking the `Model` (payload) to Create/Update/Get/List/Delete methods.

    The `OCIDataScienceModel` can be initialized by unpacking the properties stored in a dictionary:

    .. code-block:: python

        properties = {
            "compartment_id": "<compartment_ocid>",
            "name": "<model_name>",
            "description": "<model_description>",
        }
        ds_model = OCIDataScienceModel(**properties)

    The properties can also be OCI REST API payload, in which the keys are in camel format.

    .. code-block:: python

        payload = {
            "compartmentId": "<compartment_ocid>",
            "name": "<model_name>",
            "description": "<model_description>",
        }
        ds_model = OCIDataScienceModel(**payload)

    Methods
    -------
    create(self) -> "OCIDataScienceModel"
        Creates datascience model in model catalog.
    create_model_provenance(self, model_provenance: ModelProvenance) -> oci.data_science.models.ModelProvenance:
        Creates model provenance metadata.
    def update_model_provenance(self, ModelProvenance) -> oci.data_science.models.ModelProvenance:
        Updates model provenance metadata.
    get_model_provenance(self) -> oci.data_science.models.ModelProvenance:
        Gets model provenance metadata.
    get_artifact_info(self) -> Dict:
        Gets model artifact attachment information.
    def get_model_artifact_content(self) -> BytesIO:
        Gets model artifact content.
    create_model_artifact(self, bytes_content: BytesIO) -> None:
        Creates model artifact for specified model.
    import_model_artifact(self, bucket_uri: str, region: str = None) -> None:
        Imports model artifact content from the model catalog.
    export_model_artifact(self, bucket_uri: str, region: str = None):
        Exports model artifact to the model catalog.
    update(self) -> "OCIDataScienceModel":
        Updates datascience Model.
    delete(self, delete_associated_model_deployment: Optional[bool] = False) -> "OCIDataScienceModel":
        Deletes detascience Model.
    model_deployment(self, ...) -> List:
        Gets the list of model deployments by model ID across the compartments.
    from_id(cls, ocid: str) -> "OCIDataScienceModel":
        Gets model by OCID.

    Examples
    --------
    >>> oci_model = OCIDataScienceModel.from_id(<model_ocid>)
    >>> oci_model.model_deployment()
    >>> oci_model.get_model_provenance()
    >>> oci_model.description = "A brand new description"
    ... oci_model.update()
    >>> oci_model.sync()
    >>> oci_model.get_artifact_info()
    """

    def create(self) -> "OCIDataScienceModel":
        """Creates datascience model in model catalog.

        Returns
        -------
        OCIDataScienceModel
            The `OCIDataScienceModel` instance (self), which allows chaining additional method.
        """
        if not self.compartment_id:
            raise ValueError("The `compartment_id` must be specified.")

        if not self.project_id:
            raise ValueError("The `project_id` must be specified.")

        return self.update_from_oci_model(
            self.client.create_model(self.to_oci_model(CreateModelDetails)).data
        )

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before the provenance metadata can be created."
    )
    def create_model_provenance(
        self, model_provenance: oci.data_science.models.ModelProvenance
    ) -> oci.data_science.models.ModelProvenance:
        """Creates model provenance metadata.

        Parameters
        ----------
        model_provenance: oci.data_science.models.ModelProvenance
            OCI model provenance metadata.

        Returns
        -------
        oci.data_science.models.ModelProvenance
            The OCI model provenance object.
        """
        return self.client.create_model_provenance(self.id, model_provenance).data

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before the provenance metadata can be updated."
    )
    def update_model_provenance(
        self, model_provenance: oci.data_science.models.ModelProvenance
    ) -> oci.data_science.models.ModelProvenance:
        """Updates model provenance metadata.

        Parameters
        ----------
        model_provenance: oci.data_science.models.ModelProvenance
            OCI model provenance metadata.

        Returns
        -------
        oci.data_science.models.ModelProvenance
            The OCI model provenance object.
        """
        return self.client.update_model_provenance(self.id, model_provenance).data

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before the provenance metadata can be read."
    )
    def get_model_provenance(self) -> oci.data_science.models.ModelProvenance:
        """Gets model provenance metadata.

        Returns
        -------
        oci.data_science.models.ModelProvenance
            OCI model provenance metadata.

        Raises
        ------
        ModelProvenanceNotFoundError
            If model provenance not found.
        """
        try:
            return self.client.get_model_provenance(self.id).data
        except ServiceError as ex:
            if ex.status == 404:
                raise ModelProvenanceNotFoundError()
            raise

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before the artifact information can be read."
    )
    def get_artifact_info(self) -> Dict:
        """Gets model artifact attachment information.

        Returns
        -------
        Dict
            The model artifact attachement info.
            Example:
            {
                'Date': 'Sun, 13 Nov 2022 06:01:27 GMT',
                'opc-request-id': 'E4F7',
                'ETag': '77156317-8bb9-4c4a-882b-0d85f8140d93',
                'Content-Disposition': 'attachment; filename=artifact.zip',
                'Last-Modified': 'Sun, 09 Oct 2022 16:50:14 GMT',
                'Content-Type': 'application/json',
                'Content-MD5': 'orMy3Gs386GZLjYWATJWuA==',
                'X-Content-Type-Options': 'nosniff',
                'Content-Length': '4029958'
            }

        Raises
        ------
        ModelArtifactNotFoundError
            If model artifact attchment not found.
        """
        try:
            return self.client.head_model_artifact(model_id=self.id).headers
        except ServiceError as ex:
            if ex.status == 404:
                raise ModelArtifactNotFoundError()
        return {}

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before the artifact content can be read."
    )
    def get_model_artifact_content(self) -> BytesIO:
        """Gets model artifact content.
        Can only be used to the small artifacts, which size is less than 2GB.
        For the large artifacts needs to be used a `import_model_artifact` method.

        Returns
        -------
        BytesIO
            Object with data of type stream.

        Raises
        ------
        ModelArtifactNotFoundError
            If model artifact not found.

        """
        try:
            return self.client.get_model_artifact_content(model_id=self.id).data.content
        except ServiceError as ex:
            if ex.status == 404:
                raise ModelArtifactNotFoundError()

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before the artifact can be created."
    )
    def create_model_artifact(
        self,
        bytes_content: BytesIO,
        extension: str = None,
    ) -> None:
        """Creates model artifact for specified model.

        Parameters
        ----------
        bytes_content: BytesIO
            Model artifacts to upload.
        extension: str
            File extension, defaults to zip
        """
        ext = ".json" if extension and extension.lower() == ".json" else ".zip"
        self.client.create_model_artifact(
            self.id,
            bytes_content,
            content_disposition=f'attachment; filename="{self.id}{ext}"',
        )

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before the artifact can be created."
    )
    def import_model_artifact(self, bucket_uri: str, region: str = None) -> None:
        """Imports model artifact content from the model catalog.
        Requires to provide an Object Storage bucket for transitional saving artifacts.
        This method can be used either for small or large artifacts.

        Parameters
        ----------
        bucket_uri: str
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for downloading large artifacts which
            size is greater than 2GB.
            Example: `oci://<bucket_name>@<namespace>/prefix/`.
        region: (str, optional). Defaults to `None`.
            The destination Object Storage bucket region.
            By default the value will be extracted from the `OCI_REGION_METADATA` environment variable.

        Returns
        -------
        None

        Raises
        ------
        ModelArtifactNotFoundError
            If model artifact not found.
        """
        bucket_details = ObjectStorageDetails.from_path(bucket_uri)
        region = region or extract_region(self.auth)
        try:
            work_request_id = self.client.import_model_artifact(
                model_id=self.id,
                import_model_artifact_details=ImportModelArtifactDetails(
                    artifact_import_details=ArtifactImportDetailsObjectStorage(
                        namespace=bucket_details.namespace,
                        destination_bucket=bucket_details.bucket,
                        destination_object_name=bucket_details.filepath,
                        destination_region=region,
                    )
                ),
            ).headers["opc-work-request-id"]

            # Show progress of importing artifacts
            DataScienceWorkRequest(work_request_id).wait_work_request(
                progress_bar_description="Importing model artifacts."
            )
        except ServiceError as ex:
            if ex.status == 404:
                raise ModelArtifactNotFoundError()

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before the artifact can be exported."
    )
    def export_model_artifact(self, bucket_uri: str, region: str = None):
        """Exports model artifact to the model catalog.
        Can be used for any model artifact. Requires to provide an Object Storage bucket,
        for transitional saving artifacts. For the small artifacts use `create_model_artifact` method.

        Parameters
        ----------
        bucket_uri: str
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for downloading large artifacts which
            size is greater than 2GB.
            Example: `oci://<bucket_name>@<namespace>/prefix/`.
        region: (str, optional). Defaults to `None`.
            The destination Object Storage bucket region.
            By default the value will be extracted from the `OCI_REGION_METADATA` environment variables.

        Returns
        -------
        None
        """
        bucket_details = ObjectStorageDetails.from_path(bucket_uri)
        region = region or extract_region(self.auth)

        work_request_id = self.client.export_model_artifact(
            model_id=self.id,
            export_model_artifact_details=ExportModelArtifactDetails(
                artifact_export_details=ArtifactExportDetailsObjectStorage(
                    namespace=bucket_details.namespace,
                    source_bucket=bucket_details.bucket,
                    source_object_name=bucket_details.filepath,
                    source_region=region,
                )
            ),
        ).headers["opc-work-request-id"]

        # Show progress of exporting model artifacts
        DataScienceWorkRequest(work_request_id).wait_work_request(
            progress_bar_description="Exporting model artifacts."
        )

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before it can be updated."
    )
    def update(self) -> "OCIDataScienceModel":
        """Updates datascience Model.

        Returns
        -------
        OCIDataScienceModel
            The `OCIDataScienceModel` instance (self).
        """

        model_details = self.to_oci_model(UpdateModelDetails)

        # Clean up the model version set, otherwise it throws an error that model is already
        # associated with the model version set.
        model_details.model_version_set_id = None
        return self.update_from_oci_model(
            self.client.update_model(self.id, model_details).data
        )

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before it can be deleted."
    )
    def delete(
        self,
        delete_associated_model_deployment: Optional[bool] = False,
    ) -> "OCIDataScienceModel":
        """Deletes detascience Model.

        Parameters
        ----------
        delete_associated_model_deployment: (bool, optional). Defaults to `False`.
            Whether associated model deployments need to be deleted or not.

        Returns
        -------
        OCIDataScienceModel
            The `OCIDataScienceModel` instance (self).

        Raises
        ------
        ModelWithActiveDeploymentError
            If model has active deployments and `delete_associated_model_deployment` set to `False`.
        """
        active_deployments = tuple(
            item for item in self.model_deployment() if item.lifecycle_state == "ACTIVE"
        )

        if len(active_deployments) > 0:
            if not delete_associated_model_deployment:
                raise ModelWithActiveDeploymentError()

            logger.info(
                f"Deleting model deployments associated with the model `{self.id}`."
            )
            for oci_model_deployment in active_deployments:
                logger.info(
                    f"Deleting model deployment `{oci_model_deployment.identifier}`."
                )
                ModelDeployment.from_id(oci_model_deployment.identifier).delete()

        logger.info(f"Deleting model `{self.id}`.")
        self.client.delete_model(self.id)
        return self.sync()

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before the associated model deployments can be read."
    )
    def model_deployment(
        self,
        config: Optional[Dict] = None,
        tenant_id: Optional[str] = None,
        limit: Optional[int] = 500,
        page: Optional[str] = None,
        **kwargs: Dict,
    ) -> List:
        """
        Gets the list of model deployments by model ID across the compartments.

        Parameters
        ----------
        model_id: str
            The model ID.
        config: (Dict, optional). Defaults to `None`.
            Configuration keys and values as per SDK and Tool Configuration.
            The from_file() method can be used to load configuration from a file.
            Alternatively, a dict can be passed. You can validate_config the dict
            using validate_config(). Defaults to None.
        tenant_id: (str, optional). Defaults to `None`.
            The tenancy ID, which can be used to specify a different tenancy
            (for cross-tenancy authorization) when searching for resources in
            a different tenancy. Defaults to None.
        limit: (int, optional). Defaults to `None`.
            The maximum number of items to return. The value must be between
            1 and 1000. Defaults to 500.
        page: (str, optional). Defaults to `None`.
            The page at which to start retrieving results.

        Returns
        -------
            The list of model deployments associated with given model ID.
        """
        query = f"query datasciencemodeldeployment resources where ModelId='{self.id}'"
        return OCIResource.search(
            query,
            type=SEARCH_TYPE.STRUCTURED,
            config=config,
            tenant_id=tenant_id,
            limit=limit,
            page=page,
            **kwargs,
        )

    @classmethod
    def from_id(cls, ocid: str) -> "OCIDataScienceModel":
        """Gets model by OCID.

        Parameters
        ----------
        ocid: str
            The OCID of the datascience model.

        Returns
        -------
        OCIDataScienceModel
            An instance of `OCIDataScienceModel`.
        """
        if not ocid:
            raise ValueError("Model OCID not provided.")
        return super().from_ocid(ocid)

    def is_model_by_reference(self):
        """Checks if model is created by reference
        Returns
        -------
            bool flag denoting whether model was created by reference.

        """
        if self.custom_metadata_list:
            for metadata in self.custom_metadata_list:
                if (
                    metadata.key == MODEL_BY_REFERENCE_DESC
                    and metadata.value.lower() == "true"
                ):
                    return True
        return False
