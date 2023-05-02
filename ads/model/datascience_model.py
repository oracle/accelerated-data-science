#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import cgi
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Union

import pandas
from ads.common import utils
from ads.config import COMPARTMENT_OCID, PROJECT_OCID
from ads.feature_engineering.schema import Schema
from ads.jobs.builders.base import Builder
from ads.model.model_metadata import (
    ModelCustomMetadata,
    ModelProvenanceMetadata,
    ModelTaxonomyMetadata,
)
from ads.model.service.oci_datascience_model import (
    ModelProvenanceNotFoundError,
    OCIDataScienceModel,
)
from ads.model.artifact_downloader import (
    LargeArtifactDownloader,
    SmallArtifactDownloader,
)
from ads.model.artifact_uploader import LargeArtifactUploader, SmallArtifactUploader

logger = logging.getLogger(__name__)


_MAX_ARTIFACT_SIZE_IN_BYTES = 2147483648  # 2GB


class ModelArtifactSizeError(Exception):   # pragma: no cover
    def __init__(self, max_artifact_size: str):
        super().__init__(
            f"The model artifacts size is greater than `{max_artifact_size}`. "
            "The `bucket_uri` needs to be specified to "
            "copy artifacts to the object storage bucket. "
            "Example: `bucket_uri=oci://<bucket_name>@<namespace>/prefix/`"
        )


class DataScienceModel(Builder):
    """Represents a Data Science Model.

    Attributes
    ----------
    id: str
        Model ID.
    project_id: str
        Project OCID.
    compartment_id: str
        Compartment OCID.
    name: str
        Model name.
    description: str
        Model description.
    freeform_tags: Dict[str, str]
        Model freeform tags.
    defined_tags: Dict[str, Dict[str, object]]
        Model defined tags.
    input_schema: ads.feature_engineering.Schema
        Model input schema.
    output_schema: ads.feature_engineering.Schema, Dict
        Model output schema.
    defined_metadata_list: ModelTaxonomyMetadata
        Model defined metadata.
    custom_metadata_list: ModelCustomMetadata
        Model custom metadata.
    provenance_metadata: ModelProvenanceMetadata
        Model provenance metadata.
    artifact: str
        The artifact location. Can be either path to folder with artifacts or
        path to zip archive.
    status: Union[str, None]
        Model status.
    model_version_set_id: str
        Model version set ID
    version_label: str
        Model version label

    Methods
    -------
    create(self, **kwargs) -> "DataScienceModel"
        Creates model.
    delete(self, delete_associated_model_deployment: Optional[bool] = False) -> "DataScienceModel":
        Removes model.
    to_dict(self) -> dict
        Serializes model to a dictionary.
    from_id(cls, id: str) -> "DataScienceModel"
        Gets an existing model by OCID.
    from_dict(cls, config: dict) -> "DataScienceModel"
        Loads model instance from a dictionary of configurations.
    upload_artifact(self, ...) -> None
        Uploads model artifacts to the model catalog.
    download_artifact(self, ...) -> None
        Downloads model artifacts from the model catalog.
    update(self, **kwargs) -> "DataScienceModel"
        Updates datascience model in model catalog.
    list(cls, compartment_id: str = None, **kwargs) -> List["DataScienceModel"]
        Lists datascience models in a given compartment.
    sync(self):
        Sync up a datascience model with OCI datascience model.
    with_project_id(self, project_id: str) -> "DataScienceModel"
        Sets the project ID.
    with_description(self, description: str) -> "DataScienceModel"
        Sets the description.
    with_compartment_id(self, compartment_id: str) -> "DataScienceModel"
        Sets the compartment ID.
    with_display_name(self, name: str) -> "DataScienceModel"
        Sets the name.
    with_freeform_tags(self, **kwargs: Dict[str, str]) -> "DataScienceModel"
        Sets freeform tags.
    with_defined_tags(self, **kwargs: Dict[str, Dict[str, object]]) -> "DataScienceModel"
        Sets defined tags.
    with_input_schema(self, schema: Union[Schema, Dict]) -> "DataScienceModel"
        Sets the model input schema.
    with_output_schema(self, schema: Union[Schema, Dict]) -> "DataScienceModel"
        Sets the model output schema.
    with_defined_metadata_list(self, metadata: Union[ModelTaxonomyMetadata, Dict]) -> "DataScienceModel"
        Sets model taxonomy (defined) metadata.
    with_custom_metadata_list(self, metadata: Union[ModelCustomMetadata, Dict]) -> "DataScienceModel"
        Sets model custom metadata.
    with_provenance_metadata(self, metadata: Union[ModelProvenanceMetadata, Dict]) -> "DataScienceModel"
        Sets model provenance metadata.
    with_artifact(self, uri: str)
        Sets the artifact location. Can be a local.
    with_model_version_set_id(self, model_version_set_id: str):
        Sets the model version set ID.
    with_version_label(self, version_label: str):
        Sets the model version label.


    Examples
    --------
    >>> ds_model = (DataScienceModel()
    ...    .with_compartment_id(os.environ["NB_SESSION_COMPARTMENT_OCID"])
    ...    .with_project_id(os.environ["PROJECT_OCID"])
    ...    .with_display_name("TestModel")
    ...    .with_description("Testing the test model")
    ...    .with_freeform_tags(tag1="val1", tag2="val2")
    ...    .with_artifact("/path/to/the/model/artifacts/"))
    >>> ds_model.create()
    >>> ds_model.status()
    >>> ds_model.with_description("new description").update()
    >>> ds_model.download_artifact("/path/to/dst/folder/")
    >>> ds_model.delete()
    >>> DataScienceModel.list()
    """

    _PREFIX = "datascience_model"

    CONST_ID = "id"
    CONST_PROJECT_ID = "projectId"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_DISPLAY_NAME = "displayName"
    CONST_DESCRIPTION = "description"
    CONST_FREEFORM_TAG = "freeformTags"
    CONST_DEFINED_TAG = "definedTags"
    CONST_INPUT_SCHEMA = "inputSchema"
    CONST_OUTPUT_SCHEMA = "outputSchema"
    CONST_CUSTOM_METADATA = "customMetadataList"
    CONST_DEFINED_METADATA = "definedMetadataList"
    CONST_PROVENANCE_METADATA = "provenanceMetadata"
    CONST_ARTIFACT = "artifact"
    CONST_MODEL_VERSION_SET_ID = "modelVersionSetId"
    CONST_MODEL_VERSION_LABEL = "versionLabel"

    attribute_map = {
        CONST_ID: "id",
        CONST_PROJECT_ID: "project_id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_DISPLAY_NAME: "display_name",
        CONST_DESCRIPTION: "description",
        CONST_FREEFORM_TAG: "freeform_tags",
        CONST_DEFINED_TAG: "defined_tags",
        CONST_INPUT_SCHEMA: "input_schema",
        CONST_OUTPUT_SCHEMA: "output_schema",
        CONST_CUSTOM_METADATA: "custom_metadata_list",
        CONST_DEFINED_METADATA: "defined_metadata_list",
        CONST_PROVENANCE_METADATA: "provenance_metadata",
        CONST_ARTIFACT: "artifact",
        CONST_MODEL_VERSION_SET_ID: "model_version_set_id",
        CONST_MODEL_VERSION_LABEL: "version_label",
    }

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        """Initializes datascience model.

        Parameters
        ----------
        spec: (Dict, optional). Defaults to None.
            Object specification.

        kwargs: Dict
            Specification as keyword arguments.
            If 'spec' contains the same key as the one in kwargs,
            the value from kwargs will be used.

            - project_id: str
            - compartment_id: str
            - name: str
            - description: str
            - defined_tags: Dict[str, Dict[str, object]]
            - freeform_tags: Dict[str, str]
            - input_schema: Union[ads.feature_engineering.Schema, Dict]
            - output_schema: Union[ads.feature_engineering.Schema, Dict]
            - defined_metadata_list: Union[ModelTaxonomyMetadata, Dict]
            - custom_metadata_list: Union[ModelCustomMetadata, Dict]
            - provenance_metadata: Union[ModelProvenanceMetadata, Dict]
            - artifact: str
        """
        super().__init__(spec=spec, **deepcopy(kwargs))
        # Reinitiate complex attributes
        self._init_complex_attributes()
        # Specify oci datascience model instance
        self.dsc_model = self._to_oci_dsc_model()

    @property
    def id(self) -> Optional[str]:
        """The model OCID."""
        if self.dsc_model:
            return self.dsc_model.id
        return None

    @property
    def status(self) -> Union[str, None]:
        """Status of the model.

        Returns
        -------
        str
            Status of the model.
        """
        if self.dsc_model:
            return self.dsc_model.status
        return None

    @property
    def kind(self) -> str:
        """The kind of the object as showing in a YAML."""
        return "datascienceModel"

    @property
    def project_id(self) -> str:
        return self.get_spec(self.CONST_PROJECT_ID)

    def with_project_id(self, project_id: str) -> "DataScienceModel":
        """Sets the project ID.

        Parameters
        ----------
        project_id: str
            The project ID.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self)
        """
        return self.set_spec(self.CONST_PROJECT_ID, project_id)

    @property
    def description(self) -> str:
        return self.get_spec(self.CONST_DESCRIPTION)

    def with_description(self, description: str) -> "DataScienceModel":
        """Sets the description.

        Parameters
        ----------
        description: str
            The description of the model.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self)
        """
        return self.set_spec(self.CONST_DESCRIPTION, description)

    @property
    def compartment_id(self) -> str:
        return self.get_spec(self.CONST_COMPARTMENT_ID)

    def with_compartment_id(self, compartment_id: str) -> "DataScienceModel":
        """Sets the compartment ID.

        Parameters
        ----------
        compartment_id: str
            The compartment ID.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self)
        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    @property
    def display_name(self) -> str:
        return self.get_spec(self.CONST_DISPLAY_NAME)

    @display_name.setter
    def display_name(self, name: str) -> "DataScienceModel":
        return self.set_spec(self.CONST_DISPLAY_NAME, name)

    def with_display_name(self, name: str) -> "DataScienceModel":
        """Sets the name.

        Parameters
        ----------
        name: str
            The name.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self)
        """
        return self.set_spec(self.CONST_DISPLAY_NAME, name)

    @property
    def freeform_tags(self) -> Dict[str, str]:
        return self.get_spec(self.CONST_FREEFORM_TAG)

    def with_freeform_tags(self, **kwargs: Dict[str, str]) -> "DataScienceModel":
        """Sets freeform tags.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self)
        """
        return self.set_spec(self.CONST_FREEFORM_TAG, kwargs)

    @property
    def defined_tags(self) -> Dict[str, Dict[str, object]]:
        return self.get_spec(self.CONST_DEFINED_TAG)

    def with_defined_tags(
        self, **kwargs: Dict[str, Dict[str, object]]
    ) -> "DataScienceModel":
        """Sets defined tags.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self)
        """
        return self.set_spec(self.CONST_DEFINED_TAG, kwargs)

    @property
    def input_schema(self) -> Schema:
        """Returns model input schema.

        Returns
        -------
        ads.feature_engineering.Schema
            Model input schema.
        """
        return self.get_spec(self.CONST_INPUT_SCHEMA)

    def with_input_schema(self, schema: Union[Schema, Dict]) -> "DataScienceModel":
        """Sets the model input schema.

        Parameters
        ----------
        schema: Union[ads.feature_engineering.Schema, Dict]
            The model input schema.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self)
        """
        if schema and isinstance(schema, Dict):
            schema = Schema.from_dict(schema)
        return self.set_spec(self.CONST_INPUT_SCHEMA, schema)

    @property
    def output_schema(self) -> Schema:
        """Returns model output schema.

        Returns
        -------
        ads.feature_engineering.Schema
            Model output schema.
        """
        return self.get_spec(self.CONST_OUTPUT_SCHEMA)

    def with_output_schema(self, schema: Union[Schema, Dict]) -> "DataScienceModel":
        """Sets the model output schema.

        Parameters
        ----------
        schema: Union[ads.feature_engineering.Schema, Dict]
            The model output schema.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self)
        """
        if schema and isinstance(schema, Dict):
            schema = Schema.from_dict(schema)
        return self.set_spec(self.CONST_OUTPUT_SCHEMA, schema)

    @property
    def defined_metadata_list(self) -> ModelTaxonomyMetadata:
        """Returns model taxonomy (defined) metadatda."""
        return self.get_spec(self.CONST_DEFINED_METADATA)

    def with_defined_metadata_list(
        self, metadata: Union[ModelTaxonomyMetadata, Dict]
    ) -> "DataScienceModel":
        """Sets model taxonomy (defined) metadata.

        Parameters
        ----------
        metadata: Union[ModelTaxonomyMetadata, Dict]
            The defined metadata.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self)
        """
        if metadata and isinstance(metadata, Dict):
            metadata = ModelTaxonomyMetadata.from_dict(metadata)
        return self.set_spec(self.CONST_DEFINED_METADATA, metadata)

    @property
    def custom_metadata_list(self) -> ModelCustomMetadata:
        """Returns model custom metadatda."""
        return self.get_spec(self.CONST_CUSTOM_METADATA)

    def with_custom_metadata_list(
        self, metadata: Union[ModelCustomMetadata, Dict]
    ) -> "DataScienceModel":
        """Sets model custom metadata.

        Parameters
        ----------
        metadata: Union[ModelCustomMetadata, Dict]
            The custom metadata.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self)
        """
        if metadata and isinstance(metadata, Dict):
            metadata = ModelCustomMetadata.from_dict(metadata)
        return self.set_spec(self.CONST_CUSTOM_METADATA, metadata)

    @property
    def provenance_metadata(self) -> ModelProvenanceMetadata:
        """Returns model provenance metadatda."""
        return self.get_spec(self.CONST_PROVENANCE_METADATA)

    def with_provenance_metadata(
        self, metadata: Union[ModelProvenanceMetadata, Dict]
    ) -> "DataScienceModel":
        """Sets model provenance metadata.

        Parameters
        ----------
        provenance_metadata: Union[ModelProvenanceMetadata, Dict]
            The provenance metadata.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self)
        """
        if metadata and isinstance(metadata, Dict):
            metadata = ModelProvenanceMetadata.from_dict(metadata)
        return self.set_spec(self.CONST_PROVENANCE_METADATA, metadata)

    @property
    def artifact(self) -> str:
        return self.get_spec(self.CONST_ARTIFACT)

    def with_artifact(self, uri: str):
        """Sets the artifact location. Can be a local.

        Parameters
        ----------
        uri: str
            Path to artifact directory or to the ZIP archive.
            It could contain a serialized model(required) as well as any files needed for deployment.
            The content of the source folder will be zipped and uploaded to the model catalog.

        Examples
        --------
        >>> .with_artifact(uri="./model1/")
        >>> .with_artifact(uri="./model1.zip")
        """
        return self.set_spec(self.CONST_ARTIFACT, uri)

    @property
    def model_version_set_id(self) -> str:
        return self.get_spec(self.CONST_MODEL_VERSION_SET_ID)

    def with_model_version_set_id(self, model_version_set_id: str):
        """Sets the model version set ID.

        Parameters
        ----------
        urmodel_version_set_idi: str
            The Model version set OCID.
        """
        return self.set_spec(self.CONST_MODEL_VERSION_SET_ID, model_version_set_id)

    @property
    def version_label(self) -> str:
        return self.get_spec(self.CONST_MODEL_VERSION_LABEL)

    def with_version_label(self, version_label: str):
        """Sets the model version label.

        Parameters
        ----------
        version_label: str
            The model version label.
        """
        return self.set_spec(self.CONST_MODEL_VERSION_LABEL, version_label)

    def create(self, **kwargs) -> "DataScienceModel":
        """Creates datascience model.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `oci.data_science.models.Model` accepts.

            In addition can be also provided the attributes listed below.

            bucket_uri: (str, optional). Defaults to None.
                The OCI Object Storage URI where model artifacts will be copied to.
                The `bucket_uri` is only necessary for uploading large artifacts which
                size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
            overwrite_existing_artifact: (bool, optional). Defaults to `True`.
                Overwrite target bucket artifact if exists.
            remove_existing_artifact: (bool, optional). Defaults to `True`.
                Wether artifacts uploaded to object storage bucket need to be removed or not.
            region: (str, optional). Defaults to `None`.
                The destination Object Storage bucket region.
                By default the value will be extracted from the `OCI_REGION_METADATA` environment variable.
            auth: (Dict, optional). Defaults to `None`.
                The default authentication is set using `ads.set_auth` API.
                If you need to override the default, use the `ads.common.auth.api_keys` or
                `ads.common.auth.resource_principal` to create appropriate authentication signer
                and kwargs required to instantiate IdentityClient object.
            timeout: (int, optional). Defaults to 10 seconds.
                The connection timeout in seconds for the client.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self)

        Raises
        ------
        ValueError
            If compartment id not provided.
            If project id not provided.
        """
        if not self.compartment_id:
            raise ValueError("Compartment id must be provided.")

        if not self.project_id:
            raise ValueError("Project id must be provided.")

        if not self.display_name:
            self.display_name = self._random_display_name()

        payload = deepcopy(self._spec)
        payload.pop("id", None)
        logger.debug(f"Creating a model with payload {payload}")

        # Create model in the model catalog
        logger.info("Saving model to the Model Catalog.")
        self.dsc_model = self._to_oci_dsc_model(**kwargs).create()

        # Create model provenance
        if self.provenance_metadata:
            logger.info("Saving model provenance metadata.")
            self.dsc_model.create_model_provenance(
                self.provenance_metadata._to_oci_metadata()
            )

        # Upload artifacts
        logger.info("Uploading model artifacts.")
        self.upload_artifact(
            bucket_uri=kwargs.pop("bucket_uri", None),
            overwrite_existing_artifact=kwargs.pop("overwrite_existing_artifact", True),
            remove_existing_artifact=kwargs.pop("remove_existing_artifact", True),
            region=kwargs.pop("region", None),
            auth=kwargs.pop("auth", None),
            timeout=kwargs.pop("timeout", None),
        )

        # Sync up model
        self.sync()
        logger.info(f"Model {self.id} has been successfully saved.")

        return self

    def upload_artifact(
        self,
        bucket_uri: Optional[str] = None,
        auth: Optional[Dict] = None,
        region: Optional[str] = None,
        overwrite_existing_artifact: Optional[bool] = True,
        remove_existing_artifact: Optional[bool] = True,
        timeout: Optional[int] = None,
    ) -> None:
        """Uploads model artifacts to the model catalog.

        Parameters
        ----------
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for uploading large artifacts which
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
        auth: (Dict, optional). Defaults to `None`.
            The default authentication is set using `ads.set_auth` API.
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
        timeout: (int, optional). Defaults to 10 seconds.
            The connection timeout in seconds for the client.
        """
        # Upload artifact to the model catalog
        if not self.artifact:
            logger.warn(
                "Model artifact location not provided. "
                "Provide the artifact location to upload artifacts to the model catalog."
            )
            return

        if timeout:
            self.dsc_model._client = None
            self.dsc_model.__class__.kwargs = {
                **(self.dsc_model.__class__.kwargs or {}),
                "timeout": timeout,
            }

        if bucket_uri or utils.folder_size(self.artifact) > _MAX_ARTIFACT_SIZE_IN_BYTES:
            if not bucket_uri:
                raise ModelArtifactSizeError(
                    max_artifact_size=utils.human_size(_MAX_ARTIFACT_SIZE_IN_BYTES)
                )

            artifact_uploader = LargeArtifactUploader(
                dsc_model=self.dsc_model,
                artifact_path=self.artifact,
                auth=auth,
                region=region,
                bucket_uri=bucket_uri,
                overwrite_existing_artifact=overwrite_existing_artifact,
                remove_existing_artifact=remove_existing_artifact,
            )
        else:
            artifact_uploader = SmallArtifactUploader(
                dsc_model=self.dsc_model,
                artifact_path=self.artifact,
            )

        artifact_uploader.upload()

    def download_artifact(
        self,
        target_dir: str,
        auth: Optional[Dict] = None,
        force_overwrite: Optional[bool] = False,
        bucket_uri: Optional[str] = None,
        region: Optional[str] = None,
        overwrite_existing_artifact: Optional[bool] = True,
        remove_existing_artifact: Optional[bool] = True,
        timeout: Optional[int] = None,
    ):
        """Downloads model artifacts from the model catalog.

        Parameters
        ----------
        target_dir: str
            The target location of model artifacts.
        auth: (Dict, optional). Defaults to `None`.
            The default authentication is set using `ads.set_auth` API.
            If you need to override the default, use the `ads.common.auth.api_keys` or
            `ads.common.auth.resource_principal` to create appropriate authentication signer
            and kwargs required to instantiate IdentityClient object.
        force_overwrite: (bool, optional). Defaults to `False`.
            Overwrite target directory if exists.
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for uploading large artifacts which
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
        region: (str, optional). Defaults to `None`.
            The destination Object Storage bucket region.
            By default the value will be extracted from the `OCI_REGION_METADATA` environment variables.
        overwrite_existing_artifact: (bool, optional). Defaults to `True`.
            Overwrite target bucket artifact if exists.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Wether artifacts uploaded to object storage bucket need to be removed or not.
        timeout: (int, optional). Defaults to 10 seconds.
            The connection timeout in seconds for the client.

        Raises
        ------
        ModelArtifactSizeError
            If model artifacts size greater than 2GB and temporary OS bucket uri not provided.
        """
        # Upload artifact to the model catalog
        if not self.artifact:
            logger.warn(
                "Model doesn't contain an artifact. "
                "The artifact needs to be uploaded to the model catalog at first. "
            )
            return

        if timeout:
            self.dsc_model._client = None
            self.dsc_model.__class__.kwargs = {
                **(self.dsc_model.__class__.kwargs or {}),
                "timeout": timeout,
            }

        artifact_info = self.dsc_model.get_artifact_info()
        artifact_size = int(artifact_info.get("content-length"))
        if not bucket_uri and artifact_size > _MAX_ARTIFACT_SIZE_IN_BYTES:
            raise ModelArtifactSizeError(utils.human_size(_MAX_ARTIFACT_SIZE_IN_BYTES))

        if artifact_size > _MAX_ARTIFACT_SIZE_IN_BYTES or bucket_uri:
            artifact_downloader = LargeArtifactDownloader(
                dsc_model=self.dsc_model,
                target_dir=target_dir,
                auth=auth,
                force_overwrite=force_overwrite,
                region=region,
                bucket_uri=bucket_uri,
                overwrite_existing_artifact=overwrite_existing_artifact,
                remove_existing_artifact=remove_existing_artifact,
            )
        else:
            artifact_downloader = SmallArtifactDownloader(
                dsc_model=self.dsc_model,
                target_dir=target_dir,
                force_overwrite=force_overwrite,
            )

        artifact_downloader.download()

    def update(self, **kwargs) -> "DataScienceModel":
        """Updates datascience model in model catalog.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `oci.data_science.models.Model` accepts.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self).
        """
        if not self.id:
            logger.warn(
                "Model needs to be saved to the model catalog before it can be updated."
            )
            return

        logger.debug(f"Updating a model with payload {self._spec}")
        logger.info(f"Updating model {self.id} in the Model Catalog.")
        self.dsc_model = self._to_oci_dsc_model(**kwargs).update()

        logger.debug(f"Updating a model provenance metadata {self.provenance_metadata}")
        try:
            self.dsc_model.get_model_provenance()
            self.dsc_model.update_model_provenance(
                self.provenance_metadata._to_oci_metadata()
            )
        except ModelProvenanceNotFoundError:
            self.dsc_model.create_model_provenance(
                self.provenance_metadata._to_oci_metadata()
            )

        return self.sync()

    def delete(
        self,
        delete_associated_model_deployment: Optional[bool] = False,
    ) -> "DataScienceModel":
        """Removes model from the model catalog.

        Parameters
        ----------
        delete_associated_model_deployment: (bool, optional). Defaults to `False`.
            Whether associated model deployments need to be deleted or not.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self).
        """
        self.dsc_model.delete(delete_associated_model_deployment)
        return self.sync()

    @classmethod
    def list(
        cls, compartment_id: str = None, project_id: str = None, **kwargs
    ) -> List["DataScienceModel"]:
        """Lists datascience models in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        project_id: (str, optional). Defaults to `None`.
            The project OCID.
        kwargs
            Additional keyword arguments for filtering models.

        Returns
        -------
        List[DataScienceModel]
            The list of the datascience models.
        """
        return [
            cls()._update_from_oci_dsc_model(model)
            for model in OCIDataScienceModel.list_resource(
                compartment_id, project_id=project_id, **kwargs
            )
        ]

    @classmethod
    def list_df(
        cls, compartment_id: str = None, project_id: str = None, **kwargs
    ) -> "pandas.DataFrame":
        """Lists datascience models in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        project_id: (str, optional). Defaults to `None`.
            The project OCID.
        kwargs
            Additional keyword arguments for filtering models.

        Returns
        -------
        pandas.DataFrame
            The list of the datascience models in a pandas dataframe format.
        """
        records = []
        for model in OCIDataScienceModel.list_resource(
            compartment_id, project_id=project_id, **kwargs
        ):
            records.append(
                {
                    "id": f"...{model.id[-6:]}",
                    "display_name": model.display_name,
                    "description": model.description,
                    "time_created": model.time_created.strftime(utils.date_format),
                    "lifecycle_state": model.lifecycle_state,
                    "created_by": f"...{model.created_by[-6:]}",
                    "compartment_id": f"...{model.compartment_id[-6:]}",
                    "project_id": f"...{model.project_id[-6:]}",
                }
            )
        return pandas.DataFrame.from_records(records)

    @classmethod
    def from_id(cls, id: str) -> "DataScienceModel":
        """Gets an existing model by OCID.

        Parameters
        ----------
        id: str
            The model OCID.

        Returns
        -------
        DataScienceModel
            An instance of DataScienceModel.
        """
        return cls()._update_from_oci_dsc_model(OCIDataScienceModel.from_id(id))

    def sync(self):
        """Sync up a datascience model with OCI datascience model."""
        return self._update_from_oci_dsc_model(OCIDataScienceModel.from_id(self.id))

    def _init_complex_attributes(self):
        """Initiates complex attributes."""
        self.with_custom_metadata_list(self.custom_metadata_list)
        self.with_defined_metadata_list(self.defined_metadata_list)
        self.with_provenance_metadata(self.provenance_metadata)
        self.with_input_schema(self.input_schema)
        self.with_output_schema(self.output_schema)

    def _to_oci_dsc_model(self, **kwargs):
        """Creates an `OCIDataScienceModel` instance from the  `DataScienceModel`.

        kwargs
            Additional kwargs arguments.
            Can be any attribute that `oci.data_science.models.Model` accepts.

        Returns
        -------
        OCIDataScienceModel
            The instance of the OCIDataScienceModel.
        """
        COMPLEX_ATTRIBUTES_CONVERTER = {
            self.CONST_INPUT_SCHEMA: "to_json",
            self.CONST_OUTPUT_SCHEMA: "to_json",
            self.CONST_CUSTOM_METADATA: "_to_oci_metadata",
            self.CONST_DEFINED_METADATA: "_to_oci_metadata",
            self.CONST_PROVENANCE_METADATA: "_to_oci_metadata",
        }
        dsc_spec = {}
        for infra_attr, dsc_attr in self.attribute_map.items():
            value = self.get_spec(infra_attr)
            if infra_attr in COMPLEX_ATTRIBUTES_CONVERTER and value:
                dsc_spec[dsc_attr] = getattr(
                    self.get_spec(infra_attr), COMPLEX_ATTRIBUTES_CONVERTER[infra_attr]
                )()
            else:
                dsc_spec[dsc_attr] = value

        dsc_spec.update(**kwargs)
        return OCIDataScienceModel(**dsc_spec)

    def _update_from_oci_dsc_model(
        self, dsc_model: OCIDataScienceModel
    ) -> "DataScienceModel":
        """Update the properties from an OCIDataScienceModel object.

        Parameters
        ----------
        dsc_model: OCIDataScienceModel
            An instance of OCIDataScienceModel.

        Returns
        -------
        DataScienceModel
            The DataScienceModel instance (self).
        """
        COMPLEX_ATTRIBUTES_CONVERTER = {
            self.CONST_INPUT_SCHEMA: Schema.from_json,
            self.CONST_OUTPUT_SCHEMA: Schema.from_json,
            self.CONST_CUSTOM_METADATA: ModelCustomMetadata._from_oci_metadata,
            self.CONST_DEFINED_METADATA: ModelTaxonomyMetadata._from_oci_metadata,
        }

        # Update the main properties
        self.dsc_model = dsc_model
        for infra_attr, dsc_attr in self.attribute_map.items():
            value = utils.get_value(dsc_model, dsc_attr)
            if value:
                if infra_attr in COMPLEX_ATTRIBUTES_CONVERTER:
                    value = COMPLEX_ATTRIBUTES_CONVERTER[infra_attr](value)
                self.set_spec(infra_attr, value)

        # Update provenance metadata
        try:
            self.set_spec(
                self.CONST_PROVENANCE_METADATA,
                ModelProvenanceMetadata._from_oci_metadata(
                    self.dsc_model.get_model_provenance()
                ),
            )
        except ModelProvenanceNotFoundError:
            pass

        # Update artifact info
        try:
            artifact_info = self.dsc_model.get_artifact_info()
            _, file_name_info = cgi.parse_header(artifact_info["Content-Disposition"])
            self.set_spec(self.CONST_ARTIFACT, file_name_info["filename"])
        except:
            pass

        return self

    def to_dict(self) -> Dict:
        """Serializes model to a dictionary.

        Returns
        -------
        dict
            The model serialized as a dictionary.
        """
        spec = deepcopy(self._spec)
        for key, value in spec.items():
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            spec[key] = value

        return {
            "kind": self.kind,
            "type": self.type,
            "spec": utils.batch_convert_case(spec, "camel"),
        }

    @classmethod
    def from_dict(cls, config: Dict) -> "DataScienceModel":
        """Loads model instance from a dictionary of configurations.

        Parameters
        ----------
        config: Dict
            A dictionary of configurations.

        Returns
        -------
        DataScienceModel
            The model instance.
        """
        return cls(spec=utils.batch_convert_case(deepcopy(config["spec"]), "snake"))

    def _random_display_name(self):
        """Generates a random display name."""
        return f"{self._PREFIX}-{utils.get_random_name_for_resource()}"

    def _load_default_properties(self) -> Dict:
        """Load default properties from environment variables, notebook session, etc.

        Returns
        -------
        Dict
            A dictionary of default properties.
        """
        defaults = super()._load_default_properties()
        compartment_ocid = COMPARTMENT_OCID
        if compartment_ocid:
            defaults[self.CONST_COMPARTMENT_ID] = compartment_ocid
        if PROJECT_OCID:
            defaults[self.CONST_PROJECT_ID] = PROJECT_OCID
        defaults[self.CONST_DISPLAY_NAME] = self._random_display_name()

        return defaults

    def __getattr__(self, item):
        if f"with_{item}" in self.__dir__():
            return self.get_spec(item)
        raise AttributeError(f"Attribute {item} not found.")
