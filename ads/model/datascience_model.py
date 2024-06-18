#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import cgi
import json
import logging
import os
import shutil
import tempfile
from copy import deepcopy
from typing import Dict, List, Optional, Union, Tuple

import pandas
from jsonschema import ValidationError, validate

from ads.common import utils
from ads.common.object_storage_details import ObjectStorageDetails
from ads.config import (
    COMPARTMENT_OCID,
    PROJECT_OCID,
    AQUA_SERVICE_MODELS_BUCKET as SERVICE_MODELS_BUCKET,
)
from ads.feature_engineering.schema import Schema
from ads.jobs.builders.base import Builder
from ads.model.artifact_downloader import (
    LargeArtifactDownloader,
    SmallArtifactDownloader,
)
from ads.model.artifact_uploader import LargeArtifactUploader, SmallArtifactUploader
from ads.model.model_metadata import (
    MetadataCustomCategory,
    ModelCustomMetadata,
    ModelCustomMetadataItem,
    ModelProvenanceMetadata,
    ModelTaxonomyMetadata,
)
from ads.model.service.oci_datascience_model import (
    ModelProvenanceNotFoundError,
    OCIDataScienceModel,
)
from ads.common import oci_client as oc

logger = logging.getLogger(__name__)


_MAX_ARTIFACT_SIZE_IN_BYTES = 2147483648  # 2GB
MODEL_BY_REFERENCE_VERSION = "1.0"
MODEL_BY_REFERENCE_JSON_FILE_NAME = "model_description.json"


class ModelArtifactSizeError(Exception):  # pragma: no cover
    def __init__(self, max_artifact_size: str):
        super().__init__(
            f"The model artifacts size is greater than `{max_artifact_size}`. "
            "The `bucket_uri` needs to be specified to "
            "copy artifacts to the object storage bucket. "
            "Example: `bucket_uri=oci://<bucket_name>@<namespace>/prefix/`"
        )


class BucketNotVersionedError(Exception):  # pragma: no cover
    def __init__(
        self,
        msg="Model artifact bucket is not versioned. Enable versioning on the bucket to proceed with model creation by reference.",
    ):
        super().__init__(msg)


class ModelFileDescriptionError(Exception):  # pragma: no cover
    def __init__(self, msg="Model File Description file is not set up."):
        super().__init__(msg)


class InvalidArtifactType(Exception):  # pragma: no cover
    pass


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
    version_id: str
        Model version id
    model_file_description: dict
        Contains object path details for models created by reference.

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
    with_artifact(self, *uri: str)
        Sets the artifact location. Can be a local. For models created by reference, uri can take in single arg or multiple args in case
        of a fine-tuned or multimodel setting.
    with_model_version_set_id(self, model_version_set_id: str):
        Sets the model version set ID.
    with_version_label(self, version_label: str):
        Sets the model version label.
    with_version_id(self, version_id: str):
        Sets the model version id.
    with_model_file_description: dict
        Sets path details for models created by reference. Input can be either a dict, string or json file and
        the schema is dictated by model_file_description_schema.json


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
    CONST_MODEL_VERSION_SET_NAME = "modelVersionSetName"
    CONST_MODEL_VERSION_LABEL = "versionLabel"
    CONST_MODEL_VERSION_ID = "versionId"
    CONST_TIME_CREATED = "timeCreated"
    CONST_LIFECYCLE_STATE = "lifecycleState"
    CONST_MODEL_FILE_DESCRIPTION = "modelDescription"

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
        CONST_MODEL_VERSION_SET_NAME: "model_version_set_name",
        CONST_MODEL_VERSION_LABEL: "version_label",
        CONST_MODEL_VERSION_ID: "version_id",
        CONST_TIME_CREATED: "time_created",
        CONST_LIFECYCLE_STATE: "lifecycle_state",
        CONST_MODEL_FILE_DESCRIPTION: "model_description",
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
        self.local_copy_dir = None

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
    def lifecycle_state(self) -> Union[str, None]:
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
    def time_created(self) -> str:
        return self.get_spec(self.CONST_TIME_CREATED)

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
    def input_schema(self) -> Union[Schema, Dict]:
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
            try:
                schema = Schema.from_dict(schema)
            except Exception as err:
                logger.warn(err)

        return self.set_spec(self.CONST_INPUT_SCHEMA, schema)

    @property
    def output_schema(self) -> Union[Schema, Dict]:
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
            try:
                schema = Schema.from_dict(schema)
            except Exception as err:
                logger.warn(err)

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
    def artifact(self) -> Union[str, list]:
        return self.get_spec(self.CONST_ARTIFACT)

    def with_artifact(self, uri: str, *args):
        """Sets the artifact location. Can be a local.

        Parameters
        ----------
        uri: str
            Path to artifact directory or to the ZIP archive.
            It could contain a serialized model(required) as well as any files needed for deployment.
            The content of the source folder will be zipped and uploaded to the model catalog.
            For models created by reference, uri can take in single arg or multiple args in case of a fine-tuned or
            multimodel setting.
        Examples
        --------
        >>> .with_artifact(uri="./model1/")
        >>> .with_artifact(uri="./model1.zip")
        >>> .with_artifact("./model1", "./model2")
        """

        return self.set_spec(self.CONST_ARTIFACT, [uri] + list(args) if args else uri)

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
    def model_version_set_name(self) -> str:
        return self.get_spec(self.CONST_MODEL_VERSION_SET_NAME)

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

    @property
    def version_id(self) -> str:
        return self.get_spec(self.CONST_MODEL_VERSION_ID)

    def with_version_id(self, version_id: str):
        """Sets the model version id.

        Parameters
        ----------
        version_id: str
            The model version id.
        """
        return self.set_spec(self.CONST_MODEL_VERSION_ID, version_id)

    @property
    def model_file_description(self) -> dict:
        return self.get_spec(self.CONST_MODEL_FILE_DESCRIPTION)

    def with_model_file_description(
        self, json_dict: dict = None, json_string: str = None, json_uri: str = None
    ):
        """Sets the json file description for model passed by reference
        Parameters
        ----------
        json_dict : dict, optional
            json dict, by default None
        json_string : str, optional
            json string, by default None
        json_uri : str, optional
            URI location of file containing json, by default None

        Examples
        --------
        >>> DataScienceModel().with_model_file_description(json_string="<json_string>")
        >>> DataScienceModel().with_model_file_description(json_dict=dict())
        >>> DataScienceModel().with_model_file_description(json_uri="./model_description.json")
        """
        if json_dict:
            json_data = json_dict
        elif json_string:
            json_data = json.loads(json_string)
        elif json_uri:
            with open(json_uri, "r") as json_file:
                json_data = json.load(json_file)
        else:
            raise ValueError("Must provide either a valid json string or URI location.")

        schema_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "model_file_description_schema.json",
        )
        with open(schema_file_path, encoding="utf-8") as schema_file:
            schema = json.load(schema_file)

        try:
            validate(json_data, schema)
        except ValidationError as ve:
            message = (
                f"model_file_description_schema.json validation failed. "
                f"See Exception: {ve}"
            )
            logging.error(message)
            raise ModelFileDescriptionError(message)

        return self.set_spec(self.CONST_MODEL_FILE_DESCRIPTION, json_data)

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

                .. versionadded:: 2.8.10

                    If `artifact` is provided as an object storage path to a zip archive, `bucket_uri` will be ignored.

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
            parallel_process_count: (int, optional).
                The number of worker processes to use in parallel for uploading individual parts of a multipart upload.
            model_by_reference: (bool, optional)
                Whether model artifact is made available to Model Store by reference. Requires artifact location to be
                provided using with_artifact method.

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

        model_by_reference = kwargs.pop("model_by_reference", False)
        if model_by_reference:
            # Update custom metadata
            logger.info("Update custom metadata field with model by reference flag.")
            metadata_item = ModelCustomMetadataItem(
                key=self.CONST_MODEL_FILE_DESCRIPTION,
                value="true",
                description="model by reference flag",
                category=MetadataCustomCategory.OTHER,
            )
            if self.custom_metadata_list:
                self.custom_metadata_list._add(metadata_item, replace=True)
            else:
                custom_metadata = ModelCustomMetadata()
                custom_metadata._add(metadata_item)
                self.with_custom_metadata_list(custom_metadata)

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
            parallel_process_count=kwargs.pop("parallel_process_count", None),
            model_by_reference=model_by_reference,
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
        parallel_process_count: int = utils.DEFAULT_PARALLEL_PROCESS_COUNT,
        model_by_reference: Optional[bool] = False,
    ) -> None:
        """Uploads model artifacts to the model catalog.

        Parameters
        ----------
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for uploading large artifacts which
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.

            .. versionadded:: 2.8.10

                If `artifact` is provided as an object storage path to a zip archive, `bucket_uri` will be ignored.

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
        parallel_process_count: (int, optional)
            The number of worker processes to use in parallel for uploading individual parts of a multipart upload.
        model_by_reference: (bool, optional)
            Whether model artifact is made available to Model Store by reference.
        """
        # Upload artifact to the model catalog
        if model_by_reference and self.model_file_description:
            logger.info(
                "Model artifact will be uploaded using model_file_description contents, "
                "artifact location will not be used."
            )
        elif not self.artifact:
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

        if model_by_reference:
            self._validate_prepare_file_description_artifact()
        else:
            if isinstance(self.artifact, list):
                raise InvalidArtifactType(
                    "Multiple artifacts are only allowed for models created by reference."
                )

            if ObjectStorageDetails.is_oci_path(self.artifact):
                if bucket_uri and bucket_uri != self.artifact:
                    logger.warn(
                        "The `bucket_uri` will be ignored and the value of `self.artifact` will be used instead."
                    )
                bucket_uri = self.artifact

        if not model_by_reference and (
            bucket_uri or utils.folder_size(self.artifact) > _MAX_ARTIFACT_SIZE_IN_BYTES
        ):
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
                parallel_process_count=parallel_process_count,
            )
        else:
            artifact_uploader = SmallArtifactUploader(
                dsc_model=self.dsc_model,
                artifact_path=self.artifact,
            )
        artifact_uploader.upload()

        self._remove_file_description_artifact()

    def _remove_file_description_artifact(self):
        """Removes temporary model file description artifact for model by reference."""
        # delete if local copy directory was created
        if self.local_copy_dir:
            shutil.rmtree(self.local_copy_dir, ignore_errors=True)

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
        try:
            model_by_reference = self.custom_metadata_list.get(
                self.CONST_MODEL_FILE_DESCRIPTION
            ).value
            logging.info(
                f"modelDescription tag found in custom metadata list with value {model_by_reference}"
            )
        except ValueError:
            model_by_reference = False

        if model_by_reference:
            _, artifact_size = self._download_file_description_artifact()
            logging.warning(
                f"Model {self.dsc_model.id} was created by reference, artifacts will be downloaded from the bucket {bucket_uri}"
            )
            # artifacts will be downloaded from model_file_description
            bucket_uri = None
        else:
            artifact_info = self.dsc_model.get_artifact_info()
            artifact_size = int(artifact_info.get("content-length"))

            if not bucket_uri and artifact_size > _MAX_ARTIFACT_SIZE_IN_BYTES:
                raise ModelArtifactSizeError(
                    utils.human_size(_MAX_ARTIFACT_SIZE_IN_BYTES)
                )

        if (
            artifact_size > _MAX_ARTIFACT_SIZE_IN_BYTES
            or bucket_uri
            or model_by_reference
        ):
            artifact_downloader = LargeArtifactDownloader(
                dsc_model=self.dsc_model,
                target_dir=target_dir,
                auth=auth,
                force_overwrite=force_overwrite,
                region=region,
                bucket_uri=bucket_uri,
                overwrite_existing_artifact=overwrite_existing_artifact,
                remove_existing_artifact=remove_existing_artifact,
                model_file_description=self.model_file_description,
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
                if isinstance(value, dict):
                    dsc_spec[dsc_attr] = json.dumps(value)
                else:
                    dsc_spec[dsc_attr] = getattr(
                        self.get_spec(infra_attr),
                        COMPLEX_ATTRIBUTES_CONVERTER[infra_attr],
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
            self.CONST_INPUT_SCHEMA: [Schema.from_json, json.loads],
            self.CONST_OUTPUT_SCHEMA: [Schema.from_json, json.loads],
            self.CONST_CUSTOM_METADATA: ModelCustomMetadata._from_oci_metadata,
            self.CONST_DEFINED_METADATA: ModelTaxonomyMetadata._from_oci_metadata,
        }

        # Update the main properties
        self.dsc_model = dsc_model
        for infra_attr, dsc_attr in self.attribute_map.items():
            value = utils.get_value(dsc_model, dsc_attr)
            if value:
                if infra_attr in COMPLEX_ATTRIBUTES_CONVERTER:
                    converter = COMPLEX_ATTRIBUTES_CONVERTER[infra_attr]
                    if isinstance(converter, List):
                        for converter_item in converter:
                            try:
                                value = converter_item(value)
                            except Exception as err:
                                logger.warn(err)
                                pass
                    else:
                        value = converter(value)
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

            if self.dsc_model.is_model_by_reference():
                _, file_extension = os.path.splitext(file_name_info["filename"])
                if file_extension.lower() == ".json":
                    bucket_uri, _ = self._download_file_description_artifact()
                    self.set_spec(self.CONST_ARTIFACT, bucket_uri)
            else:
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

    def _validate_prepare_file_description_artifact(self):
        """This helper method validates the path to check if the buckets are versioned and if the OSS location and
        the files exist. Next, it creates a json dict with the path information and sets it as the artifact to be
        uploaded."""

        if not self.model_file_description:
            bucket_uri = self.artifact
            if isinstance(bucket_uri, str):
                bucket_uri = [bucket_uri]

            for uri in bucket_uri:
                os_path = ObjectStorageDetails.from_path(uri)
                # for aqua use case, user may not have access to the service bucket.
                if os_path.bucket == SERVICE_MODELS_BUCKET:
                    continue
                if not os_path.is_bucket_versioned():
                    message = f"Model artifact bucket {uri} is not versioned. Enable versioning on the bucket to proceed with model creation by reference."
                    logger.error(message)
                    raise BucketNotVersionedError(message)

            json_data = self._prepare_file_description_artifact(bucket_uri)
            self.with_model_file_description(json_dict=json_data)

        self.local_copy_dir = tempfile.mkdtemp()
        # create temp directory for model description file
        json_file_path = os.path.join(
            self.local_copy_dir, MODEL_BY_REFERENCE_JSON_FILE_NAME
        )
        with open(json_file_path, "w") as outfile:
            json.dump(self.model_file_description, outfile, indent=2)

        self.with_artifact(json_file_path)

    @staticmethod
    def _prepare_file_description_artifact(bucket_uri: list) -> dict:
        """Prepares yaml file config if model is passed by reference and uploaded to catalog.

        Returns
        -------
        dict
            json dict with the model by reference artifact details
        """

        # create json content
        content = dict()
        content["version"] = MODEL_BY_REFERENCE_VERSION
        content["type"] = "modelOSSReferenceDescription"
        content["models"] = []

        for uri in bucket_uri:
            if not ObjectStorageDetails.is_oci_path(uri) or uri.endswith(".zip"):
                msg = "Artifact path cannot be a zip file or local directory for model creation by reference."
                logging.error(msg)
                raise InvalidArtifactType(msg)

            # read list from objects from artifact location
            oss_details = ObjectStorageDetails.from_path(uri)

            # first retrieve the etag and version id
            object_versions = oss_details.list_object_versions(fields="etag")
            version_dict = {
                obj.etag: obj.version_id
                for obj in object_versions
                if obj.etag is not None
            }

            # add version id based on etag for each object
            objects = oss_details.list_objects(fields="name,etag,size").objects

            if len(objects) == 0:
                raise ModelFileDescriptionError(
                    f"The path {oss_details.path} does not exist or no objects were found in the path. "
                )

            object_list = []
            for obj in objects:
                object_list.append(
                    {
                        "name": obj.name,
                        "version": version_dict[obj.etag],
                        "sizeInBytes": obj.size,
                    }
                )
            content["models"].extend(
                [
                    {
                        "namespace": oss_details.namespace,
                        "bucketName": oss_details.bucket,
                        "prefix": oss_details.filepath,
                        "objects": object_list,
                    }
                ]
            )

        return content

    def _download_file_description_artifact(self) -> Tuple[Union[str, List[str]], int]:
        """Loads the json file from model artifact, updates the
        model file description property, and returns the bucket uri and artifact size details.

        Returns
        -------
        bucket_uri: Union[str, List[str]]
            Location(s) of bucket where model artifacts are present
        artifact_size: int
            estimated size of the model files in bytes

        """
        if not self.model_file_description:
            # get model file description from model artifact json
            with tempfile.TemporaryDirectory() as temp_dir:
                artifact_downloader = SmallArtifactDownloader(
                    dsc_model=self.dsc_model,
                    target_dir=temp_dir,
                )
                artifact_downloader.download()
                # create temp directory for model description file
                json_file_path = os.path.join(
                    temp_dir, MODEL_BY_REFERENCE_JSON_FILE_NAME
                )
                self.with_model_file_description(json_uri=json_file_path)

        model_file_desc_dict = self.model_file_description
        models = model_file_desc_dict["models"]

        bucket_uri = list()
        artifact_size = 0
        for model in models:
            namespace = model["namespace"]
            bucket_name = model["bucketName"]
            prefix = model["prefix"]
            objects = model["objects"]
            uri = f"oci://{bucket_name}@{namespace}/{prefix}"
            artifact_size += sum([obj["sizeInBytes"] for obj in objects])
            bucket_uri.append(uri)

        return bucket_uri[0] if len(bucket_uri) == 1 else bucket_uri, artifact_size

    def add_artifact(
        self,
        uri: Optional[str] = None,
        namespace: Optional[str] = None,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        files: Optional[List[str]] = None,
    ):
        """
        Adds information about objects in a specified bucket to the model description JSON.

        Parameters
        ----------
        uri : str, optional
            The URI representing the location of the artifact in OCI object storage.
        namespace : str, optional
            The namespace of the bucket containing the objects. Required if `uri` is not provided.
        bucket : str, optional
            The name of the bucket containing the objects. Required if `uri` is not provided.
        prefix : str, optional
            The prefix of the objects to add. Defaults to None. Cannot be provided if `files` is provided.
        files : list of str, optional
            A list of file names to include in the model description. If provided, only objects with matching file names will be included. Cannot be provided if `prefix` is provided.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            - If both `uri` and (`namespace` and `bucket`) are provided.
            - If neither `uri` nor both `namespace` and `bucket` are provided.
            - If both `prefix` and `files` are provided.
            - If no files are found to add to the model description.

        Note
        ----
        - If `files` is not provided, it retrieves information about all objects in the bucket.
        - If `files` is provided, it only retrieves information about objects with matching file names.
        - If no objects are found to add to the model description, a ValueError is raised.
        """

        if uri and (namespace or bucket):
            raise ValueError(
                "Either 'uri' must be provided or both 'namespace' and 'bucket' must be provided."
            )
        if uri:
            object_storage_details = ObjectStorageDetails.from_path(uri)
            bucket = object_storage_details.bucket
            namespace = object_storage_details.namespace
            prefix = (
                None
                if object_storage_details.filepath == ""
                else object_storage_details.filepath
            )
        if (not namespace) or (not bucket):
            raise ValueError("Both 'namespace' and 'bucket' must be provided.")

        # Check if both prefix and files are provided
        if prefix is not None and files is not None:
            raise ValueError(
                "Both 'prefix' and 'files' cannot be provided. Please provide only one."
            )

        if self.model_file_description == None:
            self.empty_json = {
                "version": "1.0",
                "type": "modelOSSReferenceDescription",
                "models": [],
            }
            self.set_spec(self.CONST_MODEL_FILE_DESCRIPTION, self.empty_json)

        # Get object storage client
        self.object_storage_client = oc.OCIClientFactory(
            **(self.dsc_model.auth)
        ).object_storage

        # Remove if the model already exists
        self.remove_artifact(namespace=namespace, bucket=bucket, prefix=prefix)

        def check_if_file_exists(fileName):
            isExists = False
            try:
                headResponse = self.object_storage_client.head_object(
                    namespace, bucket, object_name=fileName
                )
                if headResponse.status == 200:
                    isExists = True
            except Exception as e:
                if hasattr(e, "status") and e.status == 404:
                    logger.error(f"File not found in bucket: {fileName}")
                else:
                    logger.error(f"An error occured: {e}")
            return isExists

        # Function to un-paginate the api call with while loop
        def list_obj_versions_unpaginated():
            objectStorageList = []
            has_next_page, opc_next_page = True, None
            while has_next_page:
                response = self.object_storage_client.list_object_versions(
                    namespace_name=namespace,
                    bucket_name=bucket,
                    prefix=prefix,
                    fields="name,size",
                    page=opc_next_page,
                )
                objectStorageList.extend(response.data.items)
                has_next_page = response.has_next_page
                opc_next_page = response.next_page
            return objectStorageList

        # Fetch object details and put it into the objects variable
        objectStorageList = []
        if files == None:
            objectStorageList = list_obj_versions_unpaginated()
        else:
            for fileName in files:
                if check_if_file_exists(fileName=fileName):
                    objectStorageList.append(
                        self.object_storage_client.list_object_versions(
                            namespace_name=namespace,
                            bucket_name=bucket,
                            prefix=fileName,
                            fields="name,size",
                        ).data.items[0]
                    )

        objects = [
            {"name": obj.name, "version": obj.version_id, "sizeInBytes": obj.size}
            for obj in objectStorageList
            if obj.size > 0
        ]

        if len(objects) == 0:
            error_message = (
                f"No files to add in the bucket: {bucket} with namespace: {namespace} "
                f"and prefix: {prefix}. File names: {files}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        tmp_model_file_description = self.model_file_description
        tmp_model_file_description["models"].append(
            {
                "namespace": namespace,
                "bucketName": bucket,
                "prefix": "" if not prefix else prefix,
                "objects": objects,
            }
        )
        self.set_spec(self.CONST_MODEL_FILE_DESCRIPTION, tmp_model_file_description)

    def remove_artifact(
        self,
        uri: Optional[str] = None,
        namespace: Optional[str] = None,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
    ):
        """
        Removes information about objects in a specified bucket or using a specified URI from the model description JSON.

        Parameters
        ----------
        uri : str, optional
            The URI representing the location of the artifact in OCI object storage.
        namespace : str, optional
            The namespace of the bucket containing the objects. Required if `uri` is not provided.
        bucket : str, optional
            The name of the bucket containing the objects. Required if `uri` is not provided.
        prefix : str, optional
            The prefix of the objects to remove. Defaults to None.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            - If both 'uri' and ('namespace' and 'bucket') are provided.
            - If neither 'uri' nor both 'namespace' and 'bucket' are provided.
            - If the model description JSON is None.
        """

        if uri and (namespace or bucket):
            raise ValueError(
                "Either 'uri' must be provided or both 'namespace' and 'bucket' must be provided."
            )
        if uri:
            object_storage_details = ObjectStorageDetails.from_path(uri)
            bucket = object_storage_details.bucket
            namespace = object_storage_details.namespace
            prefix = (
                None
                if object_storage_details.filepath == ""
                else object_storage_details.filepath
            )
        if (not namespace) or (not bucket):
            raise ValueError("Both 'namespace' and 'bucket' must be provided.")

        def findModelIdx():
            for idx, model in enumerate(self.model_file_description["models"]):
                if (
                    model["namespace"],
                    model["bucketName"],
                    (model["prefix"] if ("prefix" in model) else None),
                ) == (namespace, bucket, "" if not prefix else prefix):
                    return idx
            return -1

        if self.model_file_description == None:
            return

        modelSearchIdx = findModelIdx()
        if modelSearchIdx == -1:
            return
        else:
            # model found case
            self.model_file_description["models"].pop(modelSearchIdx)
