#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import warnings

warnings.warn(
    (
        "The `ads.catalog.model` is deprecated in `oracle-ads 2.6.9` and will be removed in `oracle-ads 3.0`. "
        "Use framework specific Model utility class for saving and deploying model. "
        "Check https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/quick_start.html"
    ),
    DeprecationWarning,
    stacklevel=2,
)

import json
import os
import shutil
import tempfile
import time
import uuid
from typing import Dict, Optional, Union
from zipfile import ZipFile

import pandas as pd
import yaml
from ads.catalog.summary import SummaryList
from ads.common import auth, logger, oci_client, utils
from ads.common.decorator.deprecate import deprecated
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common.model_artifact import ConflictStrategy, ModelArtifact
from ads.model.model_metadata import (
    METADATA_SIZE_LIMIT,
    MetadataSizeTooLarge,
    ModelCustomMetadata,
    ModelTaxonomyMetadata,
)
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.config import (
    NB_SESSION_COMPARTMENT_OCID,
    OCI_ODSC_SERVICE_ENDPOINT,
    OCI_REGION_METADATA,
    PROJECT_OCID,
)
from ads.dataset.progress import TqdmProgressBar
from ads.feature_engineering.schema import Schema
from ads.model.model_version_set import ModelVersionSet, _extract_model_version_set_id
from ads.model.deployment.model_deployer import ModelDeployer
from oci.data_science.data_science_client import DataScienceClient
from oci.data_science.models import (
    ArtifactExportDetailsObjectStorage,
    ArtifactImportDetailsObjectStorage,
    CreateModelDetails,
    ExportModelArtifactDetails,
    ImportModelArtifactDetails,
)
from oci.data_science.models import Model as OCIModel
from oci.data_science.models import ModelSummary, WorkRequest
from oci.data_science.models.model_provenance import ModelProvenance
from oci.data_science.models.update_model_details import UpdateModelDetails
from oci.exceptions import ServiceError
from oci.identity import IdentityClient

_UPDATE_MODEL_DETAILS_ATTRIBUTES = [
    "display_name",
    "description",
    "freeform_tags",
    "defined_tags",
    "model_version_set_id",
    "version_label",
]
_MODEL_PROVENANCE_ATTRIBUTES = ModelProvenance().swagger_types.keys()
_ETAG_KEY = "ETag"
_MAX_ARTIFACT_SIZE_IN_BYTES = 2147483648  # 2GB
_WORK_REQUEST_INTERVAL_IN_SEC = 3


class ModelWithActiveDeploymentError(Exception):  # pragma: no cover
    pass


class ModelArtifactSizeError(Exception):  # pragma: no cover
    def __init__(self, max_artifact_size: str):
        super().__init__(
            f"The model artifacts size is greater than `{max_artifact_size}`. "
            "The `bucket_uri` needs to be specified to "
            "copy artifacts to the object storage bucket. "
            "Example: `bucket_uri=oci://<bucket_name>@<namespace>/prefix/`"
        )


def _get_etag(response) -> str:
    """Gets etag from the response."""
    if _ETAG_KEY in response.headers:
        return response.headers[_ETAG_KEY].split("--")[0]
    return None


class ModelSummaryList(SummaryList):
    """Model Summary List which represents a list of Model Object.

    Methods
    -------
    sort_by(self, columns, reverse=False)
        Performs a multi-key sort on a particular set of columns and returns the sorted ModelSummaryList.
        Results are listed in a descending order by default.
    filter(self, selection, instance=None)
        Filters the model list according to a lambda filter function, or list comprehension.
    """

    @deprecated(
        "2.6.6",
        details="Use framework specific Model utility class for saving and deploying model. Check https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/quick_start.html",
    )
    def __init__(
        self,
        model_catalog,
        model_list,
        response=None,
        datetime_format=utils.date_format,
    ):
        super(ModelSummaryList, self).__init__(
            model_list, datetime_format=datetime_format
        )
        self.mc = model_catalog
        self.response = response

    def __add__(self, rhs):
        return ModelSummaryList(
            self.mc, list.__add__(self, rhs), datetime_format=self.datetime_format
        )

    def __getitem__(self, item):
        return self.mc.get_model(super(ModelSummaryList, self).__getitem__(item).id)

    def sort_by(self, columns, reverse=False):
        """
        Performs a multi-key sort on a particular set of columns and returns the sorted ModelSummaryList.
        Results are listed in a descending order by default.

        Parameters
        ----------
        columns: List of string
          A list of columns which are provided to sort on
        reverse: Boolean (defaults to false)
          If you'd like to reverse the results (for example, to get ascending instead of descending results)

        Returns
        -------
        ModelSummaryList: A sorted ModelSummaryList
        """
        return ModelSummaryList(
            self.mc,
            self._sort_by(columns, reverse=reverse),
            datetime_format=self.datetime_format,
        )

    def filter(self, selection, instance=None):
        """
        Filters the model list according to a lambda filter function, or list comprehension.

        Parameters
        ----------
        selection: lambda function filtering model instances, or a list-comprehension
            function of list filtering projects
        instance: list, optional
            list to filter, optional, defaults to self

        Returns
        -------
        ModelSummaryList: A filtered ModelSummaryList
        """
        instance = instance if instance is not None else self

        if callable(selection):
            res = list(filter(selection, instance))
            # lambda filtering
            if len(res) == 0:
                print("No models found")
                return
            return ModelSummaryList(self.mc, res, datetime_format=self.datetime_format)
        elif isinstance(selection, list):
            # list comprehension
            if len(selection) == 0:
                print("No models found")
                return
            return ModelSummaryList(
                self.mc, selection, datetime_format=self.datetime_format
            )
        else:
            raise ValueError(
                "Filter selection must be a function or a ProjectSummaryList"
            )


class Model:
    """Class that represents the ADS implementation of model catalog item.
    Converts the metadata and schema from OCI implememtation to ADS implementation.

    Methods
    -------
    to_dataframe
        Converts model to dataframe format.
    show_in_notebook
        Shows model in the notebook in dataframe or YAML representation.
    activate
        Activates model.
    deactivate
        Deactivates model.
    commit
        Commits the changes made to the model.
    rollback
        Rollbacks the changes made to the model.
    load_model
        Loads the model from the model catalog based on model ID.
    """

    _FIELDS_TO_DECORATE = [
        "schema_input",
        "schema_output",
        "metadata_custom",
        "metadata_taxonomy",
    ]

    _NEW_ATTRIBUTES = [
        "input_schema",
        "output_schema",
        "custom_metadata_list",
        "defined_metadata_list",
    ]

    @deprecated(
        "2.6.6",
        details="Use framework specific Model utility class for saving and deploying model. Check https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/quick_start.html",
    )
    def __init__(
        self,
        model: OCIModel,
        model_etag: str,
        provenance_metadata: ModelProvenance,
        provenance_etag: str,
        ds_client: DataScienceClient,
        identity_client: IdentityClient,
    ) -> None:
        """Initializes the Model.

        Parameters
        ----------
        model: OCIModel
            The OCI model object.
        model_etag: str
            The model ETag.
        provenance_metadata: ModelProvenance
            The model provenance metadata.
        provenance_etag: str
            The model provenance metadata ETag.
        ds_client: DataScienceClient
            The Oracle DataScience client.
        identity_client: IdentityClient
            The Orcale Identity Service Client.
        """
        self.ds_client = ds_client
        self.identity_client = identity_client
        self.user_name = ""
        self._etag = model_etag
        self._provenance_metadata_etag = provenance_etag
        self.provenance_metadata = provenance_metadata
        self._extract_oci_model(model)
        self._extract_user_name(model)

    def _extract_oci_model(self, model: OCIModel) -> None:
        """Extracts the model information from OCI model."""
        for key in model.swagger_types.keys():
            if key not in self._NEW_ATTRIBUTES:
                val = getattr(model, key)
                setattr(self, key, val)
        self.schema_input = self._extract_schema("input_schema", model)
        self.schema_output = self._extract_schema("output_schema", model)
        self.metadata_custom = self._extract_metadata_custom(model)
        self.metadata_taxonomy = self._extract_metadata_taxonomy(model)
        self.swagger_types = model.swagger_types
        self.lifecycle_state = model.lifecycle_state

    def _validate_metadata(self):
        self.metadata_custom.validate()
        self.metadata_taxonomy.validate()
        total_size = self.metadata_custom.size() + self.metadata_taxonomy.size()
        if total_size > METADATA_SIZE_LIMIT:
            raise MetadataSizeTooLarge(total_size)
        return True

    def _extract_user_name(self, model: OCIModel) -> None:
        try:
            user = self.identity_client.get_user(model.created_by)
            self.user_name = user.data.name
        except:
            pass

    @staticmethod
    def _extract_schema(key, model):
        """Extracts the input and output schema."""
        schema = Schema()
        if hasattr(model, key):
            try:
                schema = (
                    Schema.from_dict(json.loads(getattr(model, key)))
                    if getattr(model, key)
                    else Schema()
                )
            except Exception as e:
                logger.warning(str(e))
        return schema

    @staticmethod
    def _extract_metadata_taxonomy(model):
        """Extracts the taxonomy metadata."""
        metadata_taxonomy = ModelTaxonomyMetadata()
        if hasattr(model, "defined_metadata_list"):
            try:
                metadata_taxonomy = ModelTaxonomyMetadata._from_oci_metadata(
                    model.defined_metadata_list
                )
            except Exception as e:
                logger.warning(str(e))
        return metadata_taxonomy

    @staticmethod
    def _extract_metadata_custom(model):
        """Extracts the custom metadata."""
        metadata_custom = ModelCustomMetadata()
        if hasattr(model, "custom_metadata_list"):
            try:
                metadata_custom = ModelCustomMetadata._from_oci_metadata(
                    model.custom_metadata_list
                )
            except Exception as e:
                logger.warning(str(e))
        return metadata_custom

    def _to_dict(self):
        """Converts the model attributes to dictionary format."""
        attributes = {}
        for key in _UPDATE_MODEL_DETAILS_ATTRIBUTES:
            if hasattr(self, key):
                attributes[key] = getattr(self, key)

        if self.provenance_metadata is not None:
            attributes.update(
                {
                    key: getattr(self.provenance_metadata, key)
                    for key in _MODEL_PROVENANCE_ATTRIBUTES
                }
            )
        for field in self._FIELDS_TO_DECORATE:
            attributes[field] = getattr(self, field).to_dict()
        return attributes

    def _to_yaml(self):
        """Converts the model attributes to yaml format."""
        attributes = self._to_dict()
        return yaml.safe_dump(attributes)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the model to dataframe format.

        Returns
        -------
        panadas.DataFrame
            Pandas dataframe.
        """
        attributes = self._to_dict()
        df = pd.DataFrame.from_dict(attributes, orient="index", columns=[""]).dropna()
        return df

    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    def show_in_notebook(self, display_format: str = "dataframe") -> None:
        """Shows model in dataframe or yaml format.
        Supported formats: `dataframe` and `yaml`. Defaults to dataframe format.

        Returns
        -------
        None
            Nothing.
        """
        if display_format == "dataframe":
            from IPython.core.display import display

            display(self.to_dataframe())
        elif display_format == "yaml":
            print(self._to_yaml())
        else:
            NotImplementedError(
                "`display_format` is not supported. Choose 'dataframe' or 'yaml'"
            )

    def _repr_html_(self):
        """Shows model in dataframe format."""
        return (
            self.to_dataframe().style.set_properties(**{"margin-left": "0px"}).to_html()
        )

    def __repr__(self):
        """Shows model in dataframe format."""
        return (
            self.to_dataframe().style.set_properties(**{"margin-left": "0px"}).to_html()
        )

    def activate(self) -> None:
        """Activates model.

        Returns
        -------
        None
            Nothing.
        """
        self.lifecycle_state = OCIModel.LIFECYCLE_STATE_ACTIVE

    def deactivate(self) -> None:
        """Deactivates model.

        Returns
        -------
        None
            Nothing.
        """
        self.lifecycle_state = OCIModel.LIFECYCLE_STATE_INACTIVE

    def commit(self, force: bool = True) -> None:
        """Commits model changes.

        Parameters
        ----------
        force: bool
            If True, any remote changes on this model would be lost.

        Returns
        -------
        None
            Nothing.
        """
        self._validate_metadata()
        attributes = {
            key: getattr(self, key) for key in _UPDATE_MODEL_DETAILS_ATTRIBUTES
        }

        if hasattr(self, "metadata_custom"):
            attributes["custom_metadata_list"] = self.metadata_custom._to_oci_metadata()
        if hasattr(self, "metadata_taxonomy"):
            attributes[
                "defined_metadata_list"
            ] = self.metadata_taxonomy._to_oci_metadata()

        update_model_details = UpdateModelDetails(**attributes)
        # freeform_tags=self._model.freeform_tags, defined_tags=self._model.defined_tags)

        # update model
        # https://docs.oracle.com/en-us/iaas/Content/API/Concepts/usingapi.htm#eleven
        # The API supports etags for the purposes of optimistic concurrency control.
        # The GET and POST calls return an etag response header with a value you should store.
        # When you later want to update or delete the resource, set the if-match header to the ETag
        # you received for the resource. The resource will then be updated or deleted
        # only if the ETag you provide matches the current value of that resource's ETag.
        kwargs = {}
        if not force:
            kwargs["if_match"] = self._etag

        self.ds_client.update_model(
            self.id, update_model_details=update_model_details, **kwargs
        )
        # store the lifecycle status, as updating the model will delete info not included in "update_model_details"
        lifecycle_status = self.lifecycle_state
        self.__dict__.update(self._load_model().__dict__)
        self.lifecycle_state = lifecycle_status

        # update model state
        if not force:
            kwargs["if_match"] = self._etag
        if self.lifecycle_state == OCIModel.LIFECYCLE_STATE_ACTIVE:
            self.ds_client.activate_model(self.id, **kwargs)
        elif self.lifecycle_state == OCIModel.LIFECYCLE_STATE_INACTIVE:
            self.ds_client.deactivate_model(self.id, **kwargs)
        self.__dict__.update(self._load_model().__dict__)

        # update model provenance
        if self.provenance_metadata != ModelProvenance():
            if not force:
                kwargs["if_match"] = self._provenance_metadata_etag
            response = self.ds_client.update_model_provenance(
                self.id, self.provenance_metadata, **kwargs
            )
        # get model etag again, as updating model provenance changes it
        self.__dict__.update(self._load_model().__dict__)

    @staticmethod
    def _get_provenance_metadata(ds_client: DataScienceClient, model_id: str):
        """Gets provenance information for specified model."""
        try:
            provenance_response = ds_client.get_model_provenance(model_id)
        except ServiceError as e:
            if e.status == 404:
                try:
                    provenance_response = ds_client.create_model_provenance(
                        model_id, ModelProvenance()
                    )
                except ServiceError as e2:
                    raise e2
            elif e.status == 409:
                print("The model has been deleted.")
                raise e
            else:
                raise e
        return provenance_response

    @classmethod
    def load_model(
        cls,
        ds_client: DataScienceClient,
        identity_client: IdentityClient,
        model_id: str,
    ) -> "Model":
        """Loads the model from the model catalog based on model ID.

        Parameters
        ----------
        ds_client: DataScienceClient
            The Oracle DataScience client.
        identity_client: IdentityClient
            The Orcale Identity Service Client.
        model_id: str
            The model ID.

        Returns
        -------
        Model
            The ADS model catalog item.

        Raises
        ------
        ServiceError: If error occures while getting model from server.
        KeyError: If model not found.
        ValueError: If error occures while getting model provenance mettadata from server.
        """

        try:
            model_response = ds_client.get_model(model_id)
        except ServiceError as e:
            if e.status == 404:
                raise KeyError(e.message) from e
            raise e

        try:
            provenance_response = cls._get_provenance_metadata(ds_client, model_id)
        except Exception as e:
            raise ValueError(
                f"Unable to fetch model provenance metadata for model {model_id}"
            )

        return cls(
            model_response.data,
            _get_etag(model_response),
            provenance_response.data,
            _get_etag(provenance_response),
            ds_client,
            identity_client,
        )

    def _load_model(self):
        """Loads the model from model catalog."""
        return self.load_model(self.ds_client, self.identity_client, self.id)

    def rollback(self) -> None:
        """Rollbacks the changes made to the model.

        Returns
        -------
        None
            Nothing.
        """
        self.__dict__.update(self._load_model().__dict__)


class ModelCatalog:
    """
    Allows to list, load, update, download, upload and delete models from model catalog.

    Methods
    -------
    get_model(self, model_id)
        Loads the model from the model catalog based on model_id.
    list_models(self, project_id=None, include_deleted=False, datetime_format=utils.date_format, **kwargs)
        Lists all models in a given compartment, or in the current project if project_id is specified.
    list_model_deployment(self, model_id, config=None, tenant_id=None, limit=500, page=None, **kwargs)
        Gets the list of model deployments by model Id across the compartments.
    update_model(self, model_id, update_model_details=None, **kwargs)
        Updates a model with given model_id, using the provided update data.
    delete_model(self, model, **kwargs)
        Deletes the model based on model_id.
    download_model(self, model_id, target_dir, force_overwrite=False, install_libs=False, conflict_strategy=ConflictStrategy.IGNORE)
        Downloads the model from model_dir to target_dir based on model_id.
    upload_model(self, model_artifact, provenance_metadata=None, project_id=None, display_name=None, description=None)
        Uploads the model artifact to cloud storage.
    """

    @deprecated(
        "2.6.6",
        details="Use framework specific Model utility class for saving and deploying model. Check https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/quick_start.html",
    )
    def __init__(
        self,
        compartment_id: Optional[str] = None,
        ds_client_auth: Optional[dict] = None,
        identity_client_auth: Optional[dict] = None,
        timeout: Optional[int] = None,
        ds_client: Optional[DataScienceClient] = None,
        identity_client: Optional[IdentityClient] = None,
    ):
        """Initializes model catalog instance.

        Parameters
        ----------
        compartment_id : (str, optional). Defaults to None.
            Model compartment OCID. If `None`, the `config.NB_SESSION_COMPARTMENT_OCID` would be used.
        ds_client_auth : (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate DataScienceClient object.
        identity_client_auth : (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        timeout: (int, optional). Defaults to 10 seconds.
            The connection timeout in seconds for the client.
        ds_client: DataScienceClient
            The Oracle DataScience client.
        identity_client: IdentityClient
            The Orcale Identity Service Client.

        Raises
        ------
        ValueError
            If compartment ID not specified.
        TypeError
            If timeout not an integer.
        """
        self.compartment_id = (
            NB_SESSION_COMPARTMENT_OCID if compartment_id is None else compartment_id
        )
        if self.compartment_id is None:
            raise ValueError("compartment_id needs to be specified.")

        if timeout and not isinstance(timeout, int):
            raise TypeError("Timeout must be an integer.")

        self.ds_client_auth = ds_client_auth
        self.identity_client_auth = identity_client_auth
        self.ds_client = ds_client
        self.identity_client = identity_client

        if not self.ds_client:
            self.ds_client_auth = (
                ds_client_auth
                if ds_client_auth
                else auth.default_signer(
                    {"service_endpoint": OCI_ODSC_SERVICE_ENDPOINT}
                )
            )
            if timeout:
                if not self.ds_client_auth.get("client_kwargs"):
                    self.ds_client_auth["client_kwargs"] = {}
                self.ds_client_auth["client_kwargs"]["timeout"] = timeout
            self.ds_client = oci_client.OCIClientFactory(
                **self.ds_client_auth
            ).data_science

        if not self.identity_client:
            self.identity_client_auth = (
                identity_client_auth
                if identity_client_auth
                else auth.default_signer(
                    {"service_endpoint": OCI_ODSC_SERVICE_ENDPOINT}
                )
            )
            if timeout:
                if not self.identity_client_auth.get("client_kwargs"):
                    self.identity_client_auth["client_kwargs"] = {}
                self.identity_client_auth["client_kwargs"]["timeout"] = timeout
            self.identity_client = oci_client.OCIClientFactory(
                **self.identity_client_auth
            ).identity

        self.short_id_index = {}

    def __getitem__(self, model_id):  # pragma: no cover
        return self.get_model(model_id)

    def __contains__(self, model_id):  # pragma: no cover
        try:
            return self.get_model(model_id) is not None
        except KeyError:
            return False
        except Exception:
            raise

    def __iter__(self):  # pragma: no cover
        return self.list_models().__iter__()

    def __len__(self):  # pragma: no cover
        return len(self.list_models())

    def get_model(self, model_id):
        """
        Loads the model from the model catalog based on model_id.

        Parameters
        ----------
        model_id: str, required
            The model ID.

        Returns
        -------
        ads.catalog.Model
            The ads.catalog.Model with the matching ID.
        """
        if not model_id.startswith("ocid"):
            model_id = self.short_id_index[model_id]
            self.id = model_id
        return Model.load_model(self.ds_client, self.identity_client, model_id)

    def list_models(
        self,
        project_id: str = None,
        include_deleted: bool = False,
        datetime_format: str = utils.date_format,
        **kwargs,
    ):
        """
        Lists all models in a given compartment, or in the current project if project_id is specified.

        Parameters
        ----------
        project_id: str
            The project_id of model.
        include_deleted: bool, optional, default=False
            Whether to include deleted models in the returned list.
        datetime_format: str, optional, default: '%Y-%m-%d %H:%M:%S'
            Change format for date time fields.

        Returns
        -------
        ModelSummaryList
            A list of models.
        """
        try:
            list_models_response = self.ds_client.list_models(
                self.compartment_id, project_id=project_id, **kwargs
            )
            if list_models_response.data is None or len(list_models_response.data) == 0:
                print("No model found.")
                return
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(se.message) from se
            else:
                raise

        model_list_filtered = [
            Model(
                model=model,
                model_etag=None,
                provenance_metadata=None,
                provenance_etag=None,
                ds_client=self.ds_client,
                identity_client=self.identity_client,
            )
            for model in list_models_response.data
            if include_deleted
            or model.lifecycle_state != ModelSummary.LIFECYCLE_STATE_DELETED
        ]
        # handle empty list
        if model_list_filtered is None or len(model_list_filtered) == 0:
            print("No model found.")
            return []

        msl = ModelSummaryList(
            self,
            model_list_filtered,
            list_models_response,
            datetime_format=datetime_format,
        )
        self.short_id_index.update(msl.short_id_index)
        return msl

    def list_model_deployment(
        self,
        model_id: str,
        config: dict = None,
        tenant_id: str = None,
        limit: int = 500,
        page: str = None,
        **kwargs,
    ):
        """
        Gets the list of model deployments by model Id across the compartments.

        Parameters
        ----------
        model_id: str
            The model ID.
        config: dict (optional)
            Configuration keys and values as per SDK and Tool Configuration.
            The from_file() method can be used to load configuration from a file.
            Alternatively, a dict can be passed. You can validate_config the dict
            using validate_config(). Defaults to None.
        tenant_id: str (optional)
            The tenancy ID, which can be used to specify a different tenancy
            (for cross-tenancy authorization) when searching for resources in
            a different tenancy. Defaults to None.
        limit: int (optional)
            The maximum number of items to return. The value must be between
            1 and 1000. Defaults to 500.
        page: str (optional)
            The page at which to start retrieving results.

        Returns
        -------
            The list of model deployments.
        """
        query = f"query datasciencemodeldeployment resources where ModelId='{model_id}'"
        return OCIResource.search(
            query,
            type=SEARCH_TYPE.STRUCTURED,
            config=config,
            tenant_id=tenant_id,
            limit=limit,
            page=page,
            **kwargs,
        )

    def update_model(self, model_id, update_model_details=None, **kwargs) -> Model:
        """
        Updates a model with given model_id, using the provided update data.

        Parameters
        ----------
        model_id: str
            The model ID.
        update_model_details: UpdateModelDetails
            Contains the update model details data to apply.
            Mandatory unless kwargs are supplied.
        kwargs: dict, optional
            Update model details can be supplied instead as kwargs.

        Returns
        -------
        Model
            The ads.catalog.Model with the matching ID.
        """
        if not model_id.startswith("ocid"):
            model_id = self.short_id_index[model_id]
        if update_model_details is None:
            update_model_details = UpdateModelDetails(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in _UPDATE_MODEL_DETAILS_ATTRIBUTES
                }
            )
            update_model_details.compartment_id = self.compartment_id
            # filter kwargs removing used keys
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in _UPDATE_MODEL_DETAILS_ATTRIBUTES
            }
        update_model_response = self.ds_client.update_model(
            model_id, update_model_details, **kwargs
        )
        provenance_response = Model._get_provenance_metadata(self.ds_client, model_id)
        return Model(
            model=update_model_response.data,
            model_etag=_get_etag(update_model_response),
            provenance_metadata=provenance_response.data,
            provenance_etag=_get_etag(provenance_response),
            ds_client=self.ds_client,
            identity_client=self.identity_client,
        )

    def delete_model(self, model: Union[str, "ads.catalog.Model"], **kwargs) -> bool:
        """
        Deletes the model from Model Catalog.

        Parameters
        ----------
        model: Union[str, "ads.catalog.Model"]
            The OCID of the model to delete as a string, or a `ads.catalog.Model` instance.

        kwargs:
            delete_associated_model_deployment: (bool, optional). Defaults to `False`.
                Whether associated model deployments need to be deletet or not.

        Returns
        -------
        bool
            `True` if the model was successfully deleted.

        Raises
        ------
        ModelWithActiveDeploymentError
            If model has active model deployments ant inout attribute
            `delete_associated_model_deployment` set to `False`.
        """
        model_id = (
            model.id
            if isinstance(model, Model)
            else self.short_id_index[model]
            if not model.startswith("ocid")
            else model
        )
        delete_associated_model_deployment = kwargs.pop(
            "delete_associated_model_deployment", None
        )
        active_deployments = tuple(
            item
            for item in self.list_model_deployment(model_id)
            if item.lifecycle_state == "ACTIVE"
        )

        if len(active_deployments) > 0:
            if not delete_associated_model_deployment:
                raise ModelWithActiveDeploymentError(
                    f"The model `{model_id}` has active model deployments: "
                    f"{[item.identifier for item in active_deployments]}. "
                    "Delete associated model deployments before deleting the model or "
                    "set the `delete_associated_model_deployment` attribute to `True`."
                )

            logger.info(
                f"Deleting model deployments associated with the model `{model_id}`."
            )
            for oci_model_deployment in active_deployments:
                (
                    ModelDeployer(config=self.ds_client_auth)
                    .get_model_deployment(oci_model_deployment.identifier)
                    .delete(wait_for_completion=True)
                )

        logger.info(f"Deleting model `{model_id}`.")
        self.ds_client.delete_model(model_id, **kwargs)
        return True

    def _download_artifact(
        self,
        model_id: str,
        target_dir: str,
        force_overwrite: Optional[bool] = False,
        bucket_uri: Optional[str] = None,
        remove_existing_artifact: Optional[bool] = True,
    ) -> None:
        """
        Downloads the model artifacts from model catalog to target_dir based on `model_id`.

        Parameters
        ----------
        model_id: str
            The OCID of the model to download.
        target_dir: str
            The target location of model artifacts.
        force_overwrite: (bool, optional). Defaults to `False`.
            Overwrite target directory if exists.
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for downloading large artifacts with
            size is greater than 2GB. Example: `bucket_uri=oci://<bucket_name>@<namespace>/prefix/`.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Whether artifacts uploaded to object storage bucket need to be removed or not.

        Raises
        ------
        ValueError
            If targeted directory does not exist.
        KeyError
            If model id not found.

        Returns
        -------
        None
            Nothing
        """
        if os.path.exists(target_dir) and os.listdir(target_dir):
            if not force_overwrite:
                raise ValueError(
                    f"The `{target_dir}` directory already exists. "
                    "Set `force_overwrite` to `True` if you wish to overwrite."
                )
            shutil.rmtree(target_dir)

        with utils.get_progress_bar(6) as progress:
            progress.update("Getting information about model artifacts")

            # If the size of artifacts greater than 2GB, then artifacts
            # need to be imported to OS bucket at first.
            try:
                model_artifact_info = self.ds_client.head_model_artifact(
                    model_id=model_id
                ).headers
            except ServiceError as ex:
                if ex.status == 404:
                    raise KeyError(f"The model `{model_id}` not found.") from ex
                raise

            artifact_size = int(model_artifact_info.get("content-length"))

            if bucket_uri or artifact_size > _MAX_ARTIFACT_SIZE_IN_BYTES:
                if not bucket_uri:
                    raise ModelArtifactSizeError(
                        utils.human_size(_MAX_ARTIFACT_SIZE_IN_BYTES)
                    )

                self._download_large_artifact(
                    model_id=model_id,
                    target_dir=target_dir,
                    bucket_uri=os.path.join(bucket_uri, f"{model_id}.zip"),
                    progress=progress,
                    remove_existing_artifact=remove_existing_artifact,
                )
            else:
                self._download_small_artifact(
                    model_id=model_id, target_dir=target_dir, progress=progress
                )
                progress.update()

            progress.update("Done")

    def _download_large_artifact(
        self,
        model_id: str,
        target_dir: str,
        bucket_uri: str,
        progress: TqdmProgressBar,
        remove_existing_artifact: Optional[bool] = True,
    ) -> None:
        """
        Downloads the model artifacts from model catalog to target_dir based on `model_id`.
        This method is used for artifacts with size greater than 2GB.

        Parameters
        ----------
        model_id: str
            The OCID of the model to download.
        target_dir: str
            The target location of model artifacts.
        bucket_uri: str
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for downloading large artifacts with
            size is greater than 2GB. Example: `bucket_uri=oci://<bucket_name>@<namespace>/prefix/`.
        progress: TqdmProgressBar
            The progress bar.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Whether artifacts uploaded to object storage bucket need to be removed or not.

        Returns
        -------
        None
            Nothing.
        """
        progress.update(f"Importing model artifacts from model catalog")
        self._import_model_artifact(model_id=model_id, bucket_uri=bucket_uri)

        progress.update("Copying model artifacts to the artifact directory")
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_file_path = os.path.join(temp_dir, f"{str(uuid.uuid4())}.zip")
            zip_file_path = utils.copy_file(
                uri_src=bucket_uri,
                uri_dst=zip_file_path,
                progressbar_description="Copying model artifacts to the artifact directory",
            )

            progress.update("Extracting model artifacts")
            with ZipFile(zip_file_path) as zip_file:
                zip_file.extractall(target_dir)

        if remove_existing_artifact:
            progress.update("Removing temporary artifacts")
            utils.remove_file(bucket_uri, self.ds_client_auth)
        else:
            progress.update()

    def _import_model_artifact(
        self,
        model_id: str,
        bucket_uri: str,
    ):
        """Imports model artifact from the model catalog to the object storage bucket.
        This method is used for the case when the artifact size is greater than 2GB.
        """
        bucket_details = ObjectStorageDetails.from_path(bucket_uri)
        response = self.ds_client.import_model_artifact(
            model_id=model_id,
            import_model_artifact_details=ImportModelArtifactDetails(
                artifact_import_details=ArtifactImportDetailsObjectStorage(
                    namespace=bucket_details.namespace,
                    destination_bucket=bucket_details.bucket,
                    destination_object_name=bucket_details.filepath,
                    destination_region=self._region,
                )
            ),
        )

        self._wait_for_work_request(
            response=response,
            first_step_description="Preparing to import model artifacts from the model catalog",
            num_steps=3,
        )

    def _download_small_artifact(
        self,
        model_id: str,
        target_dir: str,
        progress: TqdmProgressBar,
    ) -> None:
        """
        Downloads the model artifacts from model catalog to target_dir based on `model_id`.
        This method is used for the artifacts with size less than 2GB.

        Parameters
        ----------
        model_id: str
            The OCID of the model to download.
        target_dir: str
            The target location of model artifacts.
        progress: TqdmProgressBar
            The progress bar.

        Returns
        -------
        None
            Nothing
        """
        progress.update("Importing model artifacts from catalog")
        try:
            zip_contents = self.ds_client.get_model_artifact_content(
                model_id
            ).data.content
        except ServiceError as ex:
            if ex.status == 404:
                raise KeyError(ex.message) from ex
            raise

        with tempfile.TemporaryDirectory() as temp_dir:
            progress.update("Copying model artifacts to the artifact directory")
            zip_file_path = os.path.join(temp_dir, f"{str(uuid.uuid4())}.zip")
            with open(zip_file_path, "wb") as zip_file:
                zip_file.write(zip_contents)
            progress.update("Extracting model artifacts")
            with ZipFile(zip_file_path) as zip_file:
                zip_file.extractall(target_dir)

    @deprecated(
        "2.5.9",
        details="Instead use `ads.common.model_artifact.ModelArtifact.from_model_catalog()`.",
    )
    def download_model(
        self,
        model_id: str,
        target_dir: str,
        force_overwrite: bool = False,
        install_libs: bool = False,
        conflict_strategy=ConflictStrategy.IGNORE,
        bucket_uri: Optional[str] = None,
        remove_existing_artifact: Optional[bool] = True,
    ):
        """
        Downloads the model from model_dir to target_dir based on model_id.

        Parameters
        ----------
        model_id: str
            The OCID of the model to download.
        target_dir: str
            The target location of model after download.
        force_overwrite: bool
            Overwrite target_dir if exists.
        install_libs: bool, default: False
            Install the libraries specified in ds-requirements.txt which are missing in the current environment.
        conflict_strategy: ConflictStrategy, default: IGNORE
           Determines how to handle version conflicts between the current environment and requirements of
           model artifact.
           Valid values: "IGNORE", "UPDATE" or ConflictStrategy.
           IGNORE: Use the installed version in  case of conflict
           UPDATE: Force update dependency to the version required by model artifact in case of conflict
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for downloading large artifacts with
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Whether artifacts uploaded to object storage bucket need to be removed or not.

        Returns
        -------
        ModelArtifact
            A ModelArtifact instance.
        """
        self._download_artifact(
            model_id,
            target_dir,
            force_overwrite,
            bucket_uri=bucket_uri,
            remove_existing_artifact=remove_existing_artifact,
        )

        result = ModelArtifact(
            target_dir,
            conflict_strategy=conflict_strategy,
            install_libs=install_libs,
            reload=False,
        )

        try:
            model_response = self.ds_client.get_model(model_id)
        except ServiceError as e:
            if e.status == 404:
                raise KeyError(e.message) from e
            raise e

        if hasattr(model_response.data, "custom_metadata_list"):
            try:
                result.metadata_custom = ModelCustomMetadata._from_oci_metadata(
                    model_response.data.custom_metadata_list
                )
            except:
                result.metadata_custom = ModelCustomMetadata()
        if hasattr(model_response.data, "defined_metadata_list"):
            try:
                result.metadata_taxonomy = ModelTaxonomyMetadata._from_oci_metadata(
                    model_response.data.defined_metadata_list
                )
            except:
                result.metadata_taxonomy = ModelTaxonomyMetadata()
        if hasattr(model_response.data, "input_schema"):
            try:
                result.schema_input = Schema.from_dict(
                    json.loads(model_response.data.input_schema)
                    if model_response.data.input_schema != ""
                    else Schema()
                )
            except:
                result.schema_input = Schema()
        if hasattr(model_response.data, "output_schema"):
            try:
                result.schema_output = Schema.from_dict(
                    json.loads(model_response.data.output_schema)
                    if model_response.data.output_schema != ""
                    else Schema()
                )
            except:
                result.schema_output = Schema()

        if not install_libs:
            logger.warning(
                "Libraries in `ds-requirements.txt` were not installed. "
                "Use `install_requirements()` to install the required dependencies."
            )
        return result

    @deprecated(
        "2.6.6",
        details="Use framework specific Model utility class for saving and deploying model. Check https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/quick_start.html",
    )
    def upload_model(
        self,
        model_artifact: ModelArtifact,
        provenance_metadata: Optional[ModelProvenance] = None,
        project_id: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        freeform_tags: Optional[Dict[str, Dict[str, object]]] = None,
        defined_tags: Optional[Dict[str, Dict[str, object]]] = None,
        bucket_uri: Optional[str] = None,
        remove_existing_artifact: Optional[bool] = True,
        overwrite_existing_artifact: Optional[bool] = True,
        model_version_set: Optional[Union[str, ModelVersionSet]] = None,
        version_label: Optional[str] = None,
    ):
        """
        Uploads the model artifact to cloud storage.

        Parameters
        ----------
        model_artifact: Union[ModelArtifact, GenericModel]
            The model artifacts or generic model instance.
        provenance_metadata: (ModelProvenance, optional). Defaults to None.
            Model provenance gives data scientists information about the origin of their model.
            This information allows data scientists to reproduce
            the development environment in which the model was trained.
        project_id: (str, optional). Defaults to None.
            The project_id of model.
        display_name: (str, optional). Defaults to None.
            The name of model. If a display_name is not provided, a randomly generated easy to remember name
            with timestamp will be generated, like 'strange-spider-2022-08-17-23:55.02'.
        description: (str, optional). Defaults to None.
            The description of model.
        freeform_tags: (Dict[str, str], optional). Defaults to None.
            Freeform tags for the model, by default None
        defined_tags: (Dict[str, dict[str, object]], optional). Defaults to None.
            Defined tags for the model, by default None.
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for uploading large artifacts which
            size greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Whether artifacts uploaded to object storage bucket need to be removed or not.
        overwrite_existing_artifact: (bool, optional). Defaults to `True`.
            Overwrite target bucket artifact if exists.
        model_version_set: (Union[str, ModelVersionSet], optional). Defaults to None.
            The Model version set OCID, or name, or `ModelVersionSet` instance.
        version_label: (str, optional). Defaults to None.
            The model version label.

        Returns
        -------
        ads.catalog.Model
            The ads.catalog.Model with the matching ID.
        """
        project_id = project_id or PROJECT_OCID
        if not project_id:
            raise ValueError("`project_id` needs to be specified.")

        copy_artifact_to_os = False
        if (
            bucket_uri
            or utils.folder_size(model_artifact.artifact_dir)
            > _MAX_ARTIFACT_SIZE_IN_BYTES
        ):
            if not bucket_uri:
                raise ValueError(
                    f"The model artifacts size is greater than `{utils.human_size(_MAX_ARTIFACT_SIZE_IN_BYTES)}`. "
                    "The `bucket_uri` needs to be specified to "
                    "copy artifacts to the object storage bucket. "
                    "Example: `bucket_uri=oci://<bucket_name>@<namespace>/prefix/`"
                )
            copy_artifact_to_os = True

        # extract model_version_set_id from model_version_set attribute or environment
        # variables in case of saving model in context of model version set.
        model_version_set_id = _extract_model_version_set_id(model_version_set)
        # Set default display_name if not specified - randomly generated easy to remember name generated
        display_name = display_name or utils.get_random_name_for_resource()

        with utils.get_progress_bar(5) as progress:
            project_id = PROJECT_OCID if project_id is None else project_id
            if project_id is None:
                raise ValueError("project_id needs to be specified.")
            schema_file = os.path.join(model_artifact.artifact_dir, "schema.json")
            if os.path.exists(schema_file):
                with open(schema_file, "r") as schema:
                    metadata = json.load(schema)
                    freeform_tags = {"problem_type": metadata["problem_type"]}

            progress.update("Saving model in the model catalog")
            create_model_details = CreateModelDetails(
                display_name=display_name,
                description=description,
                project_id=project_id,
                compartment_id=self.compartment_id,
                custom_metadata_list=model_artifact.metadata_custom._to_oci_metadata()
                if model_artifact.metadata_custom is not None
                else [],
                defined_metadata_list=model_artifact.metadata_taxonomy._to_oci_metadata()
                if model_artifact.metadata_taxonomy is not None
                else [],
                input_schema=model_artifact.schema_input.to_json()
                if model_artifact.schema_input is not None
                else '{"schema": []}',
                output_schema=model_artifact.schema_output.to_json()
                if model_artifact.schema_output is not None
                else '{"schema": []}',
                freeform_tags=freeform_tags,
                defined_tags=defined_tags,
                model_version_set_id=model_version_set_id,
                version_label=version_label,
            )
            model = self.ds_client.create_model(create_model_details)

            if provenance_metadata is not None:
                progress.update("Saving provenance metadata")
                self.ds_client.create_model_provenance(
                    model.data.id, provenance_metadata
                )
            else:
                progress.update()

            # if the model artifact size greater than 2GB then export function
            # needs to be used instead of upload. The export function will copy
            # model artifacts to the OS bucket at first and then will upload
            # the artifacts to the model catalog.
            if copy_artifact_to_os:
                self._export_model_artifact(
                    model_id=model.data.id,
                    model_artifact=model_artifact,
                    progress=progress,
                    bucket_uri=bucket_uri,
                    remove_existing_artifact=remove_existing_artifact,
                    overwrite_existing_artifact=overwrite_existing_artifact,
                )
            else:
                self._upload_model_artifact(
                    model_id=model.data.id,
                    model_artifact=model_artifact,
                    progress=progress,
                )
                progress.update()

            progress.update("Done")
        return self.get_model(model.data.id)

    def _prepare_model_artifact(self, model_artifact, progress: TqdmProgressBar) -> str:
        """Prepares model artifacts to save in the Model Catalog.

        Returns
        -------
        str
            The path to the model artifact zip archive.
        """
        progress.update("Preparing model artifacts zip")
        files_to_upload = model_artifact._get_files()
        artifact_path = "/tmp/saved_model_" + str(uuid.uuid4()) + ".zip"
        zf = ZipFile(artifact_path, "w")
        for matched_file in files_to_upload:
            zf.write(
                os.path.join(model_artifact.artifact_dir, matched_file),
                arcname=matched_file,
            )
        zf.close()
        return artifact_path

    def _upload_model_artifact(self, model_id, model_artifact, progress):
        """Uploads model artifact to the model catalog.
        This method can be used only if the size of model artifact is less than 2GB.
        For the artifacts with size greater than 2 GB the `_export_model_artifact`
        method should be used instead.
        """
        artifact_zip_path = self._prepare_model_artifact(model_artifact, progress)
        progress.update("Uploading model artifacts to the catalog")
        with open(artifact_zip_path, "rb") as file_data:
            bytes_content = file_data.read()
            self.ds_client.create_model_artifact(
                model_id,
                bytes_content,
                content_disposition=f'attachment; filename="{model_id}.zip"',
            )
        os.remove(artifact_zip_path)
        progress.update()

    def _export_model_artifact(
        self,
        model_id: str,
        model_artifact: ModelArtifact,
        bucket_uri: str,
        progress,
        remove_existing_artifact: Optional[bool] = True,
        overwrite_existing_artifact: Optional[bool] = True,
    ):
        """Exports model artifact to the model catalog.
        This method is used for the case when the artifact size is greater than 2GB.
        1. Archive model artifact.
        2. Copies the artifact to the object storage bucket.
        3. Exports artifact from the user's object storage bucket to the system one.
        """
        artifact_zip_path = self._prepare_model_artifact(model_artifact, progress)
        progress.update(f"Copying model artifact to the Object Storage bucket")

        try:
            bucket_uri_file_name = os.path.basename(bucket_uri)
            if not bucket_uri_file_name:
                bucket_uri = os.path.join(bucket_uri, f"{model_id}.zip")
            elif not bucket_uri.lower().endswith(".zip"):
                bucket_uri = f"{bucket_uri}.zip"

            bucket_file_name = utils.copy_file(
                artifact_zip_path,
                bucket_uri,
                force_overwrite=overwrite_existing_artifact,
                auth=self.ds_client_auth,
                progressbar_description="Copying model artifact to the Object Storage bucket",
            )
        except FileExistsError:
            raise FileExistsError(
                f"The `{bucket_uri}` exists. Please use a new file name or "
                "set `overwrite_existing_artifact` to `True` if you wish to overwrite."
            )

        os.remove(artifact_zip_path)

        progress.update("Exporting model artifact to the model catalog")
        bucket_details = ObjectStorageDetails.from_path(bucket_file_name)
        response = self.ds_client.export_model_artifact(
            model_id=model_id,
            export_model_artifact_details=ExportModelArtifactDetails(
                artifact_export_details=ArtifactExportDetailsObjectStorage(
                    namespace=bucket_details.namespace,
                    source_bucket=bucket_details.bucket,
                    source_object_name=bucket_details.filepath,
                    source_region=self._region,
                )
            ),
        )

        self._wait_for_work_request(
            response=response,
            first_step_description="Preparing to export model artifact to the model catalog",
            num_steps=4,
        )

        if remove_existing_artifact:
            progress.update(
                "Removing temporary model artifact from the Object Storage bucket"
            )
            utils.remove_file(bucket_file_name, self.ds_client_auth)
        else:
            progress.update()

    @property
    def _region(self):
        """Gets current region."""
        if "region" in self.ds_client_auth.get("config", {}):
            return self.ds_client_auth["config"]["region"]
        return json.loads(OCI_REGION_METADATA)["regionIdentifier"]

    def _wait_for_work_request(
        self, response, first_step_description: str = "", num_steps=4
    ):
        """Waits for the work request to be completed."""
        STOP_STATE = (
            WorkRequest.STATUS_SUCCEEDED,
            WorkRequest.STATUS_CANCELED,
            WorkRequest.STATUS_CANCELING,
            WorkRequest.STATUS_FAILED,
        )
        work_request_id = response.headers["opc-work-request-id"]
        work_request_logs = None

        i = 0
        with utils.get_progress_bar(num_steps) as progress:
            progress.update(first_step_description)
            while not work_request_logs or len(work_request_logs) < num_steps:
                time.sleep(_WORK_REQUEST_INTERVAL_IN_SEC)
                work_request = self.ds_client.get_work_request(work_request_id)
                work_request_logs = self.ds_client.list_work_request_logs(
                    work_request_id
                ).data
                if work_request_logs:
                    new_work_request_logs = work_request_logs[i:]

                    for wr_item in new_work_request_logs:
                        progress.update(wr_item.message)
                        i += 1

                if work_request.data.status in STOP_STATE:
                    if work_request.data.status != WorkRequest.STATUS_SUCCEEDED:
                        if work_request_logs:
                            raise Exception(work_request_logs[-1].message)
                        else:
                            raise Exception(
                                "An error occurred in attempt to perform the operation. Check the service logs to get more details."
                            )
                    else:
                        break
        return work_request
