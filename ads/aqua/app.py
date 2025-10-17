#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from datetime import datetime, timedelta
from itertools import chain
from typing import Any, Dict, List, Optional, Union

import oci
from cachetools import TTLCache, cached
from oci.data_science.models import (
    ContainerSummary,
    UpdateModelDetails,
    UpdateModelProvenanceDetails,
)

from ads import set_auth
from ads.aqua import logger
from ads.aqua.common.entities import ModelConfigResult
from ads.aqua.common.enums import ConfigFolder, Tags
from ads.aqua.common.errors import AquaValueError
from ads.aqua.common.utils import (
    _is_valid_mvs,
    get_artifact_path,
    is_valid_ocid,
    load_config,
)
from ads.aqua.config.container_config import (
    AquaContainerConfig,
    AquaContainerConfigItem,
)
from ads.aqua.constants import SERVICE_MANAGED_CONTAINER_URI_SCHEME
from ads.common import oci_client as oc
from ads.common.auth import default_signer
from ads.common.utils import UNKNOWN, extract_region, is_path_exists
from ads.config import (
    AQUA_TELEMETRY_BUCKET,
    AQUA_TELEMETRY_BUCKET_NS,
    OCI_ODSC_SERVICE_ENDPOINT,
    OCI_RESOURCE_PRINCIPAL_VERSION,
)
from ads.model.datascience_model import DataScienceModel
from ads.model.deployment.model_deployment import ModelDeployment
from ads.model.model_metadata import (
    ModelCustomMetadata,
    ModelProvenanceMetadata,
    ModelTaxonomyMetadata,
)
from ads.model.model_version_set import ModelVersionSet
from ads.telemetry import telemetry
from ads.telemetry.client import TelemetryClient


class AquaApp:
    """Base Aqua App to contain common components."""

    MAX_WORKERS = 10  # Number of workers for asynchronous resource loading

    @telemetry(name="aqua")
    def __init__(self) -> None:
        if OCI_RESOURCE_PRINCIPAL_VERSION:
            set_auth("resource_principal")
        self._auth = default_signer({"service_endpoint": OCI_ODSC_SERVICE_ENDPOINT})
        self.ds_client = oc.OCIClientFactory(**self._auth).data_science
        self.compute_client = oc.OCIClientFactory(**default_signer()).compute
        self.logging_client = oc.OCIClientFactory(**default_signer()).logging_management
        self.identity_client = oc.OCIClientFactory(**default_signer()).identity
        self.region = extract_region(self._auth)
        self._telemetry = None

    def list_resource(
        self,
        list_func_ref,
        **kwargs,
    ) -> list:
        """Generic method to list OCI Data Science resources.

        Parameters
        ----------
        list_func_ref : function
            A reference to the list operation which will be called.
        **kwargs :
            Additional keyword arguments to filter the resource.
            The kwargs are passed into OCI API.

        Returns
        -------
        list
            A list of OCI Data Science resources.
        """
        return oci.pagination.list_call_get_all_results(
            list_func_ref,
            **kwargs,
        ).data

    def update_model(self, model_id: str, update_model_details: UpdateModelDetails):
        """Updates model details.

        Parameters
        ----------
        model_id : str
            The id of target model.
        update_model_details: UpdateModelDetails
            The model details to be updated.
        """
        self.ds_client.update_model(
            model_id=model_id, update_model_details=update_model_details
        )

    def update_model_provenance(
        self,
        model_id: str,
        update_model_provenance_details: UpdateModelProvenanceDetails,
    ):
        """Updates model provenance details.

        Parameters
        ----------
        model_id : str
            The id of target model.
        update_model_provenance_details: UpdateModelProvenanceDetails
            The model provenance details to be updated.
        """
        self.ds_client.update_model_provenance(
            model_id=model_id,
            update_model_provenance_details=update_model_provenance_details,
        )

    @staticmethod
    def get_source(source_id: str) -> Union[ModelDeployment, DataScienceModel]:
        """
        Fetches a model or model deployment based on the provided OCID.

        Parameters
        ----------
        source_id : str
            OCID of the Data Science model or model deployment.

        Returns
        -------
        Union[ModelDeployment, DataScienceModel]
            The corresponding resource object.

        Raises
        ------
        AquaValueError
            If the OCID is invalid or unsupported.
        """
        logger.debug(f"Resolving source for ID: {source_id}")
        if not is_valid_ocid(source_id):
            logger.error(f"Invalid OCID format: {source_id}")
            raise AquaValueError(
                f"Invalid source ID: {source_id}. Please provide a valid model or model deployment OCID."
            )

        if "datasciencemodeldeployment" in source_id:
            logger.debug(f"Identified as ModelDeployment OCID: {source_id}")
            return ModelDeployment.from_id(source_id)

        if "datasciencemodel" in source_id:
            logger.debug(f"Identified as DataScienceModel OCID: {source_id}")
            return DataScienceModel.from_id(source_id)

        logger.error(f"Unrecognized OCID type: {source_id}")
        raise AquaValueError(
            f"Unsupported source ID type: {source_id}. Must be a model or model deployment OCID."
        )

    def get_multi_source(
        self,
        ids: List[str],
    ) -> Dict[str, Union[ModelDeployment, DataScienceModel]]:
        """
        Retrieves multiple DataScience resources concurrently.

        Parameters
        ----------
        ids : List[str]
            A list of DataScience OCIDs.

        Returns
        -------
        Dict[str, Union[ModelDeployment, DataScienceModel]]
            A mapping from OCID to the corresponding resolved resource object.
        """
        logger.debug(f"Fetching {ids} sources in parallel.")
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            results = list(executor.map(self.get_source, ids))

        return dict(zip(ids, results))

    # TODO: refactor model evaluation implementation to use it.
    @staticmethod
    def create_model_version_set(
        model_version_set_id: str = None,
        model_version_set_name: str = None,
        description: str = None,
        compartment_id: str = None,
        project_id: str = None,
        freeform_tags: dict = None,
        defined_tags: dict = None,
        **kwargs,
    ) -> tuple:
        """Creates ModelVersionSet from given ID or Name.

        Parameters
        ----------
        model_version_set_id: (str, optional):
            ModelVersionSet OCID.
        model_version_set_name: (str, optional):
            ModelVersionSet Name.
        description: (str, optional):
            TBD
        compartment_id: (str, optional):
            Compartment OCID.
        project_id: (str, optional):
            Project OCID.
        tag: (str, optional)
            calling tag, can be Tags.AQUA_FINE_TUNING or Tags.AQUA_EVALUATION
        freeform_tags: (dict, optional)
            Freeform tags for the model version set
        defined_tags: (dict, optional)
            Defined tags for the model version set
        Returns
        -------
        tuple: (model_version_set_id, model_version_set_name)
        """
        # TODO: tag should be selected based on which operation (eval/FT) invoke this method
        #   currently only used by fine-tuning flow.
        tag = Tags.AQUA_FINE_TUNING

        if not model_version_set_id:
            try:
                model_version_set = ModelVersionSet.from_name(
                    name=model_version_set_name,
                    compartment_id=compartment_id,
                )

                if not _is_valid_mvs(model_version_set, tag):
                    raise AquaValueError(
                        f"Invalid model version set name. Please provide a model version set with `{tag}` in tags."
                    )

            except Exception:
                logger.debug(
                    f"Model version set {model_version_set_name} doesn't exist. "
                    "Creating new model version set."
                )
                mvs_freeform_tags = {
                    tag: tag,
                }
                mvs_freeform_tags = {**mvs_freeform_tags, **(freeform_tags or {})}
                model_version_set = (
                    ModelVersionSet()
                    .with_compartment_id(compartment_id)
                    .with_project_id(project_id)
                    .with_name(model_version_set_name)
                    .with_description(description)
                    .with_freeform_tags(**mvs_freeform_tags)
                    .with_defined_tags(**(defined_tags or {}))
                    # TODO: decide what parameters will be needed
                    # when refactor eval to use this method, we need to pass tag here.
                    .create(**kwargs)
                )
                logger.debug(
                    f"Successfully created model version set {model_version_set_name} with id {model_version_set.id}."
                )
            return (model_version_set.id, model_version_set_name)
        else:
            model_version_set = ModelVersionSet.from_id(model_version_set_id)
            # TODO: tag should be selected based on which operation (eval/FT) invoke this method
            if not _is_valid_mvs(model_version_set, tag):
                raise AquaValueError(
                    f"Invalid model version set id. Please provide a model version set with `{tag}` in tags."
                )
            return (model_version_set_id, model_version_set.name)

    # TODO: refactor model evaluation implementation to use it.
    @staticmethod
    def create_model_catalog(
        display_name: str,
        description: str,
        model_version_set_id: str,
        model_custom_metadata: Union[ModelCustomMetadata, Dict],
        model_taxonomy_metadata: Union[ModelTaxonomyMetadata, Dict],
        compartment_id: str,
        project_id: str,
        freeform_tags: Dict = None,
        defined_tags: Dict = None,
        **kwargs,
    ) -> DataScienceModel:
        model = (
            DataScienceModel()
            .with_compartment_id(compartment_id)
            .with_project_id(project_id)
            .with_display_name(display_name)
            .with_description(description)
            .with_model_version_set_id(model_version_set_id)
            .with_custom_metadata_list(model_custom_metadata)
            .with_defined_metadata_list(model_taxonomy_metadata)
            .with_provenance_metadata(ModelProvenanceMetadata(training_id=UNKNOWN))
            .with_freeform_tags(**(freeform_tags or {}))
            .with_defined_tags(
                **(defined_tags or {})
            )  # Create defined tags when a model is created.
            .create(
                **kwargs,
            )
        )
        return model

    def if_artifact_exist(self, model_id: str, **kwargs) -> bool:
        """Checks if the artifact exists.

        Parameters
        ----------
        model_id : str
            The model OCID.
        **kwargs :
            Additional keyword arguments passed in head_model_artifact.

        Returns
        -------
        bool
            Whether the artifact exists.
        """

        try:
            response = self.ds_client.head_model_artifact(model_id=model_id, **kwargs)
            return response.status == 200
        except oci.exceptions.ServiceError as ex:
            if ex.status == 404:
                logger.info(f"Artifact not found in model {model_id}.")
                return False

    @cached(cache=TTLCache(maxsize=5, ttl=timedelta(minutes=1), timer=datetime.now))
    def get_config_from_metadata(
        self,
        model_id: str,
        metadata_key: str,
    ) -> ModelConfigResult:
        """Gets the config for the given Aqua model from model catalog metadata content.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.
        metadata_key: str
            The metadata key name where artifact content is stored
        Returns
        -------
        ModelConfigResult
            A Pydantic model containing the model_details (extracted from OCI) and the config dictionary.
        """
        config: Dict[str, Any] = {}
        oci_model = self.ds_client.get_model(model_id).data

        try:
            config = self.ds_client.get_model_defined_metadatum_artifact_content(
                model_id, metadata_key
            ).data.content.decode("utf-8")
            return ModelConfigResult(config=json.loads(config), model_details=oci_model)
        except UnicodeDecodeError as ex:
            logger.error(
                f"Failed to decode content for '{metadata_key}' in defined metadata for model '{model_id}' : {ex}"
            )
        except json.JSONDecodeError as ex:
            logger.error(
                f"Invalid JSON format for '{metadata_key}' in defined metadata for model '{model_id}' : {ex}"
            )
        except Exception as ex:
            logger.error(
                f"Failed to retrieve defined metadata key '{metadata_key}' for model '{model_id}': {ex}"
            )
        return ModelConfigResult(config=config, model_details=oci_model)

    @cached(cache=TTLCache(maxsize=1, ttl=timedelta(minutes=5), timer=datetime.now))
    def get_config(
        self,
        model_id: str,
        config_file_name: str,
        config_folder: Optional[str] = ConfigFolder.CONFIG,
    ) -> ModelConfigResult:
        """
        Gets the configuration for the given Aqua model along with the model details.

        Parameters
        ----------
        model_id : str
            The OCID of the Aqua model.
        config_file_name : str
            The name of the configuration file.
        config_folder : Optional[str]
            The subfolder path where config_file_name is searched.
            Defaults to ConfigFolder.CONFIG. For model artifact directories, use ConfigFolder.ARTIFACT.

        Returns
        -------
        ModelConfigResult
            A Pydantic model containing the model_details (extracted from OCI) and the config dictionary.
        """
        config: Dict[str, Any] = {}
        oci_model = self.ds_client.get_model(model_id).data

        config_folder = config_folder or ConfigFolder.CONFIG
        oci_aqua = (
            (
                Tags.AQUA_TAG in oci_model.freeform_tags
                or Tags.AQUA_TAG.lower() in oci_model.freeform_tags
            )
            if oci_model.freeform_tags
            else False
        )
        if not oci_aqua:
            logger.debug(f"Target model {oci_model.id} is not an Aqua model.")
            return ModelConfigResult(config=config, model_details=oci_model)

        artifact_path = get_artifact_path(oci_model.custom_metadata_list)
        if not artifact_path:
            logger.debug(
                f"Failed to get artifact path from custom metadata for the model: {model_id}"
            )
            return ModelConfigResult(config=config, model_details=oci_model)

        config_path = os.path.join(os.path.dirname(artifact_path), config_folder)
        if not is_path_exists(config_path):
            config_path = os.path.join(artifact_path.rstrip("/"), config_folder)
            if not is_path_exists(config_path):
                config_path = f"{artifact_path.rstrip('/')}/"
        config_file_path = os.path.join(config_path, config_file_name)
        if is_path_exists(config_file_path):
            try:
                logger.info(
                    f"Loading config: `{config_file_name}` from `{config_path}`"
                )
                config = load_config(
                    config_path,
                    config_file_name=config_file_name,
                )
            except Exception:
                logger.debug(
                    f"Error loading the {config_file_name} at path {config_path}.\n"
                    f"{traceback.format_exc()}"
                )

        if not config:
            logger.debug(
                f"{config_file_name} is not available for the model: {model_id}. "
                f"Check if the custom metadata has the artifact path set."
            )

        return ModelConfigResult(config=config, model_details=oci_model)

    def get_container_image(self, container_type: str = None) -> str:
        """
        Gets the latest smc container complete image name from the given container type.

        Parameters
        ----------
        container_type: str
            type of container, can be either odsc-vllm-serving, odsc-llm-fine-tuning, odsc-llm-evaluate

        Returns
        -------
        str:
            A complete container name along with version. ex: dsmc://odsc-vllm-serving:0.7.4.1
        """

        containers = self.list_service_containers()
        container = next(
            (c for c in containers if c.is_latest and c.family_name == container_type),
            None,
        )
        if not container:
            raise AquaValueError(f"Invalid container type : {container_type}")
        container_image = (
            SERVICE_MANAGED_CONTAINER_URI_SCHEME
            + container.container_name
            + ":"
            + container.tag
        )
        return container_image

    @cached(cache=TTLCache(maxsize=20, ttl=timedelta(minutes=30), timer=datetime.now))
    def list_service_containers(self) -> List[ContainerSummary]:
        """
        List containers from containers.conf in OCI Datascience control plane
        """
        containers = self.ds_client.list_containers().data
        return containers

    def get_container_config(self) -> AquaContainerConfig:
        """
        Fetches latest containers from containers.conf in OCI Datascience control plane

        Returns
        -------
        AquaContainerConfig
            An Object that contains latest container info for the given container family

        """
        return AquaContainerConfig.from_service_config(
            service_containers=self.list_service_containers()
        )

    def get_container_config_item(
        self, container_family: str
    ) -> AquaContainerConfigItem:
        """
        Fetches latest container for given container_family_name from containers.conf in OCI Datascience control plane

        Returns
        -------
        AquaContainerConfigItem
            An Object that contains latest container info for the given container family

        """

        aqua_container_config = self.get_container_config()
        inference_config = aqua_container_config.inference.values()
        ft_config = aqua_container_config.finetune.values()
        eval_config = aqua_container_config.evaluate.values()
        container = next(
            (
                container
                for container in chain(inference_config, ft_config, eval_config)
                if container.family.lower() == container_family.lower()
            ),
            None,
        )
        return container

    @property
    def telemetry(self):
        if not self._telemetry:
            self._telemetry = TelemetryClient(
                bucket=AQUA_TELEMETRY_BUCKET, namespace=AQUA_TELEMETRY_BUCKET_NS
            )
        return self._telemetry


class CLIBuilderMixin:
    """
    CLI builder from API interface. To be used with the DataClass only.
    """

    def build_cli(self) -> str:
        """
        Method to turn the dataclass attributes to CLI
        """
        cmd = f"ads aqua {self._command}"
        params = [
            (
                f"--{field.name} {json.dumps(getattr(self, field.name))}"
                if isinstance(getattr(self, field.name), dict)
                else f"--{field.name} {getattr(self, field.name)}"
            )
            for field in fields(self.__class__)
            if getattr(self, field.name) is not None
        ]
        cmd = f"{cmd} {' '.join(params)}"
        return cmd
