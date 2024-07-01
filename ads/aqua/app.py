#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import fields
from typing import Dict, Union

import oci
from oci.data_science.models import UpdateModelDetails, UpdateModelProvenanceDetails

from ads import set_auth
from ads.aqua import logger
from ads.aqua.common.enums import Tags
from ads.aqua.common.errors import AquaRuntimeError, AquaValueError
from ads.aqua.common.utils import (
    _is_valid_mvs,
    get_artifact_path,
    is_valid_ocid,
    load_config,
)
from ads.aqua.constants import UNKNOWN
from ads.common import oci_client as oc
from ads.common.auth import default_signer
from ads.common.utils import extract_region
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

    @telemetry(name="aqua")
    def __init__(self) -> None:
        if OCI_RESOURCE_PRINCIPAL_VERSION:
            set_auth("resource_principal")
        self._auth = default_signer({"service_endpoint": OCI_ODSC_SERVICE_ENDPOINT})
        self.ds_client = oc.OCIClientFactory(**self._auth).data_science
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

    # TODO: refactor model evaluation implementation to use it.
    @staticmethod
    def get_source(source_id: str) -> Union[ModelDeployment, DataScienceModel]:
        if is_valid_ocid(source_id):
            if "datasciencemodeldeployment" in source_id:
                return ModelDeployment.from_id(source_id)
            elif "datasciencemodel" in source_id:
                return DataScienceModel.from_id(source_id)

        raise AquaValueError(
            f"Invalid source {source_id}. "
            "Specify either a model or model deployment id."
        )

    # TODO: refactor model evaluation implementation to use it.
    @staticmethod
    def create_model_version_set(
        model_version_set_id: str = None,
        model_version_set_name: str = None,
        description: str = None,
        compartment_id: str = None,
        project_id: str = None,
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

            except:
                logger.debug(
                    f"Model version set {model_version_set_name} doesn't exist. "
                    "Creating new model version set."
                )
                mvs_freeform_tags = {
                    tag: tag,
                }
                model_version_set = (
                    ModelVersionSet()
                    .with_compartment_id(compartment_id)
                    .with_project_id(project_id)
                    .with_name(model_version_set_name)
                    .with_description(description)
                    .with_freeform_tags(**mvs_freeform_tags)
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
            # TODO: decide what parameters will be needed
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
            return True if response.status == 200 else False
        except oci.exceptions.ServiceError as ex:
            if ex.status == 404:
                logger.info(f"Artifact not found in model {model_id}.")
                return False

    def get_config(self, model_id: str, config_file_name: str) -> Dict:
        """Gets the config for the given Aqua model.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.
        config_file_name: str
            name of the config file

        Returns
        -------
        Dict:
            A dict of allowed configs.
        """
        oci_model = self.ds_client.get_model(model_id).data
        oci_aqua = (
            (
                Tags.AQUA_TAG in oci_model.freeform_tags
                or Tags.AQUA_TAG.lower() in oci_model.freeform_tags
            )
            if oci_model.freeform_tags
            else False
        )

        if not oci_aqua:
            raise AquaRuntimeError(f"Target model {oci_model.id} is not Aqua model.")

        config = {}
        artifact_path = get_artifact_path(oci_model.custom_metadata_list)
        if not artifact_path:
            logger.error(
                f"Failed to get artifact path from custom metadata for the model: {model_id}"
            )
            return config

        try:
            config_path = f"{os.path.dirname(artifact_path)}/config/"
            config = load_config(
                config_path,
                config_file_name=config_file_name,
            )
        except:
            # todo: temp fix for issue related to config load for byom models, update logic to choose the right path
            try:
                config_path = f"{artifact_path.rstrip('/')}/config/"
                config = load_config(
                    config_path,
                    config_file_name=config_file_name,
                )
            except:
                pass

        if not config:
            logger.error(
                f"{config_file_name} is not available for the model: {model_id}. Check if the custom metadata has the artifact path set."
            )
            return config

        return config

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
            f"--{field.name} {getattr(self,field.name)}"
            for field in fields(self.__class__)
            if getattr(self, field.name)
        ]
        cmd = f"{cmd} {' '.join(params)}"
        return cmd
