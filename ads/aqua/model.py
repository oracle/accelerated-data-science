#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from dataclasses import dataclass
from enum import Enum
from typing import List

import fsspec
from oci.exceptions import ClientError, ServiceError

from ads.aqua import logger
from ads.aqua.base import AquaApp
from ads.aqua.exception import AquaClientError, AquaServiceError
from ads.aqua.utils import create_word_icon
from ads.common.serializer import DataClassSerializable
from ads.config import COMPARTMENT_OCID

ICON_FILE_NAME = "icon.txt"
README = "README.md"
UNKNOWN = "Unknown"


class Tags(Enum):
    TASK = "task"
    LICENSE = "license"
    ORGANIZATION = "organization"
    AQUA_TAG = "OCI_AQUA"
    AQUA_SERVICE_MODEL_TAG = "aqua_service_model"
    AQUA_FINE_TUNED_MODEL_TAG = "aqua_fine_tuned_model"


@dataclass(repr=False)
class AquaModelSummary(DataClassSerializable):
    """Represents a summary of Aqua model."""

    name: str
    id: str
    compartment_id: str
    project_id: str
    time_created: str
    icon: str
    task: str
    license: str
    organization: str
    is_fine_tuned_model: bool


@dataclass(repr=False)
class AquaModel(AquaModelSummary, DataClassSerializable):
    """Represents an Aqua model."""

    model_card: str


class AquaModelApp(AquaApp):
    """Contains APIs for Aqua model.

    Attributes
    ----------

    Methods
    -------
    create(self, **kwargs) -> "AquaModel"
        Creates an instance of Aqua model.
    get(..., **kwargs)
        Gets the information of an Aqua model.
    list(...) -> List["AquaModelSummary"]
        List existing models created via Aqua.
    """

    def create(self, **kwargs) -> "AquaModel":
        pass

    def get(self, model_id) -> "AquaModel":
        """Gets the information of an Aqua model.

        Parameters
        ----------
        model_id: str
            The model OCID.

        Returns
        -------
        AquaModel:
            The instance of the Aqua model.
        """
        try:
            oci_model = self.client.get_model(model_id).data
        except ServiceError as se:
            raise AquaServiceError(opc_request_id=se.request_id, status_code=se.code)
        except ClientError as ce:
            raise AquaClientError(str(ce))

        if not self._if_show(oci_model):
            raise AquaClientError(f"Target model {oci_model.id} is not Aqua model.")

        custom_metadata_list = oci_model.custom_metadata_list
        artifact_path = self._get_artifact_path(custom_metadata_list)

        return AquaModel(
            compartment_id=oci_model.compartment_id,
            project_id=oci_model.project_id,
            name=oci_model.display_name,
            id=oci_model.id,
            time_created=oci_model.time_created,
            icon=str(self._read_file(f"{artifact_path}/{ICON_FILE_NAME}")),
            task=oci_model.freeform_tags.get(Tags.TASK.value, UNKNOWN),
            license=oci_model.freeform_tags.get(Tags.LICENSE.value, UNKNOWN),
            organization=oci_model.freeform_tags.get(Tags.ORGANIZATION.value, UNKNOWN),
            is_fine_tuned_model=True
            if oci_model.freeform_tags.get(Tags.AQUA_FINE_TUNED_MODEL_TAG.value)
            else False,
            model_card=str(self._read_file(f"{artifact_path}/{README}")),
        )

    def list(
        self, compartment_id: str = None, project_id: str = None, **kwargs
    ) -> List["AquaModelSummary"]:
        """List Aqua models in a given compartment and under certain project.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        project_id: (str, optional). Defaults to `None`.
            The project OCID.
        kwargs
            Additional keyword arguments for `list_call_get_all_results <https://docs.oracle.com/en-us/iaas/tools/python/2.118.1/api/pagination.html#oci.pagination.list_call_get_all_results>`_

        Returns
        -------
        List[AquaModelSummary]:
            The list of the `ads.aqua.model.AquaModelSummary`.
        """
        compartment_id = compartment_id or COMPARTMENT_OCID
        kwargs.update({"compartment_id": compartment_id, "project_id": project_id})
        models = self.list_resource(self.client.list_models, **kwargs)
        if not models:
            logger.error(
                f"No model found in compartment_id={compartment_id}, project_id={project_id}."
            )

        aqua_models = []
        # TODO: build index.json locally as caching if needed.

        def process_model(model):
            icon = self._load_icon(model.display_name)

            return AquaModelSummary(
                name=model.display_name,
                id=model.id,
                compartment_id=model.compartment_id,
                project_id=model.project_id,
                time_created=model.time_created,
                icon=icon,
                task=model.freeform_tags.get(Tags.TASK.value, UNKNOWN),
                license=model.freeform_tags.get(Tags.LICENSE.value, UNKNOWN),
                organization=model.freeform_tags.get(Tags.ORGANIZATION.value, UNKNOWN),
                is_fine_tuned_model=True
                if model.freeform_tags.get(Tags.AQUA_FINE_TUNED_MODEL_TAG.value)
                else False,
            )

        for model in models:  # ModelSummary
            if self._if_show(model):
                aqua_models.append(process_model(model))

        return aqua_models

    def _if_show(self, model: "ModelSummary") -> bool:
        """Determine if the given model should be return by `list`."""
        TARGET_TAGS = model.freeform_tags.keys()
        if not Tags.AQUA_TAG.value in TARGET_TAGS:
            return False

        return (
            True
            if (
                Tags.AQUA_SERVICE_MODEL_TAG.value in TARGET_TAGS
                or Tags.AQUA_FINE_TUNED_MODEL_TAG.value in TARGET_TAGS
            )
            else False
        )

    def _get_artifact_path(self, custom_metadata_list: List) -> str:
        """Get the artifact path from the custom metadata list of model.

        Parameters
        ----------
        custom_metadata_list: List
            A list of custom metadata of model.

        Returns
        -------
        str:
            The artifact path from model.
        """
        for custom_metadata in custom_metadata_list:
            if custom_metadata.key == "Object Storage Path":
                return custom_metadata.value
        logger.debug("Failed to get artifact path from custom metadata.")
        return None

    def _read_file(self, file_path: str) -> str:
        try:
            with fsspec.open(file_path, "rb", **self._auth) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to retreive model icon. {e}")
            return None

    def _load_icon(self, model_name) -> str:
        """Loads icon."""

        # TODO: switch to the official logo
        try:
            return create_word_icon(model_name, return_as_datauri=True)
        except Exception as e:
            logger.error(f"Failed to load icon for the model={model_name}.")
            return None
