#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import fsspec
from dataclasses import dataclass
from typing import List
from enum import Enum
from ads.aqua.exception import AquaClientError, AquaServiceError, oci_exception_handler
from ads.config import COMPARTMENT_OCID
from ads.aqua.base import AquaApp
from oci.exceptions import ServiceError, ClientError
import asyncio


ICON_FILE_NAME = "icon.txt"
README = "readme.md"
UNKNOWN = "Unknown"


class Tags(Enum):
    TASK = "task"
    LICENSE = "license"
    ORGANIZATION = "organization"
    AQUA_TAG = "OCI_AQUA"
    AQUA_SERVICE_MODEL_TAG = "aqua_service_model"
    AQUA_FINE_TUNED_MODEL_TAG = "aqua_fine_tuned_model"


@dataclass
class AquaModelSummary:
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


@dataclass
class AquaModel(AquaModelSummary):
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
    deploy(..., **kwargs)
        Deploys an Aqua model.
    list(self, ..., **kwargs) -> List["AquaModel"]
        List existing models created via Aqua

    """

    def __init__(self, **kwargs):
        """Initializes an Aqua model."""
        super().__init__(**kwargs)

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
            icon=self._read_file(f"{artifact_path}/{ICON_FILE_NAME}"),
            task=oci_model.freeform_tags.get(Tags.TASK.value, UNKNOWN),
            license=oci_model.freeform_tags.get(Tags.LICENSE.value, UNKNOWN),
            organization=oci_model.freeform_tags.get(Tags.ORGANIZATION.value, UNKNOWN),
            is_fine_tuned_model=True
            if oci_model.freeform_tags.get(Tags.AQUA_FINE_TUNED_MODEL_TAG.value)
            else False,
            model_card=self._read_file(f"{artifact_path}/{README}"),
        )

    async def list(
        self, compartment_id: str = None, project_id: str = None, **kwargs
    ) -> List["AquaModelSummary"]:
        compartment_id = compartment_id or COMPARTMENT_OCID
        kwargs.update({"compartment_id": compartment_id, "project_id": project_id})

        models = self.list_resource(self.client.list_models, **kwargs)
        tasks = []

        async def process_model(model):
            # TODO: the way to fetch icon will be updated after model by reference release
            icon = None
            try:
                thismodel = await self._client_get_model(model.id)
                custom_metadata_list = thismodel.data.custom_metadata_list

                artifact_path = self._get_artifact_path(custom_metadata_list)
                if artifact_path:
                    icon = await self._read_file_async(
                        f"{artifact_path}/{ICON_FILE_NAME}"
                    )

            except Exception as e:
                # Failed to retrieve icon, icon remains None
                pass

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
                tasks.append(process_model(model))

        aqua_models = await asyncio.gather(*tasks)
        return aqua_models

    @oci_exception_handler
    async def _client_get_model(self, model_id):
        return await asyncio.to_thread(self.client.get_model, model_id)

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
        """Reads content from given path. Returns None if path cannot be access.

        Parameters
        ----------
        file_path: str
            Object storage path.

        Returns
        -------
        bytes:
            content read from given path.
        """
        try:
            with fsspec.open(file_path, "rb", **self._auth) as f:
                return f.read()
        except Exception as e:
            logger.debug(
                f"Failed to retreive content from `file_path={file_path}`. {e}"
            )
            return None

    async def _read_file_async(self, file_path) -> str:
        try:
            with fsspec.open(file_path, "rb", **self._auth) as f:
                return f.read()
        except Exception as e:
            logger.debug(
                f"Failed to retreive content from `file_path={file_path}`. {e}"
            )
            return None
