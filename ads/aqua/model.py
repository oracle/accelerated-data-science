#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import logging
import fsspec
from dataclasses import dataclass
from typing import List
from enum import Enum
from ads.config import COMPARTMENT_OCID
from ads.aqua.base import AquaApp
from ads.model.service.oci_datascience_model import OCIDataScienceModel

logger = logging.getLogger(__name__)

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
        import json
        import os

        root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dummy_data")

        with open(f"{root}/oci_models.json", "rb") as f:
            oci_model = OCIDataScienceModel(**json.loads(f.read())[0])

            return AquaModel(
                compartment_id=oci_model.compartment_id,
                project_id=oci_model.project_id,
                name=oci_model.display_name,
                id=oci_model.id,
                time_created=str(oci_model.time_created),
                icon=self._read_file(f"{root}/{ICON_FILE_NAME}"),
                task=oci_model.freeform_tags.get(Tags.TASK.value, UNKNOWN),
                license=oci_model.freeform_tags.get(
                    Tags.LICENSE.value, UNKNOWN
                ),
                organization=oci_model.freeform_tags.get(
                    Tags.ORGANIZATION.value, UNKNOWN
                ),
                is_fine_tuned_model=True
                if oci_model.freeform_tags.get(
                    Tags.AQUA_FINE_TUNED_MODEL_TAG.value
                )
                else False,
                model_card=self._read_file(f"{root}/{README}")
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
        List[dict]:
            The list of the Aqua models.
        """
        compartment_id = compartment_id or COMPARTMENT_OCID
        kwargs.update({"compartment_id": compartment_id, "project_id": project_id})

        models = self.list_resource(self.client.list_models, **kwargs)

        aqua_models = []
        for model in models:  # ModelSummary
            if self._if_show(model):
                # TODO: need to update after model by reference release
                try:
                    custom_metadata_list = self.client.get_model(
                        model.id
                    ).data.custom_metadata_list
                except Exception as e:
                    # show opc-request-id and status code
                    logger.error(f"Failing to retreive model information. {e}")
                    return []
                
                artifact_path = self._get_artifact_path(custom_metadata_list)

                aqua_models.append(
                    AquaModelSummary(
                        name=model.display_name,
                        id=model.id,
                        compartment_id=model.compartment_id,
                        project_id=model.project_id,
                        time_created=str(model.time_created),
                        icon=self._read_file(f"{artifact_path}/{ICON_FILE_NAME}"),
                        task=model.freeform_tags.get(Tags.TASK.value, UNKNOWN),
                        license=model.freeform_tags.get(
                            Tags.LICENSE.value, UNKNOWN
                        ),
                        organization=model.freeform_tags.get(
                            Tags.ORGANIZATION.value, UNKNOWN
                        ),
                        is_fine_tuned_model=True
                        if model.freeform_tags.get(
                            Tags.AQUA_FINE_TUNED_MODEL_TAG.value
                        )
                        else False,
                    )
                )
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

        raise FileNotFoundError("Failed to retrieve model artifact path from AQUA model.")
    
    def _read_file(self, file_path: str) -> str:
        with fsspec.open(file_path, "rb", **self._auth) as f:
            return f.read()