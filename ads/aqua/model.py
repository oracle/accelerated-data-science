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

logger = logging.getLogger(__name__)

ICON_FILE_NAME = "icon.txt"


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
    time_created: int
    icon: str = None
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
        """Gets the information of an Aqua model."""
        return AquaModel(
            id=model_id, compartment_id="ocid1.compartment", project_id="ocid1.project"
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
                artifact_path = ""
                custom_metadata_list = self.client.get_model(
                    model.id
                ).custom_metadata_list

                for custom_metadata in custom_metadata_list:
                    if custom_metadata.key == "Object Storage Path":
                        artifact_path = custom_metadata.value
                        break
                if not artifact_path:
                    raise FileNotFoundError("Failed to retrieve model artifact path.")

                with fsspec.open(
                    f"{artifact_path}/{ICON_FILE_NAME}", "rb", **self._auth
                ) as f:
                    icon = f.read()
                    aqua_models.append(
                        AquaModelSummary(
                            name=model.display_name,
                            id=model.id,
                            compartment_id=model.compartment_id,
                            project_id=model.project_id,
                            time_created=model.time_created,
                            icon=icon,
                            task=model.freeform_tags.get(Tags.TASK),
                            license=model.freeform_tags.get(Tags.LICENSE),
                            organization=model.freeform_tags.get(Tags.ORGANIZATION),
                            is_fine_tuned_model=True
                            if model.freeform_tags.get(Tags.AQUA_FINE_TUNED_MODEL_TAG)
                            else False,
                        )
                    )
        return aqua_models

    def _if_show(model: "ModelSummary") -> bool:
        """Determine if the given model should be return by `list`."""
        if not model.freeform_tags.contains(Tags.AQUA_TAG):
            return False

        return (
            True
            if model.freeform_tags.contains(Tags.AQUA_SERVICE_MODEL_TAG)
            or model.freeform_tags.contains(Tags.AQUA_FINE_TUNED_MODEL_TAG)
            else False
        )
