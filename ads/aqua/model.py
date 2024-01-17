#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from dataclasses import dataclass
from typing import List

from ads.config import COMPARTMENT_OCID
from ads.aqua.base import AquaApp

logger = logging.getLogger(__name__)


# Freeform-tag/Define-tag
# key=OCI_AQUA (100 chars max), val = (256 chars max)

AQUA_TAG = "OCI_AQUA"
AQUA_SERVICE_MODEL_TAG = "aqua_service_model"


@dataclass
class AquaModelSummary:
    """Represents a summary of Aqua model."""

    name: str
    ocid: str
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
    ) -> List[dict]:
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

        models = self.list_resource(kwargs)

        aqua_models = []
        for model in models:  # ModelSummary
            if model.freeform_tags.contains(
                AQUA_TAG
            ) and not model.freeform_tags.contains(AQUA_SERVICE_MODEL_TAG):
                aqua_models.append(AquaModel(**model).to_dict())
        return aqua_models
