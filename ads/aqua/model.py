#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
from dataclasses import dataclass
from typing import List
from ads.aqua.base import AquaApp

logger = logging.getLogger(__name__)

@dataclass
class AquaModelSummary:
    """Represents a summary of Aqua model."""
    id: str
    compartment_id: str
    project_id: str
    created_by: str
    display_name: str
    lifecycle_state: str
    time_created: str
    task: str
    license: str
    organization: str
    is_fine_tuned: bool
    model_card: str

@dataclass
class AquaModel(AquaModelSummary):
    """Represents an Aqua model."""
    icon: str = None

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
            **{
                "compartment_id": "ocid1.compartment.oc1..aaaaaaaaktvnqqbjs6tiwhoy7vfgjnikzz6gz322sg3nxqmvyyv4h4sokxia",
                "project_id": "ocid1.datascienceproject.oc1.eu-frankfurt-1.amaaaaaay75uckqaxkxeqtztujrlwjxh3fdywnjvlidbwdhue3jik2bq4elq",
                "created_by": "ocid1.user.oc1..aaaaaaaaodynsttry7fz5xttwsogkeegy55qrrmyshxehogqlureuahe4ala",
                "display_name": "codellama/CodeLlama-7b-Instruct-hf",
                "id": "ocid1.datasciencemodel.oc1.eu-frankfurt-1.amaaaaaay75uckqalphbxlkaxzncdkun5c67gnmpqzd6wdjfgaydbuuqzlka",
                "lifecycle_state": "ACTIVE",
                "time_created": "2024-01-08T22:45:42.443000+00:00",
                "icon": "The icon of the model",
                "task": "text_generation",
                "license": "Apache 2.0",
                "organization": "Meta AI",
                "is_fine_tuned": False,
                "model_card": "This will be the content of the model card from Huggingface."
            }
        )

    def list(self, compartment_id, project_id=None, **kwargs) -> List["AquaModelSummary"]:
        """Lists Aqua models."""
        return [
            AquaModel(id=f"ocid{i}", compartment_id=compartment_id, project_id=project_id)
            for i in range(5)
        ]
