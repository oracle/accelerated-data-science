#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class AquaModelSummary:
    """Represents a summary of Aqua model."""
    id: str
    compartment_id: str
    project_id: str


@dataclass
class AquaModel(AquaModelSummary):
    """Represents an Aqua model."""
    icon: str = None

class AquaModelApp:
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
        return AquaModel(id=model_id, compartment_id="ocid1.compartment", project_id="ocid1.project")

    def list(self, compartment_id, project_id=None, **kwargs) -> List["AquaModelSummary"]:
        """Lists Aqua models."""
        return [
            AquaModel(id=f"ocid{i}", compartment_id=compartment_id, project_id=project_id)
            for i in range(5)
        ]
