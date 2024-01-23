#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import List, Dict

from dataclasses import dataclass
from ads.aqua.base import AquaApp
from ads.config import COMPARTMENT_OCID
from ads.model.deployment.model_deployment import ModelDeployment


logger = logging.getLogger(__name__)

@dataclass
class AquaDeployment:
    """Represents an Aqua Model Deployment"""
    display_name: str
    aqua_service_model: str
    state: str
    description: str
    created_on: str
    created_by: str


class AquaDeploymentApp(AquaApp):
    """Contains APIs for Aqua deployments.

    Attributes
    ----------

    Methods
    -------
    create(self, **kwargs) -> "AquaDeployment"
        Creates an instance of model deployment via Aqua
    list(self, ..., **kwargs) -> List["AquaDeployment"]
        List existing model deployments created via Aqua
    clone()
        Clone an existing model deployment
    suggest()
        Provide suggestions for model deployment via Aqua
    stats()
        Get model deployment statistics
    """

    def create(self, **kwargs) -> "AquaDeployment":
        pass

    def list(self, **kwargs) -> List["AquaDeployment"]:
        return [
            AquaDeployment(
                **{
                    "display_name": f"aqua model deployment {i}",
                    "aqua_service_model": f"aqua service model {i}",
                    "state": "ACTIVE" if i%2==0 else "FAILED",
                    "description": "test description",
                    "created_on": "test created on",
                    "created_by": "test created by"
                }
            ) for i in range(8)
        ]

    def clone(self, **kwargs) -> "AquaDeployment":
        pass

    def suggest(self, **kwargs) -> Dict:
        pass

    def stats(self, **kwargs) -> Dict:
        pass
