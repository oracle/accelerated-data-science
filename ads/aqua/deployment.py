#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import List, Dict

from dataclasses import dataclass
from ads.aqua.base import AquaApp
from ads.config import COMPARTMENT_OCID


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
        """List Aqua model deployments in a given compartment and under certain project.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as compartment_id and project_id,
            for `list_call_get_all_results <https://docs.oracle.com/en-us/iaas/tools/python/2.118.1/api/pagination.html#oci.pagination.list_call_get_all_results>`_

        Returns
        -------
        List[AquaDeployment]:
            The list of the Aqua model deployments.
        """
        compartment_id = kwargs.get("compartment_id", None)
        kwargs.update({"compartment_id": compartment_id or COMPARTMENT_OCID})

        model_deployments = self.list_resource(self.client.list_model_deployments, **kwargs)
        return [
            AquaDeployment(
                display_name=model_deployment.display_name,
                aqua_service_model=model_deployment.model_deployment_configuration_details.model_configuration_details.model_id,
                state=model_deployment.lifecycle_state,
                description=model_deployment.description,
                created_on=str(model_deployment.time_created),
                created_by=model_deployment.created_by
            ) for model_deployment in model_deployments
        ]

    def clone(self, **kwargs) -> "AquaDeployment":
        pass

    def suggest(self, **kwargs) -> Dict:
        pass

    def stats(self, **kwargs) -> Dict:
        pass
