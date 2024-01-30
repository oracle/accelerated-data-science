#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import List, Dict

from dataclasses import dataclass
from ads.aqua.base import AquaApp
from ads.aqua.exception import AquaClientError, AquaServiceError
from ads.config import COMPARTMENT_OCID
from oci.exceptions import ServiceError, ClientError


AQUA_SERVICE_MODEL = "aqua_service_model"


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
        model_deployments = self.list_resource(
            self.client.list_model_deployments, **kwargs
        )
        return [
            AquaDeployment(
                display_name=model_deployment.display_name,
                aqua_service_model=(
                    model_deployment.freeform_tags.get(AQUA_SERVICE_MODEL, None)
                    if model_deployment.freeform_tags
                    else None
                ),
                state=model_deployment.lifecycle_state,
                description=model_deployment.description,
                created_on=str(model_deployment.time_created),
                created_by=model_deployment.created_by,
            )
            for model_deployment in model_deployments
        ]

    def clone(self, **kwargs) -> "AquaDeployment":
        pass

    def suggest(self, **kwargs) -> Dict:
        pass

    def stats(self, **kwargs) -> Dict:
        pass

    def get(self, **kwargs) -> "AquaDeployment":
        """Gets the information of Aqua model deployment.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as model_deployment_id,
            for `get_model_deployment <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/data_science/client/oci.data_science.DataScienceClient.html#oci.data_science.DataScienceClient.get_model_deployment>`_

        Returns
        -------
        AquaDeployment:
            The instance of the Aqua model deployment.
        """
        model_deployment_id = kwargs.get("model_deployment_id", None)
        if not model_deployment_id:
            raise AquaClientError(
                "Aqua model deployment ocid must be provided to fetch the deployment."
            )

        try:
            model_deployment = self.client.get_model_deployment(
                model_deployment_id=model_deployment_id, **kwargs
            ).data
        except ServiceError as se:
            raise AquaServiceError(opc_request_id=se.request_id, status_code=se.code)
        except ClientError as ce:
            raise AquaClientError(str(ce))

        return AquaDeployment(
            display_name=model_deployment.display_name,
            aqua_service_model=(
                model_deployment.freeform_tags.get(AQUA_SERVICE_MODEL, None)
                if model_deployment.freeform_tags
                else None
            ),
            state=model_deployment.lifecycle_state,
            description=model_deployment.description,
            created_on=str(model_deployment.time_created),
            created_by=model_deployment.created_by,
        )
