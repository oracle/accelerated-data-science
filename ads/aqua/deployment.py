#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import List, Dict

from dataclasses import dataclass
from ads.aqua.base import AquaApp
from ads.config import COMPARTMENT_OCID


AQUA_SERVICE_MODEL = "aqua_service_model"


logger = logging.getLogger(__name__)

@dataclass
class AquaDeployment:
    """Represents an Aqua Model Deployment"""
    id: str
    display_name: str
    aqua_service_model: str
    state: str
    description: str
    created_on: str
    created_by: str
    endpoint: str
    instance_shape: str
    ocpus: float
    memory_in_gbs: float


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

        results = []
        for model_deployment in model_deployments:
            aqua_service_model=(
                model_deployment.freeform_tags.get(AQUA_SERVICE_MODEL, None)
                if model_deployment.freeform_tags else None
            )
            if aqua_service_model:
                results.append(
                    AquaDeploymentApp.from_oci_model_deployment(model_deployment)
                )

        return results

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
        # add error handler
        # if not kwargs.get("model_deployment_id", None):
        #     raise AquaClientError("Aqua model deployment ocid must be provided to fetch the deployment.")
        
        # add error handler
        model_deployment = self.client.get_model_deployment(**kwargs).data
        
        aqua_service_model=(
            model_deployment.freeform_tags.get(AQUA_SERVICE_MODEL, None) 
            if model_deployment.freeform_tags else None
        )

        # add error handler
        # if not aqua_service_model:
        #     raise AquaClientError(f"Target deployment {model_deployment.id} is not Aqua deployment.")

        return AquaDeploymentApp.from_oci_model_deployment(model_deployment)
    
    @classmethod
    def from_oci_model_deployment(cls, oci_model_deployment) -> "AquaDeployment":
        """Converts oci model deployment response to AquaDeployment instance.

        Parameters
        ----------
        oci_model_deployment: oci.data_science.models.ModelDeployment
            The oci.data_science.models.ModelDeployment instance.

        Returns
        -------
        AquaDeployment:
            The instance of the Aqua model deployment.
        """
        instance_configuration = (
            oci_model_deployment
            .model_deployment_configuration_details
            .model_configuration_details
            .instance_configuration
        )
        instance_shape_config_details = (
            instance_configuration.model_deployment_instance_shape_config_details
        )
        return AquaDeployment(
            id=oci_model_deployment.id,
            display_name=oci_model_deployment.display_name,
            aqua_service_model=oci_model_deployment.freeform_tags.get(AQUA_SERVICE_MODEL),
            state=oci_model_deployment.lifecycle_state,
            description=oci_model_deployment.description,
            created_on=str(oci_model_deployment.time_created),
            created_by=oci_model_deployment.created_by,
            endpoint=oci_model_deployment.model_deployment_url,
            instance_shape=instance_configuration.instance_shape_name,
            ocpus=instance_shape_config_details.ocpus,
            memory_in_gbs=instance_shape_config_details.memory_in_gbs
        ) 
