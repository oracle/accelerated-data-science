#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import List, Dict

from dataclasses import dataclass
from ads.aqua.base import AquaApp
from ads.model.deployment import (
    ModelDeployment,
    ModelDeploymentInfrastructure,
    ModelDeploymentContainerRuntime,
)

logger = logging.getLogger(__name__)


@dataclass
class AquaDeployment:
    """Represents an Aqua Model Deployment"""

    display_name: str
    aqua_service_model: str
    status: str
    description: str
    created_on: str
    created_by: str
    endpoint: str


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

    def create(
        self,
        compartment_id: str = None,
        project_id: str = None,
        region: str = None,
        model_id: str = None,
        aqua_service_model: str = None,
        display_name: str = None,
        description: str = None,
        instance_count: int = None,
        instance_shape: str = None,
        log_group_id: str = None,
        access_log_id: str = None,
        predict_log_id: str = None,
        bandwidth_mbps: int = None,
        web_concurrency: int = None,
        deployment_image: str = None,
        entrypoint: List[str] = None,
        server_port: int = 5000,
        health_check_port: int = 5000,
        env_var: Dict = None,
    ) -> "AquaDeployment":
        """
        Creates a new Aqua deployment

        Parameters
        ----------
        model_id: str
            The model OCID to deploy.
        compartment_id: str
            The compartment OCID
        project_id: str
            Target project to list deployments from.
        region: str (None)
            The Region Identifier that the client should connect to.
            Regions can be found here:
            https://docs.oracle.com/en-us/iaas/Content/General/Concepts/regions.htm
        aqua_service_model: str
            The aqua tag that gets passed from the model that identifies the type of model
        display_name: str
            The name of model deployment.
        description: str
            The description of the deployment.
        instance_count: (int, optional). Defaults to 1.
            The number of instance used for deployment.
        instance_shape: (str, optional). Default to `VM.Standard2.1`.
            The shape of the instance used for deployment.
        log_group_id: (str, optional)
            The oci logging group id. The access log and predict log share the same log group.
        access_log_id: (str, optional). Defaults to None.
            The access log OCID for the access logs. https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm
        predict_log_id: (str, optional). Defaults to None.
            The predict log OCID for the predict logs. https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm
        bandwidth_mbps: (int, optional). Defaults to 10.
            The bandwidth limit on the load balancer in Mbps.
        web_concurrency: str
            The number of worker processes/threads to handle incoming requests
        deployment_image: (str, optional). Defaults to None.
            The OCIR path of docker container image. Required for deploying model on container runtime.
        with_bucket_uri(bucket_uri)
            Sets the bucket uri when uploading large size model.
        entrypoint: (List, optional). Defaults to empty.
            The entrypoint for running docker container image.
        server_port: (int, optional). Defaults to 8080.
            The server port for docker container image.
        health_check_port: (int, optional). Defaults to 8080.
            The health check port for docker container image.
        env_var : dict, optional
            Environment variable for the deployment, by default None.
        Returns
        -------
        Session
            The playground session instance.

        Raises
        ------
        ValueError
            If model ID not provided.
        """
        # todo: create a model catalog entry with model path pointing to service bucket

        # Start model deployment
        # configure model deployment infrastructure
        # todo : any other infrastructure params needed?
        infrastructure = (
            ModelDeploymentInfrastructure()
            .with_project_id(project_id)
            .with_compartment_id(compartment_id)
            .with_shape_name(instance_shape)
            .with_bandwidth_mbps(bandwidth_mbps)
            .with_replica(instance_count)
            .with_web_concurrency(web_concurrency)
            .with_access_log(
                log_group_id=log_group_id,
                log_id=access_log_id,
            )
            .with_predict_log(
                log_group_id=log_group_id,
                log_id=predict_log_id,
            )
        )
        logger.info(f"Infra: {infrastructure}")
        # configure model deployment runtime
        # todo : any other runtime params needed?
        container_runtime = (
            ModelDeploymentContainerRuntime()
            .with_image(deployment_image)
            .with_entrypoint(entrypoint)
            .with_server_port(server_port)
            .with_health_check_port(health_check_port)
            .with_env(env_var)
            .with_deployment_mode("HTTPS_ONLY")
            .with_model_uri(model_id)
            .with_region(region)
            .with_overwrite_existing_artifact(False)
            .with_remove_existing_artifact(False)
        )
        logger.info(f"Infra: {container_runtime}")
        # configure model deployment and deploy model on container runtime
        # todo : any other deployment params needed?
        deployment = (
            ModelDeployment()
            .with_display_name(display_name)
            .with_description(description)
            .with_freeform_tags(aqua_service_model=aqua_service_model)
            .with_infrastructure(infrastructure)
            .with_runtime(container_runtime)
        ).deploy(wait_for_completion=False)

        return AquaDeployment(
            display_name=deployment.display_name,
            aqua_service_model=aqua_service_model,
            status=deployment.status.name,
            description=deployment.description,
            created_on=deployment.time_created,
            created_by=deployment.created_by,
            endpoint=deployment.url,
        )

    def list(self, **kwargs) -> List["AquaDeployment"]:
        return [
            AquaDeployment(
                **{
                    "display_name": f"aqua model deployment {i}",
                    "aqua_service_model": f"aqua service model {i}",
                    "state": "ACTIVE" if i % 2 == 0 else "FAILED",
                    "description": "test description",
                    "created_on": "test created on",
                    "created_by": "test created by",
                }
            )
            for i in range(8)
        ]
