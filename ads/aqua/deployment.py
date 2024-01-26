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
        aqua_service_model_tag: str = None,
        display_name: str = None,
        description: str = None,
        instance_count: int = None,
        instance_shape: str = None,
        access_log_group_id: str = None,
        access_log_id: str = None,
        predict_log_group_id: str = None,
        predict_log_id: str = None,
        bandwidth: int = None,
        web_concurrency: int = None,
        deployment_image: str = None,
        server_port: int = None,
        health_check_port: int = None,
        wait_for_completion: bool = False,
        **kwargs,
    ) -> "AquaDeployment":
        env_var = {}
        # configure model deployment infrastructure
        infrastructure = (
            ModelDeploymentInfrastructure()
            .with_project_id(project_id)
            .with_compartment_id(compartment_id)
            .with_shape_name(instance_shape)
            .with_bandwidth_mbps(bandwidth)
            .with_replica(instance_count)
            .with_web_concurrency(web_concurrency)
            .with_access_log(
                log_group_id=access_log_group_id,
                log_id=access_log_id,
            )
            .with_predict_log(
                log_group_id=predict_log_group_id,
                log_id=predict_log_id,
            )
        )
        # configure model deployment runtime
        container_runtime = (
            ModelDeploymentContainerRuntime()
            .with_image(deployment_image)
            .with_server_port(server_port)
            .with_health_check_port(health_check_port)
            .with_env(env_var)
            .with_model_uri(model_id)
            .with_region(region)
            .with_overwrite_existing_artifact(False)
            .with_remove_existing_artifact(False)
        )
        # configure model deployment
        deployment = (
            ModelDeployment()
            .with_display_name(display_name)
            .with_description(description)
            .with_freeform_tags()
            .with_infrastructure(infrastructure)
            .with_runtime(container_runtime)
        )
        # Deploy model on container runtime
        deployment.deploy(wait_for_completion=wait_for_completion)

        return AquaDeployment(
            display_name=deployment.display_name,
            aqua_service_model=aqua_service_model_tag,
            status=deployment.status.name,
            description=deployment.description,
            created_on=deployment.time_created,
            created_by=deployment.created_by,
            endpoint=deployment.model_deployment_url,
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
