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
    ModelDeploymentMode,
)
from ads.common.serializer import DataClassSerializable
from ads.aqua.exception import AquaClientError, AquaServiceError
from ads.config import COMPARTMENT_OCID
from oci.exceptions import ServiceError, ClientError

AQUA_SERVICE_MODEL = "aqua_service_model"
CONSOLE_LINK_URL = (
    "https://cloud.oracle.com/data-science/model-deployments/{}?region={}"
)

logger = logging.getLogger(__name__)


@dataclass(repr=False)
class AquaDeployment(DataClassSerializable):
    """Represents an Aqua Model Deployment"""

    id: str
    display_name: str
    aqua_service_model: str
    state: str
    description: str
    created_on: str
    created_by: str
    endpoint: str
    console_link: str


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
    """

    def create(
        self,
        model_id: str,
        compartment_id: str,
        aqua_service_model: str,
        instance_count: int,
        instance_shape: str,
        log_group_id: str,
        access_log_id: str,
        predict_log_id: str,
        deployment_image: str,
        entrypoint: List[str],
        project_id: str = None,
        region: str = None,
        display_name: str = None,
        description: str = None,
        bandwidth_mbps: int = None,
        web_concurrency: int = None,
        server_port: int = 5000,
        health_check_port: int = 5000,
        env_var: Dict = None,
        **kwargs,
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
        instance_shape: (str). Default to `VM.Standard2.1`.
            The shape of the instance used for deployment.
        log_group_id: (str)
            The oci logging group id. The access log and predict log share the same log group.
        access_log_id: (str).
            The access log OCID for the access logs. https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm
        predict_log_id: (str).
            The predict log OCID for the predict logs. https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm
        bandwidth_mbps: (int). Defaults to 10.
            The bandwidth limit on the load balancer in Mbps.
        web_concurrency: str
            The number of worker processes/threads to handle incoming requests
        deployment_image: (str).
            The OCIR path of docker container image. Required for deploying model on container runtime.
        with_bucket_uri(bucket_uri)
            Sets the bucket uri when uploading large size model.
        entrypoint: (List). Defaults to empty.
            The entrypoint for running docker container image.
        server_port: (int). Defaults to 8080.
            The server port for docker container image.
        health_check_port: (int). Defaults to 8080.
            The health check port for docker container image.
        env_var : dict, optional
            Environment variable for the deployment, by default None.
        Returns
        -------
        AquaDeployment
            An Aqua deployment instance

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
        # configure model deployment runtime
        # todo : any other runtime params needed?
        container_runtime = (
            ModelDeploymentContainerRuntime()
            .with_image(deployment_image)
            .with_entrypoint(entrypoint)
            .with_server_port(server_port)
            .with_health_check_port(health_check_port)
            .with_env(env_var)
            .with_deployment_mode(ModelDeploymentMode.HTTPS)
            .with_model_uri(model_id)
            .with_region(self.region)
            .with_overwrite_existing_artifact(False)
            .with_remove_existing_artifact(False)
        )
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
            id=deployment.model_deployment_id,
            display_name=deployment.display_name,
            aqua_service_model=aqua_service_model,
            state=deployment.status.name,
            description=deployment.description,
            created_on=deployment.time_created,
            created_by=deployment.created_by,
            endpoint=deployment.url,
            console_link=CONSOLE_LINK_URL.format(
                deployment.model_deployment_id, self.region
            ),
        )

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
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)

        model_deployments = self.list_resource(
            self.client.list_model_deployments, compartment_id=compartment_id, **kwargs
        )

        results = []
        for model_deployment in model_deployments:
            aqua_service_model = (
                model_deployment.freeform_tags.get(AQUA_SERVICE_MODEL, None)
                if model_deployment.freeform_tags
                else None
            )
            if aqua_service_model:
                results.append(
                    AquaDeployment(
                        id=model_deployment.id,
                        display_name=model_deployment.display_name,
                        aqua_service_model=aqua_service_model,
                        state=model_deployment.lifecycle_state,
                        description=model_deployment.description,
                        created_on=str(model_deployment.time_created),
                        created_by=model_deployment.created_by,
                        endpoint=model_deployment.model_deployment_url,
                        console_link=CONSOLE_LINK_URL.format(
                            model_deployment.id, self.region
                        ),
                    )
                )

        return results

    def get(self, model_deployment_id: str, **kwargs) -> "AquaDeployment":
        """Gets the information of Aqua model deployment.

        Parameters
        ----------
        model_deployment_id: str
            The OCID of the Aqua model deployment.

        kwargs
            Keyword arguments, for `get_model_deployment
            <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/data_science/client/oci.data_science.DataScienceClient.html#oci.data_science.DataScienceClient.get_model_deployment>`_

        Returns
        -------
        AquaDeployment:
            The instance of the Aqua model deployment.
        """
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

        aqua_service_model = (
            model_deployment.freeform_tags.get(AQUA_SERVICE_MODEL, None)
            if model_deployment.freeform_tags
            else None
        )

        if not aqua_service_model:
            raise AquaClientError(
                # todo: add link to point users to check the documentation
                f"Target deployment {model_deployment.id} is not compatible with AI Quick Actions."
            )

        return AquaDeployment(
            id=model_deployment.id,
            display_name=model_deployment.display_name,
            aqua_service_model=aqua_service_model,
            state=model_deployment.lifecycle_state,
            description=model_deployment.description,
            created_on=str(model_deployment.time_created),
            created_by=model_deployment.created_by,
            endpoint=model_deployment.model_deployment_url,
            console_link=CONSOLE_LINK_URL.format(model_deployment.id, self.region),
        )
