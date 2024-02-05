#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
from typing import List, Dict

from dataclasses import dataclass, field

from ads.aqua.base import AquaApp
from ads.aqua import logger, utils
from ads.aqua.model import AquaModelApp, Tags
from ads.model.deployment import (
    ModelDeployment,
    ModelDeploymentInfrastructure,
    ModelDeploymentContainerRuntime,
    ModelDeploymentMode,
)
from ads.model.service.oci_datascience_model_deployment import (
    OCIDataScienceModelDeployment,
)
from ads.common.utils import get_console_link
from ads.common.serializer import DataClassSerializable
from ads.aqua.exception import AquaClientError, AquaServiceError
from ads.config import COMPARTMENT_OCID, AQUA_MODEL_DEPLOYMENT_IMAGE

DEPLOYMENT_CONFIG = "deployment_config.json"
DEPLOYMENT_SHAPE = "shape"
UNKNOWN_JSON_STR = "{}"


@dataclass
class ShapeInfo:
    instance_shape: str = None
    instance_count: int = None
    ocpus: float = None
    memory_in_gbs: float = None


@dataclass(repr=False)
class AquaDeployment(DataClassSerializable):
    """Represents an Aqua Model Deployment"""

    id: str = None
    display_name: str = None
    aqua_service_model: bool = None
    state: str = None
    description: str = None
    created_on: str = None
    created_by: str = None
    endpoint: str = None
    console_link: str = None
    shape_info: field(default_factory=ShapeInfo) = None
    tags: dict = None

    @classmethod
    def from_oci_model_deployment(
        cls,
        oci_model_deployment: OCIDataScienceModelDeployment,
        region: str,
    ) -> "AquaDeployment":
        """Converts oci model deployment response to AquaDeployment instance.

        Parameters
        ----------
        oci_model_deployment: oci.data_science.models.ModelDeployment
            The oci.data_science.models.ModelDeployment instance.
        region: str
            The region of this model deployment.

        Returns
        -------
        AquaDeployment:
            The instance of the Aqua model deployment.
        """
        instance_configuration = (
            oci_model_deployment.model_deployment_configuration_details.model_configuration_details.instance_configuration
        )
        instance_shape_config_details = (
            instance_configuration.model_deployment_instance_shape_config_details
        )
        instance_count = (
            oci_model_deployment.model_deployment_configuration_details.model_configuration_details.scaling_policy.instance_count
        )
        shape_info = ShapeInfo(
            instance_shape=instance_configuration.instance_shape_name,
            instance_count=instance_count,
            ocpus=(
                instance_shape_config_details.ocpus
                if instance_shape_config_details
                else None
            ),
            memory_in_gbs=(
                instance_shape_config_details.memory_in_gbs
                if instance_shape_config_details
                else None
            ),
        )

        return AquaDeployment(
            id=oci_model_deployment.id,
            display_name=oci_model_deployment.display_name,
            aqua_service_model=oci_model_deployment.freeform_tags.get(
                Tags.AQUA_SERVICE_MODEL_TAG.value
            )
            is not None,
            shape_info=shape_info,
            state=oci_model_deployment.lifecycle_state,
            description=oci_model_deployment.description,
            created_on=str(oci_model_deployment.time_created),
            created_by=oci_model_deployment.created_by,
            endpoint=oci_model_deployment.model_deployment_url,
            console_link=get_console_link(
                resource="model-deployments",
                ocid=oci_model_deployment.id,
                region=region,
            ),
            tags=oci_model_deployment.freeform_tags,
        )


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
        instance_count: int,
        instance_shape: str,
        log_group_id: str,
        access_log_id: str,
        predict_log_id: str,
        project_id: str = None,
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
        with_bucket_uri(bucket_uri)
            Sets the bucket uri when uploading large size model.
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
        # todo: revisit error handling
        if not AQUA_MODEL_DEPLOYMENT_IMAGE:
            raise AquaClientError(
                f"AQUA_MODEL_DEPLOYMENT_IMAGE must be available in environment variables to "
                f"continue with Aqua model deployment."
            )

        # Create a model catalog entry in the user compartment
        aqua_model = AquaModelApp().create(
            model_id=model_id, comparment_id=compartment_id, project_id=project_id
        )
        logging.debug(
            f"Aqua Model {aqua_model.id} created with the service model {model_id}"
        )
        logging.debug(aqua_model)

        # todo: remove entrypoint, this will go in the image. For now, added for testing
        #  the image iad.ocir.io/ociodscdev/aqua_deploy:1.0.0
        entrypoint = ["python", "/opt/api/api.py"]

        tags = {}
        for tag in [
            Tags.AQUA_SERVICE_MODEL_TAG.value,
            Tags.AQUA_FINE_TUNED_MODEL_TAG.value,
            Tags.AQUA_TAG.value,
        ]:
            if tag in aqua_model.tags:
                tags[tag] = aqua_model.tags[tag]

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
            .with_image(AQUA_MODEL_DEPLOYMENT_IMAGE)
            .with_entrypoint(entrypoint)
            .with_server_port(server_port)
            .with_health_check_port(health_check_port)
            .with_env(env_var)
            .with_deployment_mode(ModelDeploymentMode.HTTPS)
            .with_model_uri(aqua_model.id)
            .with_region(self.region)
            .with_overwrite_existing_artifact(True)
            .with_remove_existing_artifact(True)
        )
        # configure model deployment and deploy model on container runtime
        # todo : any other deployment params needed?
        deployment = (
            ModelDeployment()
            .with_display_name(display_name)
            .with_description(description)
            .with_freeform_tags(**tags)
            .with_infrastructure(infrastructure)
            .with_runtime(container_runtime)
        ).deploy(wait_for_completion=False)

        return AquaDeployment.from_oci_model_deployment(
            deployment.dsc_model_deployment, self.region
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
            self.ds_client.list_model_deployments,
            compartment_id=compartment_id,
            **kwargs,
        )

        results = []
        for model_deployment in model_deployments:
            aqua_service_model = (
                model_deployment.freeform_tags.get(
                    Tags.AQUA_SERVICE_MODEL_TAG.value, None
                )
                if model_deployment.freeform_tags
                else None
            )
            if aqua_service_model:
                results.append(
                    AquaDeployment.from_oci_model_deployment(
                        model_deployment, self.region
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
        try:
            model_deployment = self.ds_client.get_model_deployment(
                model_deployment_id=model_deployment_id, **kwargs
            ).data
        except Exception as se:
            # TODO: adjust error raising
            logging.error(
                f"Failed to retreive model deployment from the given id {model_deployment_id}"
            )
            raise AquaServiceError(opc_request_id=se.request_id, status_code=se.code)

        aqua_service_model = (
            model_deployment.freeform_tags.get(Tags.AQUA_SERVICE_MODEL_TAG.value, None)
            if model_deployment.freeform_tags
            else None
        )

        if not aqua_service_model:
            raise AquaClientError(
                f"Target deployment {model_deployment_id} is not Aqua deployment."
            )

        return AquaDeployment.from_oci_model_deployment(model_deployment, self.region)
    
    def get_shape_config(self, model_id: str) -> List[str]:
        """Gets the shape config of given Aqua model.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.
    
        Returns
        -------
        List:
            A list of allowed instance shapes.
        """
        try:
            oci_model = self.ds_client.get_model(
                model_id
            ).data
        except Exception as se:
            # TODO: adjust error raising
            logger.error(f"Failed to retreive model from the given id {model_id}")
            raise AquaServiceError(opc_request_id=se.request_id, status_code=se.code)
        
        artifact_path = utils.get_artifact_path(
            oci_model.custom_metadata_list
        )

        shape_config = json.loads(
            utils.read_file(
                file_path=f"{artifact_path}/{DEPLOYMENT_CONFIG}",
                auth=self._auth
            ) or UNKNOWN_JSON_STR
        )

        if (
            not shape_config or DEPLOYMENT_SHAPE not in shape_config
            or not shape_config[DEPLOYMENT_SHAPE]
        ):
            raise AquaClientError(
                f"Failed to retrieve shape config from given id {model_id}."
            )
        return shape_config[DEPLOYMENT_SHAPE]
