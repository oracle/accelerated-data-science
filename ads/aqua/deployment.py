#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Union

import requests
from oci.data_science.models import ModelDeployment, ModelDeploymentSummary

from ads.aqua.base import AquaApp
from ads.aqua.exception import AquaRuntimeError, AquaValueError
from ads.aqua.model import AquaModelApp, Tags
from ads.aqua.utils import (
    UNKNOWN,
    MODEL_BY_REFERENCE_OSS_PATH_KEY,
    load_config,
    get_container_image,
    get_base_model_from_tags,
    get_resource_name,
)
from ads.aqua.finetune import FineTuneCustomMetadata
from ads.aqua.data import AquaResourceIdentifier
from ads.common.utils import get_console_link, get_log_links
from ads.common.auth import default_signer
from ads.model.deployment import (
    ModelDeployment,
    ModelDeploymentContainerRuntime,
    ModelDeploymentInfrastructure,
    ModelDeploymentMode,
)
from ads.common.serializer import DataClassSerializable
from ads.config import (
    AQUA_MODEL_DEPLOYMENT_CONFIG,
    COMPARTMENT_OCID,
    AQUA_CONFIG_FOLDER,
    AQUA_MODEL_DEPLOYMENT_FOLDER,
    AQUA_CONTAINER_INDEX_CONFIG,
    AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME,
    AQUA_SERVED_MODEL_NAME,
)
from ads.common.object_storage_details import ObjectStorageDetails


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
    lifecycle_details: str = None
    shape_info: field(default_factory=ShapeInfo) = None
    tags: dict = None

    @classmethod
    def from_oci_model_deployment(
        cls,
        oci_model_deployment: Union[ModelDeploymentSummary, ModelDeployment],
        region: str,
    ) -> "AquaDeployment":
        """Converts oci model deployment response to AquaDeployment instance.

        Parameters
        ----------
        oci_model_deployment: Union[ModelDeploymentSummary, ModelDeployment]
            The instance of either oci.data_science.models.ModelDeployment or
            oci.data_science.models.ModelDeploymentSummary class.
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
            lifecycle_details=getattr(
                oci_model_deployment, "lifecycle_details", UNKNOWN
            ),
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


@dataclass(repr=False)
class AquaDeploymentDetail(AquaDeployment, DataClassSerializable):
    """Represents a details of Aqua deployment."""

    log_group: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    log: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)


class AquaDeploymentApp(AquaApp):
    """Provides a suite of APIs to interact with Aqua model deployments within the Oracle
    Cloud Infrastructure Data Science service, serving as an interface for deploying
    machine learning models.


    Methods
    -------
    create(model_id: str, instance_shape: str, display_name: str,...) -> AquaDeployment
        Creates a model deployment for Aqua Model.
    get(model_deployment_id: str) -> AquaDeployment:
        Retrieves details of an Aqua model deployment by its unique identifier.
    list(**kwargs) -> List[AquaModelSummary]:
        Lists all Aqua deployments within a specified compartment and/or project.
    get_deployment_config(self, model_id: str) -> Dict:
        Gets the deployment config of given Aqua model.

    Note:
        Use `ads aqua deployment <method_name> --help` to get more details on the parameters available.
        This class is designed to work within the Oracle Cloud Infrastructure
        and requires proper configuration and authentication set up to interact
        with OCI services.
    """

    def create(
        self,
        model_id: str,
        instance_shape: str,
        display_name: str,
        instance_count: int = None,
        log_group_id: str = None,
        access_log_id: str = None,
        predict_log_id: str = None,
        compartment_id: str = None,
        project_id: str = None,
        description: str = None,
        bandwidth_mbps: int = None,
        web_concurrency: int = None,
        server_port: int = 8080,
        health_check_port: int = 8080,
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
        display_name: str
            The name of model deployment.
        description: str
            The description of the deployment.
        instance_count: (int, optional). Defaults to 1.
            The number of instance used for deployment.
        instance_shape: (str).
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
        # todo: revisit error handling and pull deployment image info from config
        # if not AQUA_MODEL_DEPLOYMENT_IMAGE:
        #     raise AquaValueError(
        #         f"AQUA_MODEL_DEPLOYMENT_IMAGE must be available in environment variables to "
        #         f"continue with Aqua model deployment."
        #     )

        # todo: for fine tuned models, skip model creation.
        # Create a model catalog entry in the user compartment
        aqua_model = AquaModelApp().create(
            model_id=model_id, comparment_id=compartment_id, project_id=project_id
        )

        tags = {}
        for tag in [
            Tags.AQUA_SERVICE_MODEL_TAG.value,
            Tags.AQUA_FINE_TUNED_MODEL_TAG.value,
            Tags.AQUA_TAG.value,
        ]:
            if tag in aqua_model.freeform_tags:
                tags[tag] = aqua_model.freeform_tags[tag]

        deployment_config = load_config(
            AQUA_CONFIG_FOLDER,
            config_file_name=AQUA_MODEL_DEPLOYMENT_CONFIG,
        )
        # todo: revisit after create FT model is implemented

        is_fine_tuned_model = (
            Tags.AQUA_FINE_TUNED_MODEL_TAG.value in aqua_model.freeform_tags
        )

        if is_fine_tuned_model:
            _, model_name = get_base_model_from_tags(aqua_model.freeform_tags)
        else:
            model_name = aqua_model.display_name

        if model_name not in deployment_config:
            raise AquaValueError(
                f"{AQUA_MODEL_DEPLOYMENT_CONFIG} does not have config details for model: {model_name}"
            )

        shape_list = deployment_config[model_name]["shape"]
        if instance_shape not in shape_list:
            raise AquaValueError(
                f"{instance_shape} is not supported by the model {aqua_model.id}. Available shapes are: {shape_list}"
            )

        # set up env vars
        if not env_var:
            env_var = dict()

        try:
            model_path_prefix = aqua_model.custom_metadata_list.get(
                MODEL_BY_REFERENCE_OSS_PATH_KEY
            ).value.rstrip("/")
        except ValueError:
            raise AquaValueError(
                f"{MODEL_BY_REFERENCE_OSS_PATH_KEY} key is not available in the custom metadata field."
            )

        # todo: remove this after absolute path is removed from env var
        if ObjectStorageDetails.is_oci_path(model_path_prefix):
            os_path = ObjectStorageDetails.from_path(model_path_prefix)
            model_path_prefix = os_path.filepath.rstrip("/")

        env_var.update({"BASE_MODEL": f"{model_path_prefix}"})
        env_var.update({"PARAMS": f"--served-model-name {AQUA_SERVED_MODEL_NAME}"})
        env_var.update({"MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/completions"})
        env_var.update({"MODEL_DEPLOY_ENABLE_STREAMING": "true"})

        if is_fine_tuned_model:
            try:
                fine_tune_output_path = aqua_model.custom_metadata_list.get(
                    FineTuneCustomMetadata.FINE_TUNE_OUTPUT_PATH.value
                ).value.rstrip("/")
            except ValueError:
                raise AquaValueError(
                    f"{FineTuneCustomMetadata.FINE_TUNE_OUTPUT_PATH.value} key is not available in the custom metadata field."
                )
            if ObjectStorageDetails.is_oci_path(fine_tune_output_path):
                os_path = ObjectStorageDetails.from_path(fine_tune_output_path)
                fine_tune_output_path = os_path.filepath.rstrip("/")

            env_var.update({"FT_MODEL": f"{fine_tune_output_path}"})

        logging.info(f"Env vars used for deploying {aqua_model.id} :{env_var}")

        try:
            container_type_key = aqua_model.custom_metadata_list.get(
                AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME
            ).value
        except ValueError:
            raise AquaValueError(
                f"{AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME} key is not available in the custom metadata field for model {aqua_model.id}"
            )

        # fetch image name from config
        container_image = get_container_image(
            config_file_name=AQUA_CONTAINER_INDEX_CONFIG,
            container_type=container_type_key,
        )
        logging.info(
            f"Aqua Image used for deploying {aqua_model.id} : {container_image}"
        )

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
            .with_image(container_image)
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
            oci_aqua = (
                (
                    Tags.AQUA_TAG.value in model_deployment.freeform_tags
                    or Tags.AQUA_TAG.value.lower() in model_deployment.freeform_tags
                )
                if model_deployment.freeform_tags
                else False
            )

            if oci_aqua:
                results.append(
                    AquaDeployment.from_oci_model_deployment(
                        model_deployment, self.region
                    )
                )

        return results

    def get(self, model_deployment_id: str, **kwargs) -> "AquaDeploymentDetail":
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
        AquaDeploymentDetail:
            The instance of the Aqua model deployment details.
        """
        model_deployment = self.ds_client.get_model_deployment(
            model_deployment_id=model_deployment_id, **kwargs
        ).data

        oci_aqua = (
            (
                Tags.AQUA_TAG.value in model_deployment.freeform_tags
                or Tags.AQUA_TAG.value.lower() in model_deployment.freeform_tags
            )
            if model_deployment.freeform_tags
            else False
        )

        if not oci_aqua:
            raise AquaRuntimeError(
                f"Target deployment {model_deployment_id} is not Aqua deployment."
            )

        log_id = ""
        log_group_id = ""
        log_name = ""
        log_group_name = ""

        logs = (
            model_deployment.category_log_details.access
            or model_deployment.category_log_details.predict
        )
        if logs:
            log_id = logs.log_id
            log_group_id = logs.log_group_id
        if log_id:
            log_name = get_resource_name(log_id)
        if log_group_id:
            log_group_name = get_resource_name(log_group_id)

        log_group_url = get_log_links(region=self.region, log_group_id=log_group_id)
        log_url = get_log_links(
            region=self.region, log_group_id=log_group_id, log_id=log_id
        )

        return AquaDeploymentDetail(
            **vars(
                AquaDeployment.from_oci_model_deployment(model_deployment, self.region)
            ),
            log_group=AquaResourceIdentifier(
                log_group_id, log_group_name, log_group_url
            ),
            log=AquaResourceIdentifier(log_id, log_name, log_url),
        )

    def get_deployment_config(self, model_id: str) -> Dict:
        """Gets the deployment config of given Aqua model.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.

        Returns
        -------
        Dict:
            A dict of allowed deployment configs.
        """

        return self.get_config(model_id, AQUA_MODEL_DEPLOYMENT_CONFIG)


@dataclass
class ModelParams:
    max_tokens: int = None
    temperature: float = None
    top_k: float = None
    top_p: float = None
    model: str = None


@dataclass
class MDInferenceResponse(AquaApp):
    """Contains APIs for Aqua Model deployments Inference.

    Attributes
    ----------

    model_params: Dict
    prompt: string

    Methods
    -------
    get_model_deployment_response(self, **kwargs) -> "String"
        Creates an instance of model deployment via Aqua
    """

    prompt: str = None
    model_params: field(default_factory=ModelParams) = None

    def get_model_deployment_response(self, endpoint):
        """
        Returns MD inference response

        Parameters
        ----------
        endpoint: str
            MD predict url
        prompt: str
            User prompt.

        model_params: (Dict, optional)
            Model parameters to be associated with the message.
            Currently supported VLLM+OpenAI parameters.

            --model-params '{
                "max_tokens":500,
                "temperature": 0.5,
                "top_k": 10,
                "top_p": 0.5,
                "model": "/opt/ds/model/deployed_model",
                ...}'

        Returns
        -------
        model_response_content
        """

        params_dict = asdict(self.model_params)
        params_dict = {
            key: value for key, value in params_dict.items() if value is not None
        }
        body = {"prompt": self.prompt, **params_dict}
        request_kwargs = {"json": body, "headers": {"Content-Type": "application/json"}}
        response = requests.post(
            endpoint, auth=default_signer()["signer"], **request_kwargs
        )
        return json.loads(response.content)
