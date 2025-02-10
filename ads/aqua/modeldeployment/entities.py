#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict, List, Optional, Union

from oci.data_science.models import (
    ModelDeployment,
    ModelDeploymentSummary,
)
from pydantic import Field, model_validator

from ads.aqua.common.entities import ShapeInfo
from ads.aqua.common.enums import Tags
from ads.aqua.common.errors import AquaValueError
from ads.aqua.config.utils.serializer import Serializable
from ads.aqua.constants import UNKNOWN, UNKNOWN_DICT
from ads.aqua.data import AquaResourceIdentifier
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link


class ModelParams(Serializable):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[float] = None
    top_p: Optional[float] = None
    model: Optional[str] = None

    class Config:
        extra = "allow"
        protected_namespaces = ()


class AquaDeployment(Serializable):
    """Represents an Aqua Model Deployment"""

    id: Optional[str] = None
    display_name: Optional[str] = None
    aqua_service_model: Optional[bool] = None
    model_id: str = None
    aqua_model_name: Optional[str] = None
    state: Optional[str] = None
    description: Optional[str] = None
    created_on: Optional[str] = None
    created_by: Optional[str] = None
    endpoint: Optional[str] = None
    private_endpoint_id: Optional[str] = None
    console_link: Optional[str] = None
    lifecycle_details: Optional[str] = None
    shape_info: Optional[ShapeInfo] = None
    tags: Optional[dict] = None
    environment_variables: Optional[dict] = None
    cmd: Optional[List[str]] = None

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
        instance_configuration = oci_model_deployment.model_deployment_configuration_details.model_configuration_details.instance_configuration
        instance_shape_config_details = (
            instance_configuration.model_deployment_instance_shape_config_details
        )
        instance_count = oci_model_deployment.model_deployment_configuration_details.model_configuration_details.scaling_policy.instance_count
        environment_variables = oci_model_deployment.model_deployment_configuration_details.environment_configuration_details.environment_variables
        cmd = oci_model_deployment.model_deployment_configuration_details.environment_configuration_details.cmd
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
        model_id = oci_model_deployment._model_deployment_configuration_details.model_configuration_details.model_id
        tags = {}
        tags.update(oci_model_deployment.freeform_tags or UNKNOWN_DICT)
        tags.update(oci_model_deployment.defined_tags or UNKNOWN_DICT)

        aqua_service_model_tag = tags.get(Tags.AQUA_SERVICE_MODEL_TAG, None)
        aqua_model_name = tags.get(Tags.AQUA_MODEL_NAME_TAG, UNKNOWN)
        private_endpoint_id = getattr(
            instance_configuration, "private_endpoint_id", UNKNOWN
        )

        return AquaDeployment(
            id=oci_model_deployment.id,
            model_id=model_id,
            display_name=oci_model_deployment.display_name,
            aqua_service_model=aqua_service_model_tag is not None,
            aqua_model_name=aqua_model_name,
            shape_info=shape_info,
            state=oci_model_deployment.lifecycle_state,
            lifecycle_details=getattr(
                oci_model_deployment, "lifecycle_details", UNKNOWN
            ),
            description=oci_model_deployment.description,
            created_on=str(oci_model_deployment.time_created),
            created_by=oci_model_deployment.created_by,
            endpoint=oci_model_deployment.model_deployment_url,
            private_endpoint_id=private_endpoint_id,
            console_link=get_console_link(
                resource="model-deployments",
                ocid=oci_model_deployment.id,
                region=region,
            ),
            tags=tags,
            environment_variables=environment_variables,
            cmd=cmd,
        )

    class Config:
        extra = "ignore"


class AquaDeploymentDetail(AquaDeployment, DataClassSerializable):
    """Represents a details of Aqua deployment."""

    log_group: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)
    log: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)

    class Config:
        extra = "ignore"


class ModelInfo(Serializable):
    """Class for maintaining details of model to be deployed, usually for multi-model deployment."""

    model_id: str
    gpu_count: Optional[int] = None
    env_var: Optional[dict] = None

    class Config:
        extra = "ignore"


class CreateModelDeploymentDetails(Serializable):
    """Class for creating aqua model deployment.

    Properties
    ----------
    compartment_id: str
        The compartment OCID
    project_id: str
        Target project to list deployments from.
    display_name: str
        The name of model deployment.
    description: str
        The description of the deployment.
    model_id: (str, optional)
        The model OCID to deploy. Either model_id or model_info should be set.
    model_info: (List[ModelInfo], optional)
        The model info to deploy, used for multimodel deployment. Either model_id or model_info should be set.
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
    server_port: (int).
        The server port for docker container image.
    health_check_port: (int).
        The health check port for docker container image.
    env_var : dict, optional
        Environment variable for the deployment, by default None.
    container_family: str
        The image family of model deployment container runtime.
    memory_in_gbs: float
        The memory in gbs for the shape selected.
    ocpus: float
        The ocpu count for the shape selected.
    model_file: str
        The file used for model deployment.
    private_endpoint_id: str
        The private endpoint id of model deployment.
    container_image_uri: str
        The image of model deployment container runtime, ignored for service managed containers.
        Required parameter for BYOC based deployments if this parameter was not set during model registration.
    cmd_var: List[str]
        The cmd of model deployment container runtime.
    freeform_tags: dict
        Freeform tags for the model deployment
    defined_tags: dict
        Defined tags for the model deployment
    """

    instance_shape: str
    display_name: str
    model_id: Optional[str] = None
    model_info: Optional[List[ModelInfo]] = None
    instance_count: Optional[int] = None
    log_group_id: Optional[str] = None
    access_log_id: Optional[str] = None
    predict_log_id: Optional[str] = None
    compartment_id: Optional[str] = None
    project_id: Optional[str] = None
    description: Optional[str] = None
    bandwidth_mbps: Optional[int] = None
    web_concurrency: Optional[int] = None
    server_port: Optional[int] = None
    health_check_port: Optional[int] = None
    env_var: Optional[dict] = None
    container_family: Optional[str] = None
    memory_in_gbs: Optional[float] = None
    ocpus: Optional[float] = None
    model_file: Optional[str] = None
    private_endpoint_id: Optional[str] = None
    container_image_uri: Optional[None] = None
    cmd_var: Optional[List[str]] = None
    freeform_tags: Optional[dict] = None
    defined_tags: Optional[dict] = None

    @model_validator(mode="before")
    @classmethod
    def validate_model_fields(cls, values):
        model_id, model_info = values.get("model_id"), values.get("model_info")
        if bool(model_id) == bool(model_info):  # either both are set or unset
            raise AquaValueError(
                "Exactly one of `model_id` or `model_info` must be set to create a model deployment"
            )
        return values

    class Config:
        extra = "ignore"


class MultiModelConfig(Serializable):
    """Describes how many GPUs and the parameters of specific shape for multi model deployment.

    Attributes:
        gpu_count (int): Number of GPUs count to this model of this shape.
        parameters (Dict[str, str], optional): A dictionary of parameters (e.g., VLLM_PARAMS) to
            configure the behavior of a particular GPU shape.
    """

    gpu_count: int = Field(
        default_factory=int, description="The number of GPUs allocated to the model."
    )
    parameters: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Key-value pairs for GPU shape parameters (e.g., VLLM_PARAMS).",
    )

    class Config:
        extra = "ignore"


class ConfigurationItem(Serializable):
    """Holds key-value parameter pairs for a specific GPU shape.

    Attributes:
        parameters (Dict[str, str], optional): A dictionary of parameters (e.g., VLLM_PARAMS) to
            configure the behavior of a particular GPU shape.
        multi_model_deployment (List[MultiModelConfig], optional): A list of multi model configuration details.
    """

    parameters: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Key-value pairs for GPU shape parameters (e.g., VLLM_PARAMS).",
    )
    multi_model_deployment: Optional[List[MultiModelConfig]] = Field(
        default_factory=list, description="A list of multi model configuration details."
    )

    class Config:
        extra = "ignore"


class ModelDeploymentConfig(Serializable):
    """Represents one model's shape list and detailed configuration.

    Attributes:
        shape (List[str]): A list of shape names (e.g., BM.GPU.A10.4).
        configuration (Dict[str, ConfigurationItem]): Maps each shape to its configuration details.
    """

    shape: List[str] = Field(
        default_factory=list, description="List of supported shapes for the model."
    )
    configuration: Dict[str, ConfigurationItem] = Field(
        default_factory=dict, description="Configuration details keyed by shape."
    )

    class Config:
        extra = "ignore"


class AquaDeploymentConfig(ModelDeploymentConfig):
    """Represents multi model's shape list and detailed configuration.

    Attributes:
        shape (List[str]): A list of shape names (e.g., BM.GPU.A10.4).
        configuration (Dict[str, ConfigurationItem]): Maps each shape to its configuration details.
    """

    configuration: Dict[str, ConfigurationItem] = Field(
        default_factory=dict, description="Configuration details keyed by shape."
    )


class GPUModelAllocation(Serializable):
    """Describes how many GPUs are allocated to a particular model.

    Attributes:
        ocid (str): The unique identifier of the model.
        gpu_count (int): Number of GPUs allocated to this model.
    """

    ocid: str = Field(default_factory=str, description="The unique model OCID.")
    gpu_count: int = Field(
        default_factory=int, description="The number of GPUs allocated to the model."
    )

    class Config:
        extra = "ignore"


class GPUShapeAllocation(Serializable):
    """Allocation details for a specific GPU shape.

    Attributes:
        models (List[GPUModelAllocation]): List of model GPU allocations for this shape.
        total_gpus_available (int): The total number of GPUs available for this shape.
    """

    models: List[GPUModelAllocation] = Field(
        default_factory=list, description="List of model allocations for this shape."
    )
    total_gpus_available: int = Field(
        default_factory=int, description="Total GPUs available for this shape."
    )

    class Config:
        extra = "ignore"


class ModelDeploymentConfigSummary(Serializable):
    """Top-level configuration model for OCI-based deployments.

    Attributes:
        deployment_config (Dict[str, ModelDeploymentConfig]): Deployment configurations
            keyed by model OCID.
        gpu_allocation (Dict[str, GPUShapeAllocation]): GPU allocations keyed by GPU shape.
    """

    deployment_config: Dict[str, ModelDeploymentConfig] = Field(
        default_factory=dict,
        description=(
            "Deployment configuration details for each model, including supported shapes "
            "and shape-specific parameters."
        ),
    )
    gpu_allocation: Dict[str, GPUShapeAllocation] = Field(
        default_factory=dict,
        description=(
            "Details on how GPUs are allocated per shape, including the total "
            "GPUs available for each shape."
        ),
    )

    class Config:
        extra = "ignore"
