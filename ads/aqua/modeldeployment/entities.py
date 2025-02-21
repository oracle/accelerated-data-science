#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict, List, Optional, Union

from oci.data_science.models import ModelDeployment, ModelDeploymentSummary
from pydantic import BaseModel, Field, model_validator

from ads.aqua import logger
from ads.aqua.common.entities import AquaMultiModelRef, ShapeInfo
from ads.aqua.common.enums import Tags
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


class ShapeInfoConfig(Serializable):
    """Describes how many memory and cpu to this model for specific shape.

    Attributes:
        memory_in_gbs (int, optional): The number of memory in gbs to this model of the shape.
        ocpu (int, optional): The number of ocpus to this model of the shape.
    """

    memory_in_gbs: Optional[int] = Field(
        default_factory=int,
        description="The number of memory in gbs to this model of the shape.",
    )
    ocpu: Optional[int] = Field(
        default_factory=int,
        description="The number of ocpus to this model of the shape.",
    )

    class Config:
        extra = "allow"


class DeploymentShapeInfo(Serializable):
    """Describes the shape information to this model for specific shape.

    Attributes:
        configs (List[ShapeInfoConfig], optional): A list of memory and cpu number details to this model of the shape.
        type (str, optional): The type of the shape.
    """

    configs: Optional[List[ShapeInfoConfig]] = Field(
        default_factory=list,
        description="A list of memory and cpu number details to this model of the shape.",
    )
    type: Optional[str] = Field(
        default_factory=str, description="The type of the shape."
    )

    class Config:
        extra = "allow"


class MultiModelConfig(Serializable):
    """Describes how many GPUs and the parameters of specific shape for multi model deployment.

    Attributes:
        gpu_count (int, optional): Number of GPUs count to this model of this shape.
        parameters (Dict[str, str], optional): A dictionary of parameters (e.g., VLLM_PARAMS) to
            configure the behavior of a particular GPU shape.
    """

    gpu_count: Optional[int] = Field(
        default_factory=int, description="The number of GPUs allocated to the model."
    )
    parameters: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Key-value pairs for GPU shape parameters (e.g., VLLM_PARAMS).",
    )

    class Config:
        extra = "allow"


class ConfigurationItem(Serializable):
    """Holds key-value parameter pairs for a specific GPU or CPU shape.

    Attributes:
        parameters (Dict[str, str], optional): A dictionary of parameters (e.g., VLLM_PARAMS) to
            configure the behavior of a particular GPU shape.
        multi_model_deployment (List[MultiModelConfig], optional): A list of multi model configuration details.
        shape_info (DeploymentShapeInfo, optional): The shape information to this model for specific CPU shape.
    """

    parameters: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Key-value pairs for shape parameters.",
    )
    multi_model_deployment: Optional[List[MultiModelConfig]] = Field(
        default_factory=list, description="A list of multi model configuration details."
    )
    shape_info: Optional[DeploymentShapeInfo] = Field(
        default_factory=DeploymentShapeInfo,
        description="The shape information to this model for specific shape",
    )

    class Config:
        extra = "allow"


class AquaDeploymentConfig(Serializable):
    """Represents multi model's shape list and detailed configuration.

    Attributes:
        shape (List[str], optional): A list of shape names (e.g., BM.GPU.A10.4).
        configuration (Dict[str, ConfigurationItem], optional): Maps each shape to its configuration details.
    """

    shape: Optional[List[str]] = Field(
        default_factory=list, description="List of supported shapes for the model."
    )
    configuration: Optional[Dict[str, ConfigurationItem]] = Field(
        default_factory=dict, description="Configuration details keyed by shape."
    )

    class Config:
        extra = "allow"


class GPUModelAllocation(Serializable):
    """Describes how many GPUs are allocated to a particular model.

    Attributes:
        ocid (str, optional): The unique identifier of the model.
        gpu_count (int, optional): Number of GPUs allocated to this model.
    """

    ocid: Optional[str] = Field(
        default_factory=str, description="The unique model OCID."
    )
    gpu_count: Optional[int] = Field(
        default_factory=int, description="The number of GPUs allocated to the model."
    )

    class Config:
        extra = "allow"


class GPUShapeAllocation(Serializable):
    """
    Allocation details for a specific GPU shape.

    Attributes:
        models (List[GPUModelAllocation], optional): List of model GPU allocations for this shape.
        total_gpus_available (int, optional): The total number of GPUs available for this shape.
    """

    models: Optional[List[GPUModelAllocation]] = Field(
        default_factory=list, description="List of model allocations for this shape."
    )
    total_gpus_available: Optional[int] = Field(
        default_factory=int, description="Total GPUs available for this shape."
    )

    class Config:
        extra = "allow"

class ConfigValidationError(Exception):
    """Exception raised for config validation."""

    def __init__(
        self,
        message: str = """Validation failed: The provided model group configuration is incompatible with the selected instance shape.
        Please verify the GPU count per model and ensure multi-model deployment is supported for the chosen instance shape.""",
    ):
        super().__init__(
            message
        )

class ModelDeploymentConfigSummary(Serializable):
    """Top-level configuration model for OCI-based deployments.

    Attributes:
        deployment_config (Dict[str, AquaDeploymentConfig], optional): Deployment configurations
            keyed by model OCID.
        gpu_allocation (Dict[str, GPUShapeAllocation], optional): GPU allocations keyed by GPU shape.
        error_message (str, optional): Error message if GPU allocation is not possible.
    """

    deployment_config: Optional[Dict[str, AquaDeploymentConfig]] = Field(
        default_factory=dict,
        description=(
            "Deployment configuration details for each model, including supported shapes "
            "and shape-specific parameters."
        ),
    )
    gpu_allocation: Optional[Dict[str, GPUShapeAllocation]] = Field(
        default_factory=dict,
        description=(
            "Details on how GPUs are allocated per shape, including the total "
            "GPUs available for each shape."
        ),
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if GPU allocation is not possible."
    )

    class Config:
        extra = "allow"


class CreateModelDeploymentDetails(BaseModel):
    """Class for creating Aqua model deployments."""

    instance_shape: str = Field(
        ..., description="The instance shape used for deployment."
    )
    display_name: str = Field(..., description="The name of the model deployment.")
    compartment_id: Optional[str] = Field(None, description="The compartment OCID.")
    project_id: Optional[str] = Field(None, description="The project OCID.")
    description: Optional[str] = Field(
        None, description="The description of the deployment."
    )
    model_id: Optional[str] = Field(None, description="The model OCID to deploy.")
    models: Optional[List[AquaMultiModelRef]] = Field(
        None, description="List of models for multimodel deployment."
    )
    instance_count: int = Field(
        None, description="Number of instances used for deployment."
    )
    log_group_id: Optional[str] = Field(
        None, description="OCI logging group ID for logs."
    )
    access_log_id: Optional[str] = Field(
        None,
        description="OCID for access logs. "
        "https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm",
    )
    predict_log_id: Optional[str] = Field(
        None,
        description="OCID for prediction logs."
        "https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm",
    )
    bandwidth_mbps: Optional[int] = Field(
        None, description="Bandwidth limit on the load balancer in Mbps."
    )
    web_concurrency: Optional[int] = Field(
        None, description="Number of worker processes/threads for handling requests."
    )
    server_port: Optional[int] = Field(
        None, description="Server port for the Docker container image."
    )
    health_check_port: Optional[int] = Field(
        None, description="Health check port for the Docker container image."
    )
    env_var: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Environment variables for deployment."
    )
    container_family: Optional[str] = Field(
        None, description="Image family of the model deployment container runtime."
    )
    memory_in_gbs: Optional[float] = Field(
        None, description="Memory (in GB) for the selected shape."
    )
    ocpus: Optional[float] = Field(
        None, description="OCPU count for the selected shape."
    )
    model_file: Optional[str] = Field(
        None, description="File used for model deployment."
    )
    private_endpoint_id: Optional[str] = Field(
        None, description="Private endpoint ID for model deployment."
    )
    container_image_uri: Optional[str] = Field(
        None,
        description="Image URI for model deployment container runtime "
        "(ignored for service-managed containers). "
        "Required parameter for BYOC based deployments if this parameter was not set during "
        "model registration.",
    )
    cmd_var: Optional[List[str]] = Field(
        None, description="Command variables for the container runtime."
    )
    freeform_tags: Optional[Dict] = Field(
        None, description="Freeform tags for model deployment."
    )
    defined_tags: Optional[Dict] = Field(
        None, description="Defined tags for model deployment."
    )

    @model_validator(mode="before")
    @classmethod
    def validate(cls, values: Any) -> Any:
        """Ensures exactly one of `model_id` or `models` is provided."""
        model_id = values.get("model_id")
        models = values.get("models")
        if bool(model_id) == bool(models):  # Both set or both unset
            raise ValueError(
                "Exactly one of `model_id` or `models` must be provided to create a model deployment."
            )
        return values

    def validate_multimodel_deployment_feasibility(self, models_config_summary: ModelDeploymentConfigSummary):
        """
        Validates whether the user input of a model group (List[AquaMultiModelRef], 1+ model(s) with a specified gpu count per model)
        is feasible for a multi model deployment on the user's selected shape (instance_shape)

        Validation Criteria:
            - GPU Capacity: Ensures that the total number of GPUs requested by all models in the group does not exceed the GPU capacity of the selected instance shape.  
            - Verifies that all models in the group are compatible with the selected instance shape.
            - Ensures that each model’s GPU allocation, as specified by the user, matches the requirements in the model's deployment configuration.
            - Confirms that the selected instance shape supports multi-model deployment.
            - Requires user input for the model group to be considered a valid multi-model deployment.


        Parameters
        ----------
        models_config_summary : ModelDeploymentConfigSummary, optional
            An instance of ModelDeploymentConfigSummary containing all required
            fields (GPU Allocation, Deployment Configuration) for creating a multi model deployment via Aqua.

        Raises
        -------
        ConfigValidationError:
            When the deployment is NOT a multi model deployment
            When assigned GPU Allocations per model are NOT within the number of GPUs available in the instance shape
            When all models in model group can NOT be deployed on the instance shape with the selected GPU count
        """
        if not self.models:
            logger.error(
                "User defined model group (List[AquaMultiModelRef]) is None."
            )
            raise ConfigValidationError("Multi-model deployment requires at least one model, but none were provided. Please add one or more models to the model group to proceed.")

        selected_shape = self.instance_shape

        if selected_shape not in models_config_summary.gpu_allocation:
            logger.error(
                    f"The model group is not compatible with the selected instance shape {selected_shape}"
                )
            raise ConfigValidationError(f"The model group is not compatible with the selected instance shape '{selected_shape}'. Select a different instance shape.")

        total_available_gpus = models_config_summary.gpu_allocation[selected_shape].total_gpus_available

        model_deployment_config = models_config_summary.deployment_config

        required_model_keys = [model.model_id for model in self.models]
        missing_model_keys = required_model_keys - model_deployment_config.keys()

        if len(missing_model_keys) > 0:
            logger.error(
                    f"Missing the following model entry with key {missing_model_keys} in ModelDeploymentConfigSummary"
                )
            raise ConfigValidationError("One or more selected models are missing from the configuration, preventing validation for deployment on the given shape.")

        sum_model_gpus = 0

        for model in self.models:
            sum_model_gpus += model.gpu_count

            aqua_deployment_config = model_deployment_config[model.model_id]

            if selected_shape not in aqua_deployment_config.shape:
                logger.error(
                    f"Model with OCID {model.model_id} in the model group is not compatible with the selected instance shape: {selected_shape}"
                )
                raise ConfigValidationError(
                    "Select a different instance shape. One or more models in the group are incompatible with the selected instance shape."
                )


            multi_model_configs = aqua_deployment_config.configuration.get(
                selected_shape, ConfigurationItem()
                ).multi_model_deployment

            valid_gpu_configurations = [gpu_shape_config.gpu_count for gpu_shape_config in multi_model_configs]
            if model.gpu_count not in valid_gpu_configurations:
                valid_gpu_str = ", ".join(map(str, valid_gpu_configurations))
                logger.error(
                    f"Model {model.model_id} allocated {model.gpu_count} GPUs by user, but its deployment configuration requires either {valid_gpu_str} GPUs."
                )
                raise ConfigValidationError(
                    "Change the GPU count for one or more models in the model group. Adjust GPU allocations per model or choose a larger instance shape."
                )

        if sum_model_gpus > total_available_gpus:
            logger.error(
                f"Selected shape {selected_shape} has {total_available_gpus} GPUs while model group has {sum_model_gpus} GPUs."
            )
            raise ConfigValidationError(
                "Total requested GPU count exceeds the available GPU capacity for the selected instance shape. Adjust GPU allocations per model or choose a larger instance shape."
            )

    class Config:
        extra = "ignore"
