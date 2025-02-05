#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from oci.data_science.models import (
    ModelDeployment,
    ModelDeploymentSummary,
)
from pydantic import Field

from ads.aqua.common.enums import Tags
from ads.aqua.config.utils.serializer import Serializable
from ads.aqua.constants import UNKNOWN, UNKNOWN_DICT
from ads.aqua.data import AquaResourceIdentifier
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link


@dataclass
class ModelParams:
    max_tokens: int = None
    temperature: float = None
    top_k: float = None
    top_p: float = None
    model: str = None


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
    aqua_model_name: str = None
    state: str = None
    description: str = None
    created_on: str = None
    created_by: str = None
    endpoint: str = None
    private_endpoint_id: str = None
    console_link: str = None
    lifecycle_details: str = None
    shape_info: Optional[ShapeInfo] = None
    tags: dict = None
    environment_variables: dict = None
    cmd: List[str] = None

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


@dataclass(repr=False)
class AquaDeploymentDetail(AquaDeployment, DataClassSerializable):
    """Represents a details of Aqua deployment."""

    log_group: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    log: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)


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
    """

    parameters: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Key-value pairs for GPU shape parameters (e.g., VLLM_PARAMS).",
    )

    class Config:
        extra = "ignore"


class MultiModelConfigurationItem(ConfigurationItem):
    """Holds a list of multi model configuration.

    Attributes:
        multi_model_deployment (List[MultiModelConfig]): A list of multi model configuration details.
    """

    multi_model_deployment: Optional[List[MultiModelConfig]] = Field(
        default_factory=list, description="A list of multi model configuration details."
    )


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
        configuration (Dict[str, MultiModelConfigurationItem]): Maps each shape to its configuration details.
    """

    configuration: Dict[str, MultiModelConfigurationItem] = Field(
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
