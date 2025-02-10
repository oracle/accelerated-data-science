#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict, List, Optional, Union

from oci.data_science.models import ModelDeployment, ModelDeploymentSummary
from pydantic import BaseModel, Field, model_validator

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

    class Config:
        extra = "ignore"


class AquaDeploymentDetail(AquaDeployment, DataClassSerializable):
    """Represents a details of Aqua deployment."""

    log_group: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)
    log: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)

    class Config:
        extra = "ignore"


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
        default_factory=list, description="Command variables for the container runtime."
    )
    freeform_tags: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Freeform tags for model deployment."
    )
    defined_tags: Optional[Dict[str, Dict[str, str]]] = Field(
        default_factory=dict, description="Defined tags for model deployment."
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

    class Config:
        extra = "ignore"
