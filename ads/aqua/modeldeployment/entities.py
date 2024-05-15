#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass, field
from typing import Union

from oci.data_science.models import ModelDeployment, ModelDeploymentSummary

from ads.aqua.common.enums import Tags
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


class ContainerSpec:
    """
    Class to hold to hold keys within the container spec.
    """

    CONTAINER_SPEC = "containerSpec"
    CLI_PARM = "cliParam"
    SERVER_PORT = "serverPort"
    HEALTH_CHECK_PORT = "healthCheckPort"
    ENV_VARS = "envVars"


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

        freeform_tags = oci_model_deployment.freeform_tags or UNKNOWN_DICT
        aqua_service_model_tag = freeform_tags.get(Tags.AQUA_SERVICE_MODEL_TAG, None)
        aqua_model_name = freeform_tags.get(Tags.AQUA_MODEL_NAME_TAG, UNKNOWN)

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
            console_link=get_console_link(
                resource="model-deployments",
                ocid=oci_model_deployment.id,
                region=region,
            ),
            tags=freeform_tags,
        )


@dataclass(repr=False)
class AquaDeploymentDetail(AquaDeployment, DataClassSerializable):
    """Represents a details of Aqua deployment."""

    log_group: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    log: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
