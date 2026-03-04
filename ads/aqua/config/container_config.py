#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import json
from typing import Dict, List, Optional

from oci.data_science.models import ContainerSummary
from pydantic import Field

from ads.aqua import logger
from ads.aqua.config.utils.serializer import Serializable
from ads.aqua.constants import (
    SERVICE_MANAGED_CONTAINER_URI_SCHEME,
    UNKNOWN_JSON_LIST,
    UNKNOWN_JSON_STR,
)
from ads.common.extended_enum import ExtendedEnum
from ads.common.utils import UNKNOWN


class Usage(ExtendedEnum):
    INFERENCE = "inference"
    BATCH_INFERENCE = "batch_inference"
    MULTI_MODEL = "multi_model"
    OTHER = "other"


class AquaContainerConfigSpec(Serializable):
    """
    Represents container specification details.

    Attributes
    ----------
    cli_param (Optional[str]): CLI parameter for container configuration.
    server_port (Optional[str]): The server port for the container.
    health_check_port (Optional[str]): The health check port for the container.
    env_vars (Optional[List[Dict]]): Environment variables for the container.
    restricted_params (Optional[List[str]]): Restricted parameters for container configuration.
    """

    cli_param: Optional[str] = Field(
        default=None, description="CLI parameter for container configuration."
    )
    server_port: Optional[str] = Field(
        default=None, description="Server port for the container."
    )
    health_check_port: Optional[str] = Field(
        default=None, description="Health check port for the container."
    )
    env_vars: Optional[List[Dict]] = Field(
        default_factory=list, description="List of environment variables."
    )
    restricted_params: Optional[List[str]] = Field(
        default_factory=list, description="List of restricted parameters."
    )
    evaluation_configuration: Optional[Dict] = Field(
        default_factory=dict, description="Dict of evaluation configuration."
    )

    class Config:
        extra = "allow"


class AquaContainerConfigItem(Serializable):
    """
    Represents an item of the AQUA container configuration.

    Attributes
    ----------
    name (Optional[str]): Name of the container configuration item.
    version (Optional[str]): Version of the container.
    display_name (Optional[str]): Display name for UI.
    family (Optional[str]): Container family or category.
    platforms (Optional[List[str]]): Supported platforms.
    model_formats (Optional[List[str]]): Supported model formats.
    spec (Optional[AquaContainerConfigSpec]): Container specification details.
    """

    name: Optional[str] = Field(
        default=None, description="Name of the container configuration item."
    )
    version: Optional[str] = Field(
        default=None, description="Version of the container."
    )
    display_name: Optional[str] = Field(
        default=None, description="Display name of the container."
    )
    family: Optional[str] = Field(
        default=None, description="Container family or category."
    )
    platforms: Optional[List[str]] = Field(
        default_factory=list, description="Supported platforms."
    )
    model_formats: Optional[List[str]] = Field(
        default_factory=list, description="Supported model formats."
    )
    spec: Optional[AquaContainerConfigSpec] = Field(
        default_factory=AquaContainerConfigSpec,
        description="Detailed container specification.",
    )
    usages: Optional[List[str]] = Field(
        default_factory=list, description="Supported usages."
    )

    class Config:
        extra = "allow"
        protected_namespaces = ()


class AquaContainerConfig(Serializable):
    """
    Represents a configuration of AQUA containers to be returned to the client.

    Attributes
    ----------
    inference (Dict[str, AquaContainerConfigItem]): Inference container configuration items.
    finetune (Dict[str, AquaContainerConfigItem]): Fine-tuning container configuration items.
    evaluate (Dict[str, AquaContainerConfigItem]): Evaluation container configuration items.
    """

    inference: Dict[str, AquaContainerConfigItem] = Field(
        default_factory=dict, description="Inference container configuration items."
    )
    finetune: Dict[str, AquaContainerConfigItem] = Field(
        default_factory=dict, description="Fine-tuning container configuration items."
    )
    evaluate: Dict[str, AquaContainerConfigItem] = Field(
        default_factory=dict, description="Evaluation container configuration items."
    )

    def to_dict(self):
        return {
            "inference": list(self.inference.values()),
            "finetune": list(self.finetune.values()),
            "evaluate": list(self.evaluate.values()),
        }

    @classmethod
    def from_service_config(
        cls, service_containers: List[ContainerSummary]
    ) -> "AquaContainerConfig":
        """
        Creates an AquaContainerConfig instance from a service containers.conf.

        Parameters
        ----------
        service_containers (List[Any]):  List of containers specified in containers.conf
        Returns
        -------
        AquaContainerConfig: The constructed container configuration.
        """

        inference_items: Dict[str, AquaContainerConfigItem] = {}
        finetune_items: Dict[str, AquaContainerConfigItem] = {}
        evaluate_items: Dict[str, AquaContainerConfigItem] = {}
        for container in service_containers:
            if not container.is_latest:
                continue
            container_item = AquaContainerConfigItem(
                name=SERVICE_MANAGED_CONTAINER_URI_SCHEME + container.container_name,
                version=container.tag,
                display_name=container.display_name,
                family=container.family_name,
                usages=container.usages,
                platforms=[],
                model_formats=[],
                spec=None,
            )
            container_type = container.family_name
            usages = [x.upper() for x in container.usages]
            if "INFERENCE" in usages or "MULTI_MODEL" in usages:
                # Extract additional configurations
                additional_configurations = {}
                try:
                    additional_configurations = (
                        container.workload_configuration_details_list[
                            0
                        ].additional_configurations
                    )
                except (AttributeError, IndexError) as ex:
                    logger.debug(
                        "Failed to extract `additional_configurations` for container '%s': %s",
                        getattr(container, "container_name", "<unknown>"),
                        ex,
                    )

                container_item.platforms.append(
                    additional_configurations.get("platforms")
                )
                container_item.model_formats.append(
                    additional_configurations.get("modelFormats")
                )

                # TODO: Remove the else condition once SMC env variable config is updated everywhere
                if additional_configurations.get("env_vars", None):
                    env_vars_dict = json.loads(
                        additional_configurations.get("env_vars") or "{}"
                    )
                    env_vars = [
                        {key: str(value)} for key, value in env_vars_dict.items()
                    ]
                else:
                    config_keys = {
                        "MODEL_DEPLOY_PREDICT_ENDPOINT": UNKNOWN,
                        "MODEL_DEPLOY_HEALTH_ENDPOINT": UNKNOWN,
                        "PORT": UNKNOWN,
                        "HEALTH_CHECK_PORT": UNKNOWN,
                        "VLLM_USE_V1": UNKNOWN,
                    }

                    env_vars = [
                        {key: additional_configurations.get(key, default)}
                        for key, default in config_keys.items()
                        if key in additional_configurations
                    ]

                # Build container spec
                container_item.spec = AquaContainerConfigSpec(
                    cli_param=container.workload_configuration_details_list[0].cmd,
                    server_port=str(
                        container.workload_configuration_details_list[0].server_port
                    ),
                    health_check_port=str(
                        container.workload_configuration_details_list[
                            0
                        ].health_check_port
                    ),
                    env_vars=env_vars,
                    restricted_params=json.loads(
                        container.workload_configuration_details_list[
                            0
                        ].additional_configurations.get("restrictedParams")
                        or UNKNOWN_JSON_LIST
                    ),
                    evaluation_configuration=json.loads(
                        container.workload_configuration_details_list[
                            0
                        ].additional_configurations.get(
                            "evaluationConfiguration", UNKNOWN_JSON_STR
                        )
                    ),
                )

            if "INFERENCE" in usages or "MULTI_MODEL" in usages:
                inference_items[container_type] = container_item
            if "FINE_TUNE" in usages:
                finetune_items[container_type] = container_item
            if "EVALUATION" in usages:
                evaluate_items[container_type] = container_item

        return cls(
            inference=inference_items, finetune=finetune_items, evaluate=evaluate_items
        )
