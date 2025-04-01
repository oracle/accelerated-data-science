#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict, List, Optional

from pydantic import Field

from ads.aqua.common.entities import ContainerSpec
from ads.aqua.config.utils.serializer import Serializable


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
    def from_container_index_json(
        cls,
        config: Dict,
        enable_spec: Optional[bool] = False,
    ) -> "AquaContainerConfig":
        """
        Creates an AquaContainerConfig instance from a container index JSON.

        Parameters
        ----------
        config (Optional[Dict]): The container index JSON.
        enable_spec (Optional[bool]): If True, fetch container specification details.

        Returns
        -------
        AquaContainerConfig: The constructed container configuration.
        """
        # TODO: Return this logic back if necessary in the next iteraion.
        # if not config:
        #     config = get_container_config()

        inference_items: Dict[str, AquaContainerConfigItem] = {}
        finetune_items: Dict[str, AquaContainerConfigItem] = {}
        evaluate_items: Dict[str, AquaContainerConfigItem] = {}

        for container_type, containers in config.items():
            if isinstance(containers, list):
                for container in containers:
                    platforms = container.get("platforms", [])
                    model_formats = container.get("modelFormats", [])
                    usages = container.get("usages", [])
                    container_spec = (
                        config.get(ContainerSpec.CONTAINER_SPEC, {}).get(
                            container_type, {}
                        )
                        if enable_spec
                        else None
                    )
                    container_item = AquaContainerConfigItem(
                        name=container.get("name", ""),
                        version=container.get("version", ""),
                        display_name=container.get(
                            "displayName", container.get("version", "")
                        ),
                        family=container_type,
                        platforms=platforms,
                        model_formats=model_formats,
                        usages=usages,
                        spec=(
                            AquaContainerConfigSpec(
                                cli_param=container_spec.get(
                                    ContainerSpec.CLI_PARM, ""
                                ),
                                server_port=container_spec.get(
                                    ContainerSpec.SERVER_PORT, ""
                                ),
                                health_check_port=container_spec.get(
                                    ContainerSpec.HEALTH_CHECK_PORT, ""
                                ),
                                env_vars=container_spec.get(ContainerSpec.ENV_VARS, []),
                                restricted_params=container_spec.get(
                                    ContainerSpec.RESTRICTED_PARAMS, []
                                ),
                            )
                            if container_spec
                            else None
                        ),
                    )
                    if container.get("type") == "inference":
                        inference_items[container_type] = container_item
                    elif (
                        container.get("type") == "fine-tune"
                        or container_type == "odsc-llm-fine-tuning"
                    ):
                        finetune_items[container_type] = container_item
                    elif (
                        container.get("type") == "evaluate"
                        or container_type == "odsc-llm-evaluate"
                    ):
                        evaluate_items[container_type] = container_item

        return cls(
            inference=inference_items, finetune=finetune_items, evaluate=evaluate_items
        )
