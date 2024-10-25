#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydantic import Field

from ads.aqua.config.utils.serializer import Serializable


class ModelParamsOverrides(Serializable):
    """Defines overrides for model parameters, including exclusions and additional inclusions."""

    exclude: Optional[List[str]] = Field(default_factory=list)
    include: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        extra = "ignore"


class ModelParamsVersion(Serializable):
    """Handles version-specific model parameter overrides."""

    overrides: Optional[ModelParamsOverrides] = Field(
        default_factory=ModelParamsOverrides
    )

    class Config:
        extra = "ignore"


class ModelParamsContainer(Serializable):
    """Represents a container's model configuration, including tasks, defaults, and versions."""

    name: Optional[str] = None
    default: Optional[Dict[str, Any]] = Field(default_factory=dict)
    versions: Optional[Dict[str, ModelParamsVersion]] = Field(default_factory=dict)

    class Config:
        extra = "ignore"


class InferenceParams(Serializable):
    """Contains inference-related parameters with defaults."""

    class Config:
        extra = "allow"


class InferenceContainer(Serializable):
    """Represents the inference parameters specific to a container."""

    name: Optional[str] = None
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        extra = "ignore"


class ReportParams(Serializable):
    """Handles the report-related parameters."""

    default: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        extra = "ignore"


class InferenceParamsConfig(Serializable):
    """Combines default inference parameters with container-specific configurations."""

    default: Optional[InferenceParams] = Field(default_factory=InferenceParams)
    containers: Optional[List[InferenceContainer]] = Field(default_factory=list)

    def get_merged_params(self, container_name: str) -> InferenceParams:
        """
        Merges default inference params with those specific to the given container.

        Parameters
        ----------
        container_name (str): The name of the container.

        Returns
        -------
        InferenceParams: The merged inference parameters.
        """
        merged_params = self.default.to_dict()
        for containers in self.containers:
            if containers.name.lower() == container_name.lower():
                merged_params.update(containers.params or {})
                break
        return InferenceParams(**merged_params)

    class Config:
        extra = "ignore"


class InferenceModelParamsConfig(Serializable):
    """Encapsulates the model parameters for different containers."""

    default: Optional[Dict[str, Any]] = Field(default_factory=dict)
    containers: Optional[List[ModelParamsContainer]] = Field(default_factory=list)

    def get_merged_model_params(
        self,
        container_name: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Gets the model parameters for a given container, version,
        merged with the defaults.

        Parameters
        ----------
        container_name (str): The name of the container.
        version (Optional[str]): The specific version of the container.

        Returns
        -------
        Dict[str, Any]: The merged model parameters.
        """
        params = deepcopy(self.default)

        for container in self.containers:
            if container.name.lower() == container_name.lower():
                params.update(container.default)

                if version and version in container.versions:
                    version_overrides = container.versions[version].overrides
                    if version_overrides:
                        if version_overrides.include:
                            params.update(version_overrides.include)
                        if version_overrides.exclude:
                            for key in version_overrides.exclude:
                                params.pop(key, None)
                break

        return params

    class Config:
        extra = "ignore"


class ShapeFilterConfig(Serializable):
    """Represents the filtering options for a specific shape."""

    evaluation_container: Optional[List[str]] = Field(default_factory=list)
    evaluation_target: Optional[List[str]] = Field(default_factory=list)

    class Config:
        extra = "ignore"


class ShapeConfig(Serializable):
    """Defines the configuration for a specific shape."""

    name: Optional[str] = None
    ocpu: Optional[int] = None
    memory_in_gbs: Optional[int] = None
    block_storage_size: Optional[int] = None
    filter: Optional[ShapeFilterConfig] = Field(default_factory=ShapeFilterConfig)

    class Config:
        extra = "allow"


class MetricConfig(Serializable):
    """Handles metric configuration including task, key, and additional details."""

    task: Optional[List[str]] = Field(default_factory=list)
    key: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    args: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tags: Optional[List[str]] = Field(default_factory=list)

    class Config:
        extra = "ignore"


class ModelParamsConfig(Serializable):
    """Encapsulates the default model parameters."""

    default: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UIConfig(Serializable):
    model_params: Optional[ModelParamsConfig] = Field(default_factory=ModelParamsConfig)
    shapes: List[ShapeConfig] = Field(default_factory=list)
    metrics: List[MetricConfig] = Field(default_factory=list)

    def search_shapes(
        self,
        evaluation_container: Optional[str] = None,
        evaluation_target: Optional[str] = None,
    ) -> List[ShapeConfig]:
        """
        Searches for shapes that match the given filters.

        Parameters
        ----------
        evaluation_container (Optional[str]): Filter for evaluation_container.
        evaluation_target (Optional[str]): Filter for evaluation_target.

        Returns
        -------
        List[ShapeConfig]: A list of shapes that match the filters.
        """
        return [
            shape
            for shape in self.shapes
            if (
                not evaluation_container
                or evaluation_container in shape.filter.evaluation_container
            )
            and (
                not evaluation_target
                or evaluation_target in shape.filter.evaluation_target
            )
        ]

    class Config:
        extra = "ignore"
        protected_namespaces = ()


class EvaluationServiceConfig(Serializable):
    """
    Root configuration class for evaluation setup including model,
    inference, and shape configurations.
    """

    version: Optional[str] = "1.0"
    kind: Optional[str] = "evaluation_service_config"
    report_params: Optional[ReportParams] = Field(default_factory=ReportParams)
    inference_params: Optional[InferenceParamsConfig] = Field(
        default_factory=InferenceParamsConfig
    )
    inference_model_params: Optional[InferenceModelParamsConfig] = Field(
        default_factory=InferenceModelParamsConfig
    )
    ui_config: Optional[UIConfig] = Field(default_factory=UIConfig)

    def get_merged_inference_params(self, container_name: str) -> InferenceParams:
        """
        Merges default inference params with those specific to the given container.

        Params
        ------
        container_name (str): The name of the container.

        Returns
        -------
        InferenceParams: The merged inference parameters.
        """
        return self.inference_params.get_merged_params(container_name=container_name)

    def get_merged_inference_model_params(
        self,
        container_name: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Gets the model parameters for a given container, version, and task, merged with the defaults.

        Parameters
        ----------
        container_name (str): The name of the container.
        version (Optional[str]): The specific version of the container.

        Returns
        -------
        Dict[str, Any]: The merged model parameters.
        """
        return self.inference_model_params.get_merged_model_params(
            container_name=container_name, version=version
        )

    class Config:
        extra = "ignore"
