#!/usr/bin/env python

# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from typing import Any, Dict, List, Optional

from oci.data_science.models import ContainerSummary
from pydantic import Field

from ads.aqua.config.utils.serializer import Serializable

DEFAULT_EVALUATION_CONTAINER = "odsc-llm-evaluate"


class ShapeFilterConfig(Serializable):
    """Represents the filtering options for a specific shape."""

    evaluation_container: Optional[List[str]] = Field(default_factory=list)
    evaluation_target: Optional[List[str]] = Field(default_factory=list)

    class Config:
        extra = "allow"


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
        extra = "allow"


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
        extra = "allow"
        protected_namespaces = ()


class EvaluationServiceConfig(Serializable):
    """
    Root configuration class for evaluation setup including model,
    inference, and shape configurations.
    """

    version: Optional[str] = "1.0"
    kind: Optional[str] = "evaluation_service_config"
    ui_config: Optional[UIConfig] = Field(default_factory=UIConfig)

    class Config:
        extra = "allow"

    @staticmethod
    def from_oci_container_config(
        oci_container_config: ContainerSummary,
    ) -> "EvaluationServiceConfig":
        """
        Returns EvaluationServiceConfig for given oci_container_config
        """

        shapes = json.loads(
            oci_container_config.workload_configuration_details_list[
                0
            ].use_case_configuration.additional_configurations.get("shapes")
        )
        metrics = json.loads(
            oci_container_config.workload_configuration_details_list[
                0
            ].use_case_configuration.additional_configurations.get("metrics")
        )
        model_params = ModelParamsConfig(
            default={
                "model": "odsc-llm",
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "stop": [],
            }
        )
        return EvaluationServiceConfig(
            ui_config=UIConfig(
                model_params=model_params, shapes=shapes, metrics=metrics
            )
        )
