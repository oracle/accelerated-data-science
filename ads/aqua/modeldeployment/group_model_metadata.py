#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List, Optional

from pydantic import BaseModel, Field

from ads.aqua.config.utils.serializer import Serializable

__all__ = ["GroupModelDeploymentMetadata"]


class FineTunedModelSpec(BaseModel):
    """
    Represents a fine-tuned model associated with a base model.

    Attributes
    ----------
    model_path : str
        Object Storage path to the fine-tuned model artifacts.
    model_name : str
        Unique name for the fine-tuned model (used for inference routing).
    """

    model_path: str = Field(..., description="OCI path to the fine-tuned model.")
    model_name: str = Field(..., description="Name assigned to the fine-tuned model.")


class BaseModelSpec(BaseModel):
    """
    Defines configuration for a single base model in multi-model deployment.

    Attributes
    ----------
    model_path : str
        Path to the model in OCI Object Storage.
    model_task : str
        Model task type (e.g., text-generation, image-to-text).
    params : str
        Additional vLLM launch parameters for this model (e.g. parallelism, max context).
    fine_tuned_weights : List[FineTunedModel], optional
        List of associated fine-tuned models.
    """

    model_path: str = Field(..., description="Path to the base model.")
    model_task: str = Field(..., description="Task type the model is intended for.")
    params: str = Field(..., description="Startup parameters passed to vLLM.")
    fine_tuned_weights: Optional[List[FineTunedModelSpec]] = Field(
        default_factory=list,
        description="Optional list of fine-tuned model variants associated with this base model.",
    )


class GroupModelDeploymentMetadata(Serializable):
    """
    Schema representing the metadata passed via MULTI_MODEL_CONFIG for multi-model deployments.

    Attributes
    ----------
    models : List[BaseModelConfig]
        List of base models (with optional fine-tuned weights) to be served.
    """

    models: List[BaseModelSpec] = Field(
        ..., description="List of models in the multi-model deployment."
    )
