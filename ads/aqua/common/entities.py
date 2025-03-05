#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import re
from typing import Dict, Optional

from pydantic import Field, model_validator

from ads.aqua import logger
from ads.aqua.config.utils.serializer import Serializable


class ContainerSpec:
    """
    Class to hold to hold keys within the container spec.
    """

    CONTAINER_SPEC = "containerSpec"
    CLI_PARM = "cliParam"
    SERVER_PORT = "serverPort"
    HEALTH_CHECK_PORT = "healthCheckPort"
    ENV_VARS = "envVars"
    RESTRICTED_PARAMS = "restrictedParams"
    EVALUATION_CONFIGURATION = "evaluationConfiguration"


class GPUSpecs(Serializable):
    """
    Represents the GPU specifications for a compute instance.
    """

    gpu_memory_in_gbs: Optional[int] = Field(
        default=None, description="The amount of GPU memory available (in GB)."
    )
    gpu_count: Optional[int] = Field(
        default=None, description="The number of GPUs available."
    )
    gpu_type: Optional[str] = Field(
        default=None, description="The type of GPU (e.g., 'V100, A100, H100')."
    )


class GPUShapesIndex(Serializable):
    """
    Represents the index of GPU shapes.

    Attributes
    ----------
    shapes (Dict[str, GPUSpecs]): A mapping of compute shape names to their GPU specifications.
    """

    shapes: Dict[str, GPUSpecs] = Field(
        ..., description="Mapping of shape names to GPU specifications."
    )


class ComputeShapeSummary(Serializable):
    """
    Represents the specifications of a compute instance's shape.
    """

    core_count: Optional[int] = Field(
        default=None, description="The number of CPU cores available."
    )
    memory_in_gbs: Optional[int] = Field(
        default=None, description="The amount of memory (in GB) available."
    )
    name: Optional[str] = Field(
        default=None, description="The name identifier of the compute shape."
    )
    shape_series: Optional[str] = Field(
        default=None, description="The series or category of the compute shape."
    )
    gpu_specs: Optional[GPUSpecs] = Field(
        default=None,
        description="The GPU specifications associated with the compute shape.",
    )

    @model_validator(mode="after")
    @classmethod
    def set_gpu_specs(cls, model: "ComputeShapeSummary") -> "ComputeShapeSummary":
        """
        Validates and populates GPU specifications if the shape_series indicates a GPU-based shape.

        - If the shape_series contains "GPU", the validator first checks if the shape name exists
          in the GPU_SPECS dictionary. If found, it creates a GPUSpecs instance with the corresponding data.
        - If the shape is not found in the GPU_SPECS, it attempts to extract the GPU count from the shape name
          using a regex pattern (looking for a number following a dot at the end of the name).

        The information about shapes is taken from: https://docs.oracle.com/en-us/iaas/data-science/using/supported-shapes.htm

        Returns:
            ComputeShapeSummary: The updated instance with gpu_specs populated if applicable.
        """
        try:
            if (
                model.shape_series
                and "GPU" in model.shape_series.upper()
                and model.name
                and not model.gpu_specs
            ):
                # Try to extract gpu_count from the shape name using a regex (e.g., "VM.GPU3.2" -> gpu_count=2)
                match = re.search(r"\.(\d+)$", model.name)
                if match:
                    gpu_count = int(match.group(1))
                    model.gpu_specs = GPUSpecs(gpu_count=gpu_count)
        except Exception as err:
            logger.debug(
                f"Error occurred in attempt to extract GPU specification for the f{model.name}. "
                f"Details: {err}"
            )
        return model


class AquaMultiModelRef(Serializable):
    """
    Lightweight model descriptor used for multi-model deployment.

    This class only contains essential details
    required to fetch complete model metadata and deploy models.

    Attributes
    ----------
    model_id : str
        The unique identifier of the model.
    model_name : Optional[str]
        The name of the model.
    gpu_count : Optional[int]
        Number of GPUs required for deployment.
    env_var : Optional[Dict[str, Any]]
        Optional environment variables to override during deployment.
    """

    model_id: str = Field(..., description="The model OCID to deploy.")
    model_name: Optional[str] = Field(None, description="The name of model.")
    gpu_count: Optional[int] = Field(
        None, description="The gpu count allocation for the model."
    )
    env_var: Optional[dict] = Field(
        default_factory=dict, description="The environment variables of the model."
    )

    class Config:
        extra = "ignore"
        protected_namespaces = ()
