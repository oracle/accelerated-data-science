#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import re
from typing import Any, Dict, Optional

from oci.data_science.models import Model
from pydantic import BaseModel, Field, model_validator

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


class ModelConfigResult(BaseModel):
    """
    Represents the result of getting the AQUA model configuration.
    Attributes:
        model_details (Dict[str, Any]): A dictionary containing model details extracted from OCI.
        config (Dict[str, Any]): A dictionary of the loaded configuration.
    """

    config: Optional[Dict[str, Any]] = Field(
        None, description="Loaded configuration dictionary."
    )
    model_details: Optional[Model] = Field(
        None, description="Details of the model from OCI."
    )

    class Config:
        extra = "ignore"
        arbitrary_types_allowed = True
        protected_namespaces = ()


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
        default_factory=dict,
        description="Mapping of shape names to GPU specifications.",
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
    model_task : Optional[str]
        The task that model operates on. Supported tasks are in MultiModelSupportedTaskType
    env_var : Optional[Dict[str, Any]]
        Optional environment variables to override during deployment.
    artifact_location : Optional[str]
        Artifact path of model in the multimodel group.
    fine_tune_weights_location : Optional[str]
        For fine tuned models, the artifact path of the modified model weights
    """

    model_id: str = Field(..., description="The model OCID to deploy.")
    model_name: Optional[str] = Field(None, description="The name of model.")
    gpu_count: Optional[int] = Field(
        None, description="The gpu count allocation for the model."
    )
    model_task: Optional[str] = Field(None, description="The task that model operates on. Supported tasks are in MultiModelSupportedTaskType")
    env_var: Optional[dict] = Field(
        default_factory=dict, description="The environment variables of the model."
    )
    artifact_location: Optional[str] = Field(
        None, description="Artifact path of model in the multimodel group."
    )
    fine_tune_weights_location: Optional[str] = Field(
        None, description="For fine tuned models, the artifact path of the modified model weights"
    )

    class Config:
        extra = "ignore"
        protected_namespaces = ()


class ContainerPath(Serializable):
    """
    Represents a parsed container path, extracting the path, name, and version.

    This model is designed to parse a container path string of the format
    '<image_path>:<version>'. It extracts the following components:
    - `path`: The full path up to the version.
    - `name`: The last segment of the path, representing the image name.
    - `version`: The version number following the final colon.

    Example Usage:
    --------------
    >>> container = ContainerPath(full_path="iad.ocir.io/ociodscdev/odsc-llm-evaluate:0.1.2.9")
    >>> container.path
    'iad.ocir.io/ociodscdev/odsc-llm-evaluate'
    >>> container.name
    'odsc-llm-evaluate'
    >>> container.version
    '0.1.2.9'

    >>> container = ContainerPath(full_path="custom-scheme://path/to/versioned-model:2.5.1")
    >>> container.path
    'custom-scheme://path/to/versioned-model'
    >>> container.name
    'versioned-model'
    >>> container.version
    '2.5.1'

    Attributes
    ----------
    full_path : str
        The complete container path string to be parsed.
    path : Optional[str]
        The full path up to the version (e.g., 'iad.ocir.io/ociodscdev/odsc-llm-evaluate').
    name : Optional[str]
        The image name, which is the last segment of `path` (e.g., 'odsc-llm-evaluate').
    version : Optional[str]
        The version number following the final colon in the path (e.g., '0.1.2.9').

    Methods
    -------
    validate(values: Any) -> Any
        Validates and parses the `full_path`, extracting `path`, `name`, and `version`.
    """

    full_path: str
    path: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate(cls, values: Any) -> Any:
        """
        Validates and parses the full container path, extracting the image path, image name, and version.

        Parameters
        ----------
        values : dict
            The dictionary of values being validated, containing 'full_path'.

        Returns
        -------
        dict
            Updated values dictionary with extracted 'path', 'name', and 'version'.
        """
        full_path = values.get("full_path", "").strip()

        # Regex to parse <image_path>:<version>
        match = re.match(
            r"^(?P<image_path>.+?)(?::(?P<image_version>[\w\.]+))?$", full_path
        )

        if not match:
            raise ValueError(
                "Invalid container path format. Expected format: '<image_path>:<version>'"
            )

        # Extract image_path and version
        values["path"] = match.group("image_path")
        values["version"] = match.group("image_version")

        # Extract image_name as the last segment of image_path
        values["name"] = values["path"].split("/")[-1]

        return values

    class Config:
        extra = "ignore"
        protected_namespaces = ()
