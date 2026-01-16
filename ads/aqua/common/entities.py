#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import re
from typing import Any, Dict, List, Optional, Union

from oci.data_science.models import Model
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class ComputeRank(Serializable):
    """
    Represents the cost and performance rankings for a specific compute shape.
    These rankings help compare different shapes based on their relative pricing
    and computational capabilities.
    """

    cost: Optional[int] = Field(
        None,
        description=(
            "Relative cost ranking of the compute shape. "
            "Value ranges from 10 (most cost-effective) to 100 (most expensive). "
            "Lower values indicate cheaper compute options."
        ),
    )

    performance: Optional[int] = Field(
        None,
        description=(
            "Relative performance ranking of the compute shape. "
            "Value ranges from 10 (lowest performance) to 110 (highest performance). "
            "Higher values indicate better compute performance."
        ),
    )


class GPUSpecs(Serializable):
    """
    Represents the specifications and capabilities of a GPU-enabled compute shape.
    Includes details about GPU and CPU resources, supported quantization formats, and
    relative rankings for cost and performance.
    """

    gpu_count: Optional[int] = Field(
        default=None,
        description="Number of physical GPUs available on the compute shape.",
    )

    gpu_memory_in_gbs: Optional[int] = Field(
        default=None, description="Total GPU memory available in gigabytes (GB)."
    )

    gpu_type: Optional[str] = Field(
        default=None,
        description="Type of GPU and architecture. Example: 'H100', 'GB200'.",
    )

    quantization: Optional[List[str]] = Field(
        default_factory=list,
        description=(
            "List of supported quantization formats for the GPU. "
            "Examples: 'fp16', 'int8', 'bitsandbytes', 'bf16', 'fp4', etc."
        ),
    )

    cpu_count: Optional[int] = Field(
        default=None, description="Number of CPU cores available on the shape."
    )

    cpu_memory_in_gbs: Optional[int] = Field(
        default=None, description="Total CPU memory available in gigabytes (GB)."
    )

    ranking: Optional[ComputeRank] = Field(
        default=None,
        description=(
            "Relative cost and performance rankings of this shape. "
            "Cost is ranked from 10 (least expensive) to 100+ (most expensive), "
            "and performance from 10 (lowest) to 100+ (highest)."
        ),
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
    Represents a compute shape's specification including CPU, memory, and (if applicable) GPU configuration.
    """

    available: Optional[bool] = Field(
        default=False,
        description="True if the shape is available in the user's tenancy/region.",
    )

    core_count: Optional[int] = Field(
        default=None, description="Number of vCPUs available for the compute shape."
    )

    memory_in_gbs: Optional[int] = Field(
        default=None, description="Total CPU memory available for the shape (in GB)."
    )

    name: Optional[str] = Field(
        default=None, description="Name of the compute shape, e.g., 'VM.GPU.A10.2'."
    )

    shape_series: Optional[str] = Field(
        default=None,
        description="Series or family of the shape, e.g., 'GPU', 'Standard'.",
    )

    gpu_specs: Optional[GPUSpecs] = Field(
        default=None, description="GPU configuration for the shape, if applicable."
    )

    @model_validator(mode="after")
    @classmethod
    def populate_gpu_specs(cls, model: "ComputeShapeSummary") -> "ComputeShapeSummary":
        """
        Attempts to populate GPU specs if the shape is GPU-based and no GPU specs are explicitly set.

        Logic:
        - If `shape_series` includes 'GPU' and `gpu_specs` is None:
          - Tries to parse the shape name to extract GPU count (e.g., from 'VM.GPU.A10.2').
          - Fallback is based on suffix numeric group (e.g., '.2' → gpu_count=2).
        - If extraction fails, logs debug-level error but does not raise.

        Returns:
            ComputeShapeSummary: The updated model instance.
        """
        try:
            if (
                model.shape_series
                and "GPU" in model.shape_series.upper()
                and model.name
                and not model.gpu_specs
            ):
                match = re.search(r"\.(\d+)$", model.name)
                if match:
                    gpu_count = int(match.group(1))
                    model.gpu_specs = GPUSpecs(gpu_count=gpu_count)
        except Exception as err:
            logger.debug(
                f"[populate_gpu_specs] Failed to auto-populate GPU specs for shape '{model.name}': {err}"
            )

        return model


class LoraModuleSpec(BaseModel):
    """
    Descriptor for a LoRA (Low-Rank Adaptation) module used in fine-tuning base models.

    This class is used to define a single fine-tuned module that can be loaded during
    multi-model deployment alongside a base model.

    Attributes
    ----------
    model_id : str
        The OCID of the fine-tuned model registered in the OCI Model Catalog.
    model_name : Optional[str]
        The unique name used to route inference requests to this model variant.
    model_path : Optional[str]
        The relative path within the artifact pointing to the LoRA adapter weights.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    model_id: str = Field(
        ...,
        description="OCID of the fine-tuned model (must be registered in the Model Catalog).",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Name assigned to the fine-tuned model for serving (used as inference route).",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Relative path to the LoRA weights inside the model artifact.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_lora_module(cls, data: dict) -> dict:
        """Validates that required structure exists for a LoRA module."""
        if "model_id" not in data or not data["model_id"]:
            raise ValueError("Missing required field: 'model_id' for fine-tuned model.")
        return data


class AquaMultiModelRef(Serializable):
    """
    Lightweight model descriptor used for multi-model deployment.

    This class holds essential details required to fetch model metadata and deploy
    individual models as part of a multi-model deployment group.

    Attributes
    ----------
    model_id : str
        The unique identifier (OCID) of the base model.
    model_name : Optional[str]
        Optional name for the model.
    gpu_count : Optional[int]
        Number of GPUs required to allocate for this model during deployment.
    model_task : Optional[str]
        The machine learning task this model performs (e.g., text-generation, summarization).
        Supported values are listed in `MultiModelSupportedTaskType`.
    env_var : Optional[Dict[str, Any]]
        Optional dictionary of environment variables to inject into the runtime environment
        of the model container.
    params : Optional[Dict[str, Any]]
        Optional dictionary of container-specific inference parameters to override.
        These are typically framework-level flags required by the runtime backend.
        For example, in vLLM containers, valid params may include:
        `--tensor-parallel-size`, `--enforce-eager`, `--max-model-len`, etc.
    artifact_location : Optional[str]
        Relative path or URI of the model artifact inside the multi-model group folder.
    fine_tune_weights : Optional[List[LoraModuleSpec]]
        List of fine-tuned weight artifacts (e.g., LoRA modules) associated with this model.
    """

    model_id: str = Field(..., description="The model OCID to deploy.")
    model_name: Optional[str] = Field(None, description="The name of the model.")
    gpu_count: Optional[int] = Field(
        None, description="The number of GPUs allocated for the model."
    )
    model_task: Optional[str] = Field(
        None,
        description="The task this model performs. See `MultiModelSupportedTaskType` for supported values.",
    )
    env_var: Optional[dict] = Field(
        default_factory=dict,
        description="Environment variables to override during container startup.",
    )
    params: Optional[dict] = Field(
        default=None,
        description=(
            "Framework-specific startup parameters required by the container runtime. "
            "For example, vLLM models may use flags like `--tensor-parallel-size`, `--enforce-eager`, etc."
        ),
    )
    artifact_location: Optional[str] = Field(
        None,
        description="Path to the model artifact relative to the multi-model base folder.",
    )
    fine_tune_weights: Optional[List[LoraModuleSpec]] = Field(
        None,
        description="List of fine-tuned weight modules (e.g., LoRA) associated with this base model.",
    )

    def all_model_ids(self) -> List[str]:
        """
        Returns all model OCIDs associated with this reference, including fine-tuned weights.

        Returns
        -------
        List[str]
            A list containing the base model OCID and any fine-tuned module OCIDs.
        """
        ids = {self.model_id}
        if self.fine_tune_weights:
            ids.update(
                module.model_id for module in self.fine_tune_weights if module.model_id
            )
        return list(ids)

    @model_validator(mode="before")
    @classmethod
    def extract_params_from_env_var(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        A model-level validator that extracts `PARAMS` from the `env_var` dictionary
        and injects them into the `params` field as a dictionary.

        This is useful for backward compatibility where users pass CLI-style
        parameters via environment variables, e.g.:
        env_var = { "PARAMS": "--max-model-len 65536 --enable-streaming" }

        If `params` is already set, values from `PARAMS` in `env_var` are added
        only if they do not override existing keys.
        """
        env = values.get("env_var", {})
        param_string = env.pop("PARAMS", None)

        if param_string:
            parsed_params = cls._parse_params(params=param_string)
            existing_params = values.get("params", {}) or {}
            # Avoid overriding existing keys
            for k, v in parsed_params.items():
                if k not in existing_params:
                    existing_params[k] = v
            values["params"] = existing_params
            values["env_var"] = env  # cleaned up version without PARAMS

        return values

    @staticmethod
    def _parse_params(params: Union[str, List[str]]) -> Dict[str, str]:
        """
        Parses CLI-style parameters into a dictionary format.

        This method accepts either:
        - A single string of parameters (e.g., "--key1 val1 --key2 val2")
        - A list of strings (e.g., ["--key1", "val1", "--key2", "val2"])

        Returns a dictionary of the form { "key1": "val1", "key2": "val2" }.

        Parameters
        ----------
        params : Union[str, List[str]]
            The parameters to parse. Can be a single string or a list of strings.

        Returns
        -------
        Dict[str, str]
            Dictionary with parameter names as keys and their corresponding values as strings.
        """
        if not params or not isinstance(params, (str, list)):
            return {}

        # Normalize string to list of "--key value" strings
        if isinstance(params, str):
            params_list = [
                f"--{param.strip()}" for param in params.split("--") if param.strip()
            ]
        else:
            params_list = params

        parsed = {}
        for item in params_list:
            parts = item.strip().split()
            if not parts:
                continue
            key = parts[0]
            value = " ".join(parts[1:]) if len(parts) > 1 else ""
            parsed[key] = value

        return parsed

    class Config:
        extra = "allow"
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
