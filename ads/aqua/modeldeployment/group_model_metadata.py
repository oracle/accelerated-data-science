#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self

from ads.aqua import logger
from ads.aqua.common.entities import AquaMultiModelRef
from ads.aqua.common.errors import AquaValueError
from ads.aqua.common.utils import (
    build_params_string,
    find_restricted_params,
    get_combined_params,
    get_container_params_type,
    get_params_dict,
)
from ads.aqua.config.utils.serializer import Serializable
from ads.aqua.modeldeployment.config_loader import (
    AquaDeploymentConfig,
    ConfigurationItem,
    ModelDeploymentConfigSummary,
)
from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.utils import UNKNOWN

__all__ = ["GroupModelDeploymentMetadata", "BaseModelSpec"]

from ads.aqua.common.entities import LoraModuleSpec


class BaseModelSpec(BaseModel):
    """
    Defines configuration for a single base model in multi-model deployment.

    Attributes
    ----------
    model_path : str
        Path to the model in OCI Object Storage.
    params : str
        Additional vLLM launch parameters for this model (e.g. parallelism, max context).
    model_task : str, optional
        Model task type (e.g., text-generation, image-to-text).
    fine_tune_weights : List[FineTunedModelSpec], optional
        List of associated fine-tuned models.
    """

    model_path: str = Field(..., description="Path to the base model.")
    params: str = Field(..., description="Startup parameters passed to vLLM.")
    model_task: Optional[str] = Field(
        ..., description="Task type the model is intended for."
    )
    fine_tune_weights: Optional[List[LoraModuleSpec]] = Field(
        default_factory=list,
        description="Optional list of fine-tuned model variants associated with this base model.",
    )

    @field_validator("model_path")
    @classmethod
    def clean_model_path(cls, artifact_path_prefix: str) -> str:
        """Validates and cleans the file path for model_path parameter."""
        if ObjectStorageDetails.is_oci_path(artifact_path_prefix):
            os_path = ObjectStorageDetails.from_path(artifact_path_prefix)
            artifact_path_prefix = os_path.filepath.rstrip("/")
            return artifact_path_prefix

        raise AquaValueError(
            "The base model path is not available in the model artifact."
        )

    @field_validator("fine_tune_weights")
    @classmethod
    def set_fine_tuned_weights(cls, fine_tuned_weights: List[LoraModuleSpec]):
        """Removes duplicate LoRA Modules (duplicate model_names in fine_tuned_weights)"""
        seen_modules = set()
        unique_modules: List[LoraModuleSpec] = []

        if not fine_tuned_weights:
            return None

        for lora_module in fine_tuned_weights:
            if lora_module.model_name not in seen_modules:
                seen_modules.add(lora_module.model_name)
                unique_modules.append(lora_module)
            else:
                logger.warning(
                    f"Duplicate LoRA Modules Detected. Previously loaded LoRA Module {(lora_module.model_name,)}",
                )
        return unique_modules

    @classmethod
    def from_aqua_multi_model_ref(
        cls, model: AquaMultiModelRef, model_params: str
    ) -> Self:
        """Converts AquaMultiModelRef to BaseModelSpec. Fields are validated using @field_validator methods above."""

        return cls(
            model_path=model.artifact_location,
            params=model_params,
            model_task=model.model_task,
            fine_tuned_weights=model.fine_tune_weights,
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

    @staticmethod
    def _extract_model_params(
        model: AquaMultiModelRef,
        container_params: Union[str, List[str]],
        container_type_key: str,
    ) -> Tuple[str, str]:
        """
        Validates if user-provided parameters override pre-set parameters by AQUA.
        Updates model name and TP size parameters to user-provided parameters.
        """
        user_params = build_params_string(model.env_var)
        if user_params:
            restricted_params = find_restricted_params(
                container_params, user_params, container_type_key
            )
            if restricted_params:
                selected_model = model.model_name or model.model_id
                raise AquaValueError(
                    f"Parameters {restricted_params} are set by Aqua "
                    f"and cannot be overridden or are invalid."
                    f"Select other parameters for model {selected_model}."
                )

        # replaces `--served-model-name`` with user's model name
        container_params_dict = get_params_dict(container_params)
        container_params_dict.update({"--served-model-name": model.model_name})
        # replaces `--tensor-parallel-size` with model gpu count
        container_params_dict.update({"--tensor-parallel-size": model.gpu_count})
        params = build_params_string(container_params_dict)

        return user_params, params

    @staticmethod
    def _merge_gpu_count_params(
        model: AquaMultiModelRef,
        model_config_summary: ModelDeploymentConfigSummary,
        create_deployment_details: CreateModelDeploymentDetails,
        container_type_key: str,
        container_params,
    ):
        """Finds the corresponding deployment parameters based on the GPU count
        and combines them with user's parameters. Existing deployment parameters
        will be overriden by user's parameters."""
        user_params, params = GroupModelDeploymentMetadata._extract_model_params(
            model, container_params, container_type_key
        )

        deployment_config = model_config_summary.deployment_config.get(
            model.model_id, AquaDeploymentConfig()
        ).configuration.get(
            create_deployment_details.instance_shape, ConfigurationItem()
        )
        params_found = False
        for item in deployment_config.multi_model_deployment:
            if model.gpu_count and item.gpu_count and item.gpu_count == model.gpu_count:
                config_parameters = item.parameters.get(
                    get_container_params_type(container_type_key), UNKNOWN
                )
                params = f"{params} {get_combined_params(config_parameters, user_params)}".strip()
                params_found = True
                break

        if not params_found and deployment_config.parameters:
            config_parameters = deployment_config.parameters.get(
                get_container_params_type(container_type_key), UNKNOWN
            )
            params = f"{params} {get_combined_params(config_parameters, user_params)}".strip()
            params_found = True

        # if no config parameters found, append user parameters directly.
        if not params_found:
            params = f"{params} {user_params}".strip()

        return params

    @classmethod
    def from_create_model_deployment_details(
        cls,
        create_deployment_details: CreateModelDeploymentDetails,
        model_config_summary: ModelDeploymentConfigSummary,
        container_type_key,
        container_params,
    ) -> Self:
        """
        Converts CreateModelDeploymentDetail to GroupModelDeploymentMetadata.
        CreateModelDeploymentDetail represents user-provided parameters and models within a multi-model group after model artifact is created.
        GroupModelDeploymentMetadata is the Pydantic representation of MULTI_MODEL_CONFIG environment variable during model deployment.
        """
        models = []
        seen_models = set()
        for model in create_deployment_details.models:
            params = GroupModelDeploymentMetadata._merge_gpu_count_params(
                model,
                model_config_summary,
                create_deployment_details,
                container_type_key,
                container_params,
            )

            if model.model_name not in seen_models:
                seen_models.add(model.model_name)
                base_model_spec = BaseModelSpec.from_aqua_multi_model_ref(model, params)
                models.append(base_model_spec)
            else:
                raise AquaValueError(
                    f"Duplicate model name ‘{model.model_name}’ detected in multi-model group. "
                    "Each base model must have a unique `model_name`. "
                    "Please remove or rename the duplicate model and register the model group again."
                )

        return cls(models=models)
