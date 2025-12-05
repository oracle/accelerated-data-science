#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from typing_extensions import Self

from ads.aqua import logger
from ads.aqua.common.entities import AquaMultiModelRef
from ads.aqua.common.errors import AquaValueError
from ads.aqua.common.utils import (
    build_params_string,
    find_restricted_params,
    get_container_params_type,
    get_params_dict,
)
from ads.aqua.config.utils.serializer import Serializable
from ads.aqua.modeldeployment.config_loader import (
    AquaDeploymentConfig,
    ConfigurationItem,
    ModelDeploymentConfigSummary,
)
from ads.aqua.modeldeployment.entities import (
    CreateModelDeploymentDetails,
    UpdateModelDeploymentDetails,
)
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.utils import UNKNOWN

__all__ = ["ModelGroupConfig", "BaseModelSpec"]

from ads.aqua.common.entities import LoraModuleSpec


class BaseModelSpec(BaseModel):
    """
    Defines configuration for a single base model in multi-model deployment.

    Attributes
    ----------
    model_id: str
        The OCID of the base model.
    model_path : str
        Path to the model in OCI Object Storage.
    params : str
        Additional vLLM launch parameters for this model (e.g. parallelism, max context).
    model_task : str, optional
        Model task type (e.g., text-generation, image-to-text).
    fine_tune_weights : List[List[LoraModuleSpec]], optional
        List of associated LoRA modules for fine-tuned models.
    """

    model_id: str = Field(..., description="The base model OCID.")
    model_path: str = Field(..., description="Path to the base model.")
    params: str = Field(..., description="Startup parameters passed to vLLM.")
    model_task: Optional[str] = Field(
        ..., description="Task type the model is intended for."
    )
    fine_tune_weights: Optional[List[LoraModuleSpec]] = Field(
        default_factory=list,
        description="Optional list of fine-tuned model variants associated with this base model.",
    )

    @classmethod
    def build_model_path(cls, model_id: str, artifact_path_prefix: str) -> str:
        """Cleans and builds the file path for model_path parameter
        to format: <model_id>/<artifact_path_prefix>
        """
        if not ObjectStorageDetails.is_oci_path(artifact_path_prefix):
            raise AquaValueError(
                "The base model path is not available in the model artifact."
            )

        os_path = ObjectStorageDetails.from_path(artifact_path_prefix)
        artifact_path_prefix = os_path.filepath.rstrip("/")
        return model_id + "/" + artifact_path_prefix.lstrip("/")

    @classmethod
    def dedup_lora_modules(cls, fine_tune_weights: List[LoraModuleSpec]):
        """Removes duplicate LoRA Modules (duplicate model_names in fine_tune_weights)"""
        seen = set()
        unique_modules: List[LoraModuleSpec] = []

        for module in fine_tune_weights or []:
            if module.model_name and module.model_name in seen:
                logger.warning(
                    f"Duplicate LoRA Module detected: {module.model_name!r} (skipping duplicate)."
                )
                continue
            seen.add(module.model_name)
            unique_modules.append(module)

        return unique_modules

    @classmethod
    def from_aqua_multi_model_ref(
        cls, model: AquaMultiModelRef, model_params: str
    ) -> Self:
        """Converts AquaMultiModelRef to BaseModelSpec. Fields are validated using @field_validator methods above."""

        return cls(
            model_id=model.model_id,
            model_path=cls.build_model_path(model.model_id, model.artifact_location),
            params=model_params,
            model_task=model.model_task,
            fine_tune_weights=cls.dedup_lora_modules(model.fine_tune_weights),
        )


class ModelGroupConfig(Serializable):
    """
    Schema representing the metadata passed via MULTI_MODEL_CONFIG for multi-model deployments.

    Attributes
    ----------
    models : List[BaseModelConfig]
        List of base models (with optional fine-tune weights) to be served.
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
        user_params = build_params_string(model.params)
        if user_params:
            restricted_params = find_restricted_params(
                container_params, user_params, container_type_key
            )
            if restricted_params:
                selected_model = model.model_name or model.model_id
                raise AquaValueError(
                    f"Parameters {restricted_params} are set by AI Quick Actions "
                    f"and cannot be overridden or are invalid. "
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
        deployment_details: Union[
            CreateModelDeploymentDetails, UpdateModelDeploymentDetails
        ],
        container_type_key: str,
        container_params,
    ):
        """Finds the corresponding deployment parameters based on the GPU count
        and combines them with user's parameters. Existing deployment parameters
        will be overriden by user's parameters."""
        user_params, params = ModelGroupConfig._extract_model_params(
            model, container_params, container_type_key
        )

        deployment_config = model_config_summary.deployment_config.get(
            model.model_id, AquaDeploymentConfig()
        ).configuration.get(deployment_details.instance_shape, ConfigurationItem())

        final_model_params = user_params
        params_found = False
        user_explicitly_cleared = model.params is not None and not model.params

        # Only load defaults if user didn't provide params AND didn't explicitly clear them
        if not user_params and not user_explicitly_cleared:
            for item in deployment_config.multi_model_deployment:
                if (
                    model.gpu_count
                    and item.gpu_count
                    and item.gpu_count == model.gpu_count
                ):
                    config_parameters = item.parameters.get(
                        get_container_params_type(container_type_key), UNKNOWN
                    )
                    if config_parameters:
                        final_model_params = config_parameters
                    params_found = True
                    break

            if not params_found and deployment_config.parameters:
                config_parameters = deployment_config.parameters.get(
                    get_container_params_type(container_type_key), UNKNOWN
                )
                if config_parameters:
                    final_model_params = config_parameters
                params_found = True

        # Combine Container System Defaults (params) + Model Params (final_model_params)
        params = f"{params} {final_model_params}".strip()

        return params

    @classmethod
    def from_model_deployment_details(
        cls,
        deployment_details: Union[
            CreateModelDeploymentDetails, UpdateModelDeploymentDetails
        ],
        model_config_summary: ModelDeploymentConfigSummary,
        container_type_key,
        container_params,
    ) -> Self:
        """
        Converts CreateModelDeploymentDetail to ModelGroupConfig.
        CreateModelDeploymentDetail represents user-provided parameters and models within a multi-model group after model artifact is created.
        ModelGroupConfig is the Pydantic representation of MULTI_MODEL_CONFIG environment variable during model deployment.
        """
        models = []
        seen_models = set()
        for model in deployment_details.models:
            params = ModelGroupConfig._merge_gpu_count_params(
                model,
                model_config_summary,
                deployment_details,
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
