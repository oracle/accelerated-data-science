#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict, List, Optional, Union

from oci.data_science.models import ModelDeployment, ModelDeploymentSummary
from pydantic import BaseModel, Field, model_validator

from ads.aqua import logger
from ads.aqua.common.entities import AquaMultiModelRef, LoraModuleSpec
from ads.aqua.common.enums import Tags
from ads.aqua.common.errors import AquaValueError
from ads.aqua.common.utils import is_valid_ocid
from ads.aqua.config.utils.serializer import Serializable
from ads.aqua.constants import (
    AQUA_FINE_TUNE_MODEL_VERSION,
    INCLUDE_BASE_MODEL,
    UNKNOWN_DICT,
)
from ads.aqua.data import AquaResourceIdentifier
from ads.aqua.finetuning.constants import FineTuneCustomMetadata
from ads.aqua.modeldeployment.config_loader import (
    ConfigurationItem,
    ModelDeploymentConfigSummary,
)
from ads.common.serializer import DataClassSerializable
from ads.common.utils import UNKNOWN, get_console_link
from ads.model.datascience_model import DataScienceModel
from ads.model.deployment.model_deployment import ModelDeploymentType
from ads.model.model_metadata import ModelCustomMetadataItem


class ConfigValidationError(Exception):
    """Exception raised for config validation."""

    def __init__(
        self,
        message: str = (
            "Validation failed: The provided model group configuration is incompatible "
            "with the selected instance shape. Please verify the GPU count per model and ensure "
            "multi-model deployment is supported for the chosen instance shape."
        ),
    ):
        super().__init__(message)


class ShapeInfo(Serializable):
    """
    Represents the configuration details for a compute instance shape.
    """

    instance_shape: Optional[str] = Field(
        default=None,
        description="The identifier of the compute instance shape (e.g., VM.Standard2.1)",
    )
    instance_count: Optional[int] = Field(
        default=None, description="The number of instances for the given shape."
    )
    ocpus: Optional[float] = Field(
        default=None,
        description="The number of Oracle CPUs allocated for the instance.",
    )
    memory_in_gbs: Optional[float] = Field(
        default=None,
        description="The total memory allocated for the instance, in gigabytes.",
    )
    capacity_reservation_ids: Optional[List[str]] = Field(
        default=None,
        description="The list of capacity reservation OCIDs for the deployment.",
    )


class ModelParams(Serializable):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[float] = None
    top_p: Optional[float] = None
    model: Optional[str] = None

    class Config:
        extra = "allow"
        protected_namespaces = ()


class AquaDeployment(Serializable):
    """Represents an Aqua Model Deployment"""

    id: Optional[str] = Field(None, description="The model deployment OCID.")
    display_name: Optional[str] = Field(
        None, description="The name of the model deployment."
    )
    aqua_service_model: Optional[bool] = Field(
        False, description="The bool value to indicate if it's aqua service model."
    )
    model_id: str = Field(..., description="The model OCID to deploy.")
    models: Optional[List[AquaMultiModelRef]] = Field(
        default_factory=list, description="List of models for multi model deployment."
    )
    aqua_model_name: Optional[str] = Field(
        None, description="The name of the aqua model."
    )
    state: Optional[str] = Field(None, description="The state of the model deployment.")
    description: Optional[str] = Field(
        None, description="The description of the model deployment."
    )
    created_on: Optional[str] = Field(
        None, description="The creation time of the model deployment."
    )
    created_by: Optional[str] = Field(
        None, description="The OCID that creates the model deployment."
    )
    endpoint: Optional[str] = Field(
        None, description="The endpoint of the model deployment."
    )
    private_endpoint_id: Optional[str] = Field(
        None, description="The private endpoint id of the model deployment."
    )
    console_link: Optional[str] = Field(
        None, description="The console link of the model deployment."
    )
    lifecycle_details: Optional[str] = Field(
        None, description="The lifecycle details of the model deployment."
    )
    shape_info: Optional[ShapeInfo] = Field(
        default_factory=ShapeInfo,
        description="The shape information of the model deployment.",
    )
    tags: Optional[dict] = Field(
        default_factory=dict, description="The tags of the model deployment."
    )
    environment_variables: Optional[dict] = Field(
        default_factory=dict,
        description="The environment variables of the model deployment.",
    )
    cmd: Optional[List[str]] = Field(
        default_factory=list, description="The cmd of the model deployment."
    )
    subnet_id: Optional[str] = Field(
        None, description="The custom egress for model deployment."
    )

    @classmethod
    def from_oci_model_deployment(
        cls,
        oci_model_deployment: Union[ModelDeploymentSummary, ModelDeployment],
        region: str,
    ) -> "AquaDeployment":
        """Converts oci model deployment response to AquaDeployment instance.

        Parameters
        ----------
        oci_model_deployment: Union[ModelDeploymentSummary, ModelDeployment]
            The instance of either oci.data_science.models.ModelDeployment or
            oci.data_science.models.ModelDeploymentSummary class.
        region: str
            The region of this model deployment.

        Returns
        -------
        AquaDeployment:
            The instance of the Aqua model deployment.
        """
        model_deployment_configuration_details = (
            oci_model_deployment.model_deployment_configuration_details
        )
        if (
            model_deployment_configuration_details.deployment_type
            == ModelDeploymentType.SINGLE_MODEL
        ):
            instance_configuration = (
                model_deployment_configuration_details.model_configuration_details.instance_configuration
            )
            instance_count = (
                model_deployment_configuration_details.model_configuration_details.scaling_policy.instance_count
            )
            model_id = (
                model_deployment_configuration_details.model_configuration_details.model_id
            )
        elif (
            model_deployment_configuration_details.deployment_type
            == ModelDeploymentType.MODEL_GROUP
        ):
            instance_configuration = (
                model_deployment_configuration_details.infrastructure_configuration_details.instance_configuration
            )
            instance_count = (
                model_deployment_configuration_details.infrastructure_configuration_details.scaling_policy.instance_count
            )
            model_id = (
                model_deployment_configuration_details.model_group_configuration_details.model_group_id
            )
        else:
            allowed_deployment_types = ", ".join(
                [key for key in dir(ModelDeploymentType) if not key.startswith("__")]
            )
            raise AquaValueError(
                f"Invalid AQUA deployment with type {model_deployment_configuration_details.deployment_type}."
                f"Only {allowed_deployment_types} are supported at this moment. Specify a different AQUA model deployment."
            )

        instance_shape_config_details = (
            instance_configuration.model_deployment_instance_shape_config_details
        )
        environment_variables = (
            model_deployment_configuration_details.environment_configuration_details.environment_variables
        )
        cmd = (
            model_deployment_configuration_details.environment_configuration_details.cmd
        )

        # Extract capacity_reservation_ids if available
        capacity_reservation_ids = getattr(
            instance_configuration, "capacity_reservation_ids", None
        )

        shape_info = ShapeInfo(
            instance_shape=instance_configuration.instance_shape_name,
            instance_count=instance_count,
            ocpus=(
                instance_shape_config_details.ocpus
                if instance_shape_config_details
                else None
            ),
            memory_in_gbs=(
                instance_shape_config_details.memory_in_gbs
                if instance_shape_config_details
                else None
            ),
            capacity_reservation_ids=capacity_reservation_ids,
        )
        tags = {}
        tags.update(oci_model_deployment.freeform_tags or UNKNOWN_DICT)
        tags.update(oci_model_deployment.defined_tags or UNKNOWN_DICT)

        aqua_service_model_tag = tags.get(Tags.AQUA_SERVICE_MODEL_TAG, None)
        aqua_model_name = tags.get(Tags.AQUA_MODEL_NAME_TAG, UNKNOWN)
        private_endpoint_id = getattr(
            instance_configuration, "private_endpoint_id", UNKNOWN
        )
        subnet_id = getattr(instance_configuration, "subnet_id", UNKNOWN)

        return AquaDeployment(
            id=oci_model_deployment.id,
            model_id=model_id,
            display_name=oci_model_deployment.display_name,
            aqua_service_model=aqua_service_model_tag is not None,
            aqua_model_name=aqua_model_name,
            shape_info=shape_info,
            state=oci_model_deployment.lifecycle_state,
            lifecycle_details=getattr(
                oci_model_deployment, "lifecycle_details", UNKNOWN
            ),
            description=oci_model_deployment.description,
            created_on=str(oci_model_deployment.time_created),
            created_by=oci_model_deployment.created_by,
            endpoint=oci_model_deployment.model_deployment_url,
            private_endpoint_id=private_endpoint_id,
            console_link=get_console_link(
                resource="model-deployments",
                ocid=oci_model_deployment.id,
                region=region,
            ),
            tags=tags,
            environment_variables=environment_variables,
            cmd=cmd,
            subnet_id=subnet_id,
        )

    class Config:
        extra = "ignore"
        protected_namespaces = ()


class AquaDeploymentDetail(AquaDeployment, DataClassSerializable):
    """Represents a details of Aqua deployment."""

    log_group: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)
    log: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)

    class Config:
        extra = "allow"


class ModelDeploymentDetails(BaseModel):
    """Class for the base details of creating and updating Aqua model deployments."""

    instance_shape: str = Field(
        None, description="The instance shape used for deployment."
    )
    display_name: Optional[str] = Field(
        None, description="The name of the model deployment."
    )
    description: Optional[str] = Field(
        None, description="The description of the deployment."
    )
    models: Optional[List[AquaMultiModelRef]] = Field(
        None, description="List of models for multimodel or stacked deployment."
    )
    instance_count: Optional[int] = Field(
        None, description="Number of instances used for deployment."
    )
    container_family: Optional[str] = Field(
        None, description="Image family of the model deployment container runtime."
    )
    log_group_id: Optional[str] = Field(
        None, description="OCI logging group ID for logs."
    )
    access_log_id: Optional[str] = Field(
        None,
        description="OCID for access logs. "
        "https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm",
    )
    predict_log_id: Optional[str] = Field(
        None,
        description="OCID for prediction logs."
        "https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm",
    )
    bandwidth_mbps: Optional[int] = Field(
        None, description="Bandwidth limit on the load balancer in Mbps."
    )
    web_concurrency: Optional[int] = Field(
        None, description="Number of worker processes/threads for handling requests."
    )
    memory_in_gbs: Optional[float] = Field(
        None, description="Memory (in GB) for the selected shape."
    )
    ocpus: Optional[float] = Field(
        None, description="OCPU count for the selected shape."
    )
    freeform_tags: Optional[Dict] = Field(
        None, description="Freeform tags for model deployment."
    )
    defined_tags: Optional[Dict] = Field(
        None, description="Defined tags for model deployment."
    )

    def validate_multimodel_deployment_feasibility(
        self, models_config_summary: ModelDeploymentConfigSummary
    ) -> None:
        """
        Validates whether the selected model group is feasible for a multi-model deployment
        on the chosen instance shape.

        Validation Criteria:
        - Ensures that the model group is not empty.
        - Verifies that the selected instance shape is supported by the GPU allocation.
        - Confirms that each model in the group has a corresponding deployment configuration.
        - Ensures that each model's user-specified GPU allocation is allowed by its deployment configuration.
        - Checks that the total GPUs requested by the model group does not exceed the available GPU capacity
            for the selected instance shape.

        Parameters
        ----------
        models_config_summary : ModelDeploymentConfigSummary
            Contains GPU allocations and deployment configuration for models.

        Raises
        ------
        ConfigValidationError:
        - If the model group is empty.
        - If the selected instance shape is not supported.
        - If any model is missing from the deployment configuration.
        - If a model's GPU allocation does not match any valid configuration.
        - If the total requested GPUs exceed the instance shape’s capacity.
        """
        # Ensure that at least one model is provided.
        if not self.models:
            logger.error("No models provided in the model group.")
            raise ConfigValidationError(
                "Multi-model deployment requires at least one model. Please provide one or more models."
            )

        selected_shape = self.instance_shape

        if models_config_summary.error_message:
            logger.error(models_config_summary.error_message)
            raise ConfigValidationError(models_config_summary.error_message)

        # Verify that the selected shape is supported by the GPU allocation.
        if selected_shape not in models_config_summary.gpu_allocation:
            supported_shapes = list(models_config_summary.gpu_allocation.keys())
            error_message = (
                f"The model group is not compatible with the selected instance shape `{selected_shape}`. "
                f"Supported shapes: {supported_shapes}."
            )
            logger.error(error_message)
            raise ConfigValidationError(error_message)

        total_available_gpus: int = models_config_summary.gpu_allocation[
            selected_shape
        ].total_gpus_available
        model_deployment_config = models_config_summary.deployment_config

        # Verify that every model in the group has a corresponding deployment configuration.
        required_model_ids = {model.model_id for model in self.models}
        missing_model_ids = required_model_ids - set(model_deployment_config.keys())
        if missing_model_ids:
            error_message = (
                f"Missing deployment configuration for models: {list(missing_model_ids)}. "
                "Ensure all selected models are properly configured. If you are deploying custom "
                "models that lack AQUA service configuration, refer to the deployment guidelines here: "
                "https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/multimodel-deployment-tips.md#custom_models"
            )
            logger.error(error_message)
            raise ConfigValidationError(error_message)

        sum_model_gpus = 0
        is_single_model = len(self.models) == 1

        # Validate each model's GPU allocation against its deployment configuration.
        for model in self.models:
            sum_model_gpus += model.gpu_count
            aqua_deployment_config = model_deployment_config[model.model_id]

            # Skip validation for models without deployment configuration details.
            if not aqua_deployment_config.configuration:
                error_message = (
                    f"Missing deployment configuration for model `{model.model_id}`. "
                    "Please verify that the model is correctly configured. If you are deploying custom models without AQUA service configuration, "
                    "refer to the guidelines at: "
                    "https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/multimodel-deployment-tips.md#custom_models"
                )

                logger.error(error_message)
                raise ConfigValidationError(error_message)

            allowed_shapes = (
                list(
                    set(aqua_deployment_config.configuration.keys()).union(
                        set(aqua_deployment_config.shape or [])
                    )
                )
                if is_single_model
                else list(aqua_deployment_config.configuration.keys())
            )

            if selected_shape not in allowed_shapes:
                error_message = (
                    f"Model `{model.model_id}` is not compatible with the selected instance shape `{selected_shape}`. "
                    f"Select a different instance shape from allowed shapes {allowed_shapes}."
                )
                logger.error(error_message)
                raise ConfigValidationError(error_message)

            # Retrieve valid GPU counts for the selected shape.
            multi_model_configs = aqua_deployment_config.configuration.get(
                selected_shape, ConfigurationItem()
            ).multi_model_deployment

            valid_gpu_configurations = [cfg.gpu_count for cfg in multi_model_configs]

            if model.gpu_count not in valid_gpu_configurations:
                valid_gpu_str = valid_gpu_configurations or []

                if is_single_model:
                    # If total GPU allocation is not supported by selected model
                    if selected_shape not in aqua_deployment_config.shape:
                        error_message = (
                            f"Model `{model.model_id}` is configured with {model.gpu_count} GPU(s), "
                            f"which is invalid. The allowed GPU configurations are: {valid_gpu_str}."
                        )
                        logger.error(error_message)
                        raise ConfigValidationError(error_message)

                    if model.gpu_count != total_available_gpus:
                        error_message = (
                            f"Model '{model.model_id}' is configured to use {model.gpu_count} GPU(s), "
                            f"which not fully utilize the selected instance shape with {total_available_gpus} available GPU(s). "
                            "Consider adjusting the GPU allocation to better utilize the available resources and maximize performance."
                        )
                        logger.error(error_message)
                        raise ConfigValidationError(error_message)

                else:
                    error_message = (
                        f"Model `{model.model_id}` is configured with {model.gpu_count} GPU(s), which is invalid. "
                        f"Valid GPU configurations are: {valid_gpu_str}. Please adjust the GPU allocation "
                        f"or choose an instance shape that supports a higher GPU count."
                    )
                    logger.error(error_message)
                    raise ConfigValidationError(error_message)

        if sum_model_gpus < total_available_gpus:
            error_message = (
                f"Selected models are configured to use {sum_model_gpus} GPU(s), "
                f"which not fully utilize the selected instance shape with {total_available_gpus} available GPU(s). "
                "This configuration may lead to suboptimal performance for a multi-model deployment. "
                "Consider adjusting the GPU allocation to better utilize the available resources and maximize performance."
            )
            logger.warning(error_message)
            # raise ConfigValidationError(error_message)

        # Check that the total GPU count for the model group does not exceed the instance capacity.
        if sum_model_gpus > total_available_gpus:
            error_message = (
                f"The selected instance shape `{selected_shape}` provides `{total_available_gpus}` GPU(s), "
                f"but the total GPU allocation required by the model group is `{sum_model_gpus}` GPU(s). "
                "Please adjust the GPU allocation per model or choose an instance shape with greater GPU capacity."
            )
            logger.error(error_message)
            raise ConfigValidationError(error_message)

    def validate_input_models(self, model_details: Dict[str, DataScienceModel]) -> None:
        """
        Validates the input models for a stacked-model or multi-model deployment configuration.

        Validation Criteria:
        - The base model must be explicitly provided.
        - The base model must be in 'ACTIVE' state.
        - Fine-tuned models must have a tag 'fine_tune_model_version' as v2 to be supported.
        - Fine-tuned models must not have custom metadata 'include_base_model_artifact' as 1.
        - Fine-tuned model IDs must refer to valid, tagged fine-tuned models.
        - Fine-tuned models must refer back to the same base model.
        - All model names (including fine-tuned variants) must be unique.

        Parameters
        ----------
        model_details : Dict[str, DataScienceModel]
            Dictionary mapping model OCIDs to DataScienceModel instances.
            Includes the all models to validate including fine-tuned models.

        Raises
        ------
        ConfigValidationError
            If any of the above conditions are violated.
        """
        if not self.models:
            logger.error("Validation failed: No models specified in the model group.")
            raise ConfigValidationError(
                "Multi-model deployment requires at least one model entry. "
                "Please provide a base model in the `models` list."
            )

        seen_names = set()
        duplicate_names = set()

        for model in self.models:
            base_model_id = model.model_id
            base_model = model_details.get(base_model_id)

            if not base_model:
                logger.error(
                    "Validation failed: Base model '%s' (id: '%s') not found.",
                    model.model_name,
                    base_model_id,
                )
                raise ConfigValidationError(
                    f"Model not found: '{model.model_name}' (id: '{base_model_id}')."
                )

            if Tags.AQUA_FINE_TUNED_MODEL_TAG in (base_model.freeform_tags or {}):
                logger.error(
                    "Validation failed: Base model '%s' (id: '%s') is a fine-tuned model.",
                    base_model.display_name,
                    base_model_id,
                )
                raise ConfigValidationError(
                    f"Invalid base model '{base_model.display_name}' (id: '{base_model_id}'). "
                    "Specify a base model OCID in the `models` input, not a fine-tuned model."
                )

            if base_model.lifecycle_state != "ACTIVE":
                logger.error(
                    "Validation failed: Base model '%s' (id: '%s') is in state '%s'.",
                    base_model.display_name,
                    base_model_id,
                    base_model.lifecycle_state,
                )
                raise ConfigValidationError(
                    f"Invalid base model '{base_model.display_name}' (id: '{base_model_id}'): must be in ACTIVE state."
                )

            # Normalize and validate model name uniqueness
            model_name = model.model_name or base_model.display_name
            if model_name in seen_names:
                duplicate_names.add(model_name)
            else:
                seen_names.add(model_name)

            for lora_module in model.fine_tune_weights or []:
                ft_model_id = lora_module.model_id
                ft_model = model_details.get(ft_model_id)

                if not ft_model:
                    logger.error(
                        "Validation failed: Fine-tuned model '%s' (id: '%s') not found.",
                        lora_module.model_name,
                        ft_model_id,
                    )
                    raise ConfigValidationError(
                        f"Fine-tuned model not found: '{lora_module.model_name}' (id: '{ft_model_id}')."
                    )

                if ft_model.lifecycle_state != "ACTIVE":
                    logger.error(
                        "Validation failed: Fine-tuned model '%s' (id: '%s') is in state '%s'.",
                        ft_model.display_name,
                        ft_model_id,
                        ft_model.lifecycle_state,
                    )
                    raise ConfigValidationError(
                        f"Invalid Fine-tuned model '{ft_model.display_name}' (id: '{ft_model_id}'): must be in ACTIVE state."
                    )

                if Tags.AQUA_FINE_TUNED_MODEL_TAG not in (ft_model.freeform_tags or {}):
                    logger.error(
                        "Validation failed: Model '%s' (id: '%s') is missing tag '%s'.",
                        ft_model.display_name,
                        ft_model_id,
                        Tags.AQUA_FINE_TUNED_MODEL_TAG,
                    )
                    raise ConfigValidationError(
                        f"Invalid fine-tuned model '{ft_model.display_name}' (id: '{ft_model_id}'): missing tag '{Tags.AQUA_FINE_TUNED_MODEL_TAG}'."
                    )

                self.validate_ft_model_v2(model=ft_model)

                ft_base_model_id = ft_model.custom_metadata_list.get(
                    FineTuneCustomMetadata.FINE_TUNE_SOURCE,
                    ModelCustomMetadataItem(
                        key=FineTuneCustomMetadata.FINE_TUNE_SOURCE
                    ),
                ).value

                if ft_base_model_id != base_model_id:
                    logger.error(
                        "Validation failed: Fine-tuned model '%s' (id: '%s') is linked to base model '%s' (expected '%s' with id: '%s').",
                        ft_model.display_name,
                        ft_model_id,
                        ft_base_model_id,
                        base_model.display_name,
                        base_model_id,
                    )
                    raise ConfigValidationError(
                        f"Fine-tuned model '{ft_model.display_name}' (id: '{ft_model_id}') belongs to base model '{ft_base_model_id}', "
                        f"but was included under base model '{base_model.display_name}' (id: '{base_model_id}')."
                    )

                # Validate fine-tuned model name uniqueness
                lora_model_name = lora_module.model_name or ft_model.display_name
                if lora_model_name in seen_names:
                    duplicate_names.add(lora_model_name)
                else:
                    seen_names.add(lora_model_name)

                logger.debug(
                    "Validated fine-tuned model '%s' (id: '%s') under base model '%s' (id: '%s').",
                    ft_model.display_name,
                    ft_model_id,
                    base_model.display_name,
                    base_model_id,
                )

        if duplicate_names:
            logger.error(
                "Duplicate model names detected: %s", ", ".join(sorted(duplicate_names))
            )
            raise ConfigValidationError(
                f"The following model names are duplicated across base and fine-tuned models: "
                f"{', '.join(sorted(duplicate_names))}. Model names must be unique for proper routing in multi-model deployments."
            )

    def validate_ft_model_v2(
        self, model_id: Optional[str] = None, model: Optional[DataScienceModel] = None
    ) -> None:
        """
        Validates the input fine tuned model for model deployment configuration.

        Validation Criteria:
        - Fine-tuned models must have a tag 'fine_tune_model_version' as v2 to be supported.
        - Fine-tuned models must not have custom metadata 'include_base_model_artifact' as '1'.

        Parameters
        ----------
        model_id : str
            The OCID of DataScienceModel instance.
        model : DataScienceModel
            The DataScienceModel instance.

        Raises
        ------
        ConfigValidationError
            If any of the above conditions are violated.
        """
        base_model = DataScienceModel.from_id(model_id) if model_id else model
        if Tags.AQUA_FINE_TUNED_MODEL_TAG in base_model.freeform_tags:
            if (
                base_model.freeform_tags.get(
                    Tags.AQUA_FINE_TUNE_MODEL_VERSION, UNKNOWN
                ).lower()
                != AQUA_FINE_TUNE_MODEL_VERSION
            ):
                logger.error(
                    "Validation failed: Fine-tuned model '%s' (id: '%s') is not supported for model deployment.",
                    base_model.display_name,
                    base_model.id,
                )
                raise ConfigValidationError(
                    f"Invalid fine-tuned model '{base_model.display_name}' (id: '{base_model.id}'): only fine tune model {AQUA_FINE_TUNE_MODEL_VERSION} is supported for model deployment. "
                    f"Run 'ads aqua model convert_fine_tune --model_id {base_model.id}' to convert legacy AQUA fine tuned model to version {AQUA_FINE_TUNE_MODEL_VERSION} for deployment."
                )

            include_base_model_artifact = base_model.custom_metadata_list.get(
                FineTuneCustomMetadata.FINE_TUNE_INCLUDE_BASE_MODEL_ARTIFACT,
                ModelCustomMetadataItem(
                    key=FineTuneCustomMetadata.FINE_TUNE_INCLUDE_BASE_MODEL_ARTIFACT
                ),
            ).value

            if include_base_model_artifact == INCLUDE_BASE_MODEL:
                logger.error(
                    "Validation failed: Fine-tuned model '%s' (id: '%s') is not supported for model deployment.",
                    base_model.display_name,
                    base_model.id,
                )
                raise ConfigValidationError(
                    f"Invalid fine-tuned model '{base_model.display_name}' (id: '{base_model.id}'): for fine tuned models like Phi4, the deployment is not supported. "
                )

    def validate_base_model(self, model_id: str) -> Union[str, AquaMultiModelRef]:
        """
        Validates the input base model for single model deployment configuration.

        Validation Criteria:
        - Legacy fine-tuned models will be deployed as single model deployment.
        - Fine-tuned models v2 will be deployed as stacked deployment.

        Parameters
        ----------
        model_id : str
            The OCID of DataScienceModel instance.

        Returns
        -------
        Union[str, AquaMultiModelRef]
            A string of model id or an instance of AquaMultiModelRef.

        Raises
        ------
        ConfigValidationError
            If any of the above conditions are violated.
        """
        base_model = DataScienceModel.from_id(model_id)
        freeform_tags = base_model.freeform_tags
        aqua_fine_tuned_model = freeform_tags.get(
            Tags.AQUA_FINE_TUNED_MODEL_TAG, UNKNOWN
        )
        if aqua_fine_tuned_model:
            fine_tuned_model_version = freeform_tags.get(
                Tags.AQUA_FINE_TUNE_MODEL_VERSION, UNKNOWN
            )
            # TODO: revisit to block deploying single fine tuned model after AQUA UI is integrated.
            if fine_tuned_model_version.lower() == AQUA_FINE_TUNE_MODEL_VERSION:
                # extracts base model id from tag 'aqua_fine_tuned_model' and builds AquaMultiModelRef instance for stacked deployment.
                logger.debug(
                    f"Detected base model is fine-tuned model {AQUA_FINE_TUNE_MODEL_VERSION} and switched to stack deployment."
                )
                segments = aqua_fine_tuned_model.split("#")
                if not segments or not is_valid_ocid(segments[0]):
                    logger.error(
                        "Validation failed: Fine-tuned model '%s' (id: '%s') is not supported for model deployment.",
                        base_model.display_name,
                        base_model.id,
                    )
                    raise ConfigValidationError(
                        f"Invalid fine-tuned model '{base_model.display_name}' (id: '{base_model.id}'): missing or invalid tag '{Tags.AQUA_FINE_TUNED_MODEL_TAG}' format. "
                        f"Make sure tag '{Tags.AQUA_FINE_TUNED_MODEL_TAG}' is added with format <service_model_id>#<service_model_name>."
                    )
                # reset the model_id and models in create_model_deployment_details for stack deployment
                self.model_id = None
                self.models = [
                    AquaMultiModelRef(
                        model_id=segments[0],
                        model_name=segments[1],
                        fine_tune_weights=[
                            LoraModuleSpec(
                                model_id=base_model.id,
                                model_name=base_model.display_name,
                            )
                        ],
                    )
                ]
                return self.models[0]

        return model_id

    class Config:
        extra = "allow"
        protected_namespaces = ()


class CreateModelDeploymentDetails(ModelDeploymentDetails):
    """Class for creating Aqua model deployments."""

    instance_shape: str = Field(
        ..., description="The instance shape used for deployment."
    )
    compartment_id: Optional[str] = Field(None, description="The compartment OCID.")
    project_id: Optional[str] = Field(None, description="The project OCID.")
    model_id: Optional[str] = Field(None, description="The model OCID to deploy.")
    model_name: Optional[str] = Field(
        None, description="The model name specified by user to deploy."
    )
    server_port: Optional[int] = Field(
        None, description="Server port for the Docker container image."
    )
    health_check_port: Optional[int] = Field(
        None, description="Health check port for the Docker container image."
    )
    env_var: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Environment variables for deployment."
    )
    container_family: Optional[str] = Field(
        None, description="Image family of the model deployment container runtime."
    )
    model_file: Optional[str] = Field(
        None, description="File used for model deployment."
    )
    container_image_uri: Optional[str] = Field(
        None,
        description="Image URI for model deployment container runtime "
        "(ignored for service-managed containers). "
        "Required parameter for BYOC based deployments if this parameter was not set during "
        "model registration.",
    )
    private_endpoint_id: Optional[str] = Field(
        None, description="Private endpoint ID for model deployment."
    )
    cmd_var: Optional[List[str]] = Field(
        None, description="Command variables for the container runtime."
    )
    deployment_type: Optional[str] = Field(
        None, description="The type of model deployment."
    )
    subnet_id: Optional[str] = Field(
        None, description="The custom egress for model deployment."
    )
    capacity_reservation_ids: Optional[List[str]] = Field(
        None,
        description="List of capacity reservation OCIDs for deploying on reserved capacity.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate(cls, values: Any) -> Any:
        """Ensures exactly one of `model_id` or `models` is provided."""
        model_id = values.get("model_id")
        models = values.get("models")
        if bool(model_id) == bool(models):  # Both set or both unset
            raise ValueError(
                "Exactly one of `model_id` or `models` must be provided to create a model deployment."
            )

        # Handle backward compatibility: extract CAPACITY_RESERVATION_ID from env_var
        # and convert to capacity_reservation_ids if not already set
        env_var = values.get("env_var") or {}
        capacity_reservation_ids = values.get("capacity_reservation_ids")

        legacy_capacity_reservation_id = env_var.pop("CAPACITY_RESERVATION_ID", None)
        if not capacity_reservation_ids and legacy_capacity_reservation_id:
            values["capacity_reservation_ids"] = [legacy_capacity_reservation_id]
            logger.warning(
                "CAPACITY_RESERVATION_ID environment variable is deprecated. "
                "Use 'capacity_reservation_ids' parameter instead for native SDK support."
            )

        return values

    class Config:
        extra = "allow"
        protected_namespaces = ()


class UpdateModelDeploymentDetails(ModelDeploymentDetails):
    """Class for updating Aqua model deployments."""

    class Config:
        # forbid any other parameter for updating model group deployment
        extra = "forbid"
        protected_namespaces = ()
