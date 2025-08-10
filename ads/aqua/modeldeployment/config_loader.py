#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import copy
import itertools
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from pydantic import Field

from ads.aqua import logger
from ads.aqua.app import AquaApp
from ads.aqua.common.entities import ComputeShapeSummary, ModelConfigResult
from ads.aqua.common.enums import Tags
from ads.aqua.config.utils.serializer import Serializable
from ads.aqua.finetuning.constants import FineTuneCustomMetadata
from ads.aqua.model.constants import AquaModelMetadataKeys
from ads.config import AQUA_MODEL_DEPLOYMENT_CONFIG


class ShapeInfoConfig(Serializable):
    """Describes how many memory and cpu to this model for specific shape.

    Attributes:
        memory_in_gbs (float, optional): The number of memory in gbs to this model of the shape.
        ocpu (float, optional): The number of ocpus to this model of the shape.
    """

    memory_in_gbs: Optional[float] = Field(
        None,
        description="The number of memory in gbs to this model of the shape.",
    )
    ocpu: Optional[float] = Field(
        None,
        description="The number of ocpus to this model of the shape.",
    )

    class Config:
        extra = "allow"


class DeploymentShapeInfo(Serializable):
    """Describes the shape information to this model for specific shape.

    Attributes:
        configs (List[ShapeInfoConfig], optional): A list of memory and cpu number details to this model of the shape.
        type (str, optional): The type of the shape.
    """

    configs: Optional[List[ShapeInfoConfig]] = Field(
        default_factory=list,
        description="A list of memory and cpu number details to this model of the shape.",
    )
    type: Optional[str] = Field(
        default_factory=str, description="The type of the shape."
    )

    class Config:
        extra = "allow"


class GPUModelAllocation(Serializable):
    """Describes how many GPUs are allocated to a particular model.

    Attributes:
        ocid (str, optional): The unique identifier of the model.
        gpu_count (int, optional): Number of GPUs allocated to this model.
    """

    ocid: Optional[str] = Field(
        default_factory=str, description="The unique model OCID."
    )
    gpu_count: Optional[int] = Field(
        default_factory=int, description="The number of GPUs allocated to the model."
    )

    class Config:
        extra = "allow"


class MultiModelConfig(Serializable):
    """Describes how many GPUs and the parameters of specific shape for multi model deployment.

    Attributes:
        gpu_count (int, optional): Number of GPUs count to this model of this shape.
        parameters (Dict[str, str], optional): A dictionary of parameters (e.g., VLLM_PARAMS) to
            configure the behavior of a particular GPU shape.
        env (Dict[str, Dict[str, str]]): Environment variables grouped by namespace (e.g., "VLLM": {"VAR": "VAL"}).
    """

    gpu_count: Optional[int] = Field(
        default_factory=int, description="The number of GPUs allocated to the model."
    )
    parameters: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Key-value pairs for GPU shape parameters (e.g., VLLM_PARAMS).",
    )
    env: Optional[Dict[str, Dict[str, str]]] = Field(
        default_factory=dict,
        description="Environment variables grouped by namespace",
    )

    class Config:
        extra = "allow"


class GPUShapeAllocation(Serializable):
    """
    Allocation details for a specific GPU shape.

    Attributes:
        models (List[GPUModelAllocation], optional): List of model GPU allocations for this shape.
        total_gpus_available (int, optional): The total number of GPUs available for this shape.
    """

    models: Optional[List[GPUModelAllocation]] = Field(
        default_factory=list, description="List of model allocations for this shape."
    )
    total_gpus_available: Optional[int] = Field(
        default_factory=int, description="Total GPUs available for this shape."
    )

    class Config:
        extra = "allow"


class ConfigurationItem(Serializable):
    """Holds key-value parameter pairs for a specific GPU or CPU shape.

    Attributes:
        parameters (Dict[str, str], optional): A dictionary of parameters (e.g., VLLM_PARAMS) to
            configure the behavior of a particular GPU shape.
        multi_model_deployment (List[MultiModelConfig], optional): A list of multi model configuration details.
        shape_info (DeploymentShapeInfo, optional): The shape information to this model for specific CPU shape.
        env (Dict[str, Dict[str, str]]): Environment variables grouped by namespace (e.g., "VLLM": {"VAR": "VAL"}).
    """

    parameters: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Key-value pairs for shape parameters.",
    )
    multi_model_deployment: Optional[List[MultiModelConfig]] = Field(
        default_factory=list, description="A list of multi model configuration details."
    )
    shape_info: Optional[DeploymentShapeInfo] = Field(
        default_factory=DeploymentShapeInfo,
        description="The shape information to this model for specific shape",
    )
    env: Optional[Dict[str, Dict[str, str]]] = Field(
        default_factory=dict,
        description="Environment variables grouped by namespace",
    )

    class Config:
        extra = "allow"


class AquaDeploymentConfig(Serializable):
    """Represents multi model's shape list and detailed configuration.

    Attributes:
        shape (List[str], optional): A list of shape names (e.g., BM.GPU.A10.4).
        configuration (Dict[str, ConfigurationItem], optional): Maps each shape to its configuration details.
    """

    shape: Optional[List[str]] = Field(
        default_factory=list, description="List of supported shapes for the model."
    )
    configuration: Optional[Dict[str, ConfigurationItem]] = Field(
        default_factory=dict, description="Configuration details keyed by shape."
    )

    class Config:
        extra = "allow"


class ModelDeploymentConfigSummary(Serializable):
    """Top-level configuration model for OCI-based deployments.

    Attributes:
        deployment_config (Dict[str, AquaDeploymentConfig], optional): Deployment configurations
            keyed by model OCID.
        gpu_allocation (Dict[str, GPUShapeAllocation], optional): GPU allocations keyed by GPU shape.
        error_message (str, optional): Error message if GPU allocation is not possible.
    """

    deployment_config: Optional[Dict[str, AquaDeploymentConfig]] = Field(
        default_factory=dict,
        description=(
            "Deployment configuration details for each model, including supported shapes "
            "and shape-specific parameters."
        ),
    )
    gpu_allocation: Optional[Dict[str, GPUShapeAllocation]] = Field(
        default_factory=dict,
        description=(
            "Details on how GPUs are allocated per shape, including the total "
            "GPUs available for each shape."
        ),
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if GPU allocation is not possible."
    )

    class Config:
        extra = "allow"
        protected_namespaces = ()


class MultiModelDeploymentConfigLoader:
    """
    Processes multiple model deployment configurations to determine compatible GPU shapes
    and calculate optimal GPU allocations.
    """

    MAX_WORKERS = 10  # Number of workers for asynchronous models detail loading

    def __init__(self, deployment_app: AquaApp):
        """
        Initializes the processor with a reference to the `AquaDeploymentApp` to fetch model configurations.

        Parameters
        ----------
        deployment_app : AquaDeploymentApp
            An instance of AquaDeploymentApp used to fetch model deployment configurations.
        """
        self.deployment_app = deployment_app

    def load(
        self,
        shapes: List[ComputeShapeSummary],
        model_ids: List[str],
        primary_model_id: Optional[str] = None,
    ) -> ModelDeploymentConfigSummary:
        """
        Retrieves deployment configurations for multiple/single model and calculates compatible GPU allocations.

        Parameters
        ----------
        shapes : List[ComputeShapeSummary]
            Model deployment available shapes.
        model_ids : List[str]
            A list of OCIDs for the Aqua models.
        primary_model_id : Optional[str], optional
            The OCID of the primary Aqua model. If provided, GPU allocation prioritizes this model.
            Otherwise, GPUs are evenly allocated.

        Returns
        -------
        ModelDeploymentConfigSummary
            A summary of the deployment configurations and GPU allocations. If GPU allocation
            cannot be determined, an appropriate error message is included in the summary.
        """
        return self._load_multi_model_deployment_configuration(
            shapes=shapes, model_ids=model_ids, primary_model_id=primary_model_id
        )

    def _load_multi_model_deployment_configuration(
        self,
        shapes: List[ComputeShapeSummary],
        model_ids: List[str],
        primary_model_id: Optional[str] = None,
    ) -> ModelDeploymentConfigSummary:
        """
        Retrieves deployment configurations for multiple models and calculates compatible GPU allocations.

        Parameters
        ----------
        shapes : List[ComputeShapeSummary]
            Model deployment available shapes.
        model_ids : List[str]
            A list of OCIDs for the Aqua models.
        primary_model_id : Optional[str], optional
            The OCID of the primary Aqua model. If provided, GPU allocation prioritizes this model.
            Otherwise, GPUs are evenly allocated.

        Returns
        -------
        ModelDeploymentConfigSummary
            A summary of the deployment configurations and GPU allocations. If GPU allocation
            cannot be determined, an appropriate error message is included in the summary.
        """
        model_shape_gpu, available_shapes, summary = self._fetch_model_shape_gpu(
            shapes=shapes, model_ids=model_ids
        )

        # Identify common deployment shapes among all models.
        common_shapes, empty_configs = self._get_common_shapes(model_shape_gpu)
        logger.debug(f"Common Shapes: {common_shapes} from: {model_shape_gpu}")

        # If all models' shape configs are empty, use default deployment shapes instead
        common_shapes = (
            available_shapes
            if empty_configs
            else [
                shape_name
                for shape_name in common_shapes
                if shape_name.upper() in available_shapes
            ]
        )
        logger.debug(f"Available Common Shapes: {common_shapes}")

        if not common_shapes:
            summary.error_message = (
                "The selected models do not share any available common deployment shapes. "
                "Please ensure that all chosen models are compatible for multi-model deployment."
            )
            logger.debug(
                f"No common deployment shapes found among selected models: {model_ids}"
            )
            return summary

        # Compute GPU allocations based on the common shapes and optionally prioritize a primary model.
        gpu_allocation = self._compute_gpu_allocation(
            shapes=shapes,
            common_shapes=common_shapes,
            model_shape_gpu=model_shape_gpu,
            primary_model_id=primary_model_id,
        )

        logger.debug(f"GPU Allocation: {gpu_allocation}")

        if not gpu_allocation:
            summary.error_message = (
                "The selected models do not have a valid GPU allocation based on their current configurations. "
                "Please select a different model group. If you are deploying custom models that lack AQUA service configuration, "
                "refer to the deployment guidelines here: "
                "https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/multimodel-deployment-tips.md#custom_models"
            )

            logger.debug(
                f"GPU allocation computation failed for selected models: {model_ids}"
            )

            return summary

        summary.gpu_allocation = gpu_allocation
        return summary

    def _fetch_model_shape_gpu(
        self, shapes: List[ComputeShapeSummary], model_ids: List[str]
    ):
        """Fetches dict of model shape and gpu, list of available shapes and builds `ModelDeploymentConfigSummary` instance."""
        # Fetch deployment configurations concurrently.
        logger.debug(f"Loading model deployment configuration for models: {model_ids}")
        deployment_configs = self._fetch_deployment_configs_concurrently(model_ids)

        logger.debug(f"Loaded config: {deployment_configs}")
        model_shape_gpu, deployment = self._extract_model_shape_gpu(
            deployment_configs=deployment_configs, shapes=shapes
        )

        # Initialize the summary result with the deployment configurations.
        summary = ModelDeploymentConfigSummary(deployment_config=deployment)

        # Filter out not available shapes
        available_shapes = [item.name.upper() for item in shapes]
        logger.debug(f"Service Available Shapes: {available_shapes}")

        return model_shape_gpu, available_shapes, summary

    def _fetch_deployment_configs_concurrently(
        self, model_ids: List[str]
    ) -> Dict[str, AquaDeploymentConfig]:
        """Fetches deployment configurations in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            results = executor.map(
                self._fetch_deployment_config_from_metadata_and_oss,
                model_ids,
            )

        return {
            model_id: AquaDeploymentConfig(**config.config)
            for model_id, config in zip(model_ids, results)
        }

    def _fetch_deployment_config_from_metadata_and_oss(
        self, model_id: str
    ) -> ModelConfigResult:
        """
        Retrieves the deployment configuration for a model.

        The method first attempts to retrieve the configuration directly from the model itself
        via metadata or object storage. If not found and the model is identified as a fine-tuned
        model, it then attempts to retrieve the configuration from the associated base model.

        Sources are checked in the following order:
            1. Model metadata
            2. Object Storage
            3. (If fine-tuned) Base model metadata
            4. (If fine-tuned) Base model Object Storage

        Parameters
        ----------
        model_id : str
            OCID of the model in the Model Catalog.

        Returns
        -------
        ModelConfigResult
            A result object containing the deployment configuration and model details.
            If no config is found, `config` will be an empty dictionary.
        """
        # Try to get config from the model itself
        logger.info(
            "Attempting to retrieve config for model '%s' from metadata.", model_id
        )
        config = self.deployment_app.get_config_from_metadata(
            model_id=model_id,
            metadata_key=AquaModelMetadataKeys.DEPLOYMENT_CONFIGURATION,
        )
        if config and config.config:
            logger.info(
                "Successfully retrieved deployment config from metadata for model '%s'.",
                model_id,
            )
            return config

        logger.info(
            "Config not found in metadata. Trying Object Storage for model '%s'.",
            model_id,
        )
        config = self.deployment_app.get_config(
            model_id=model_id,
            config_file_name=AQUA_MODEL_DEPLOYMENT_CONFIG,
        )
        if config and config.config:
            logger.info(
                "Successfully retrieved deployment config from Object Storage for model '%s'.",
                model_id,
            )
            return config

        # If it's a fine-tuned model, try base model
        oci_model = self.deployment_app.ds_client.get_model(model_id).data
        is_fine_tuned_model = Tags.AQUA_FINE_TUNED_MODEL_TAG in oci_model.freeform_tags
        if not is_fine_tuned_model:
            logger.warning("No deployment config found for model '%s'.", model_id)
            return ModelConfigResult(config={}, model_details=oci_model)

        logger.info(
            "Model '%s' is a fine-tuned model. Attempting to retrieve base model ID.",
            model_id,
        )
        base_model_id = next(
            (
                item.value
                for item in oci_model.custom_metadata_list
                if item.key == FineTuneCustomMetadata.FINE_TUNE_SOURCE
            ),
            None,
        )

        if not base_model_id:
            logger.warning(
                "Base model reference not found in custom metadata for fine-tuned model '%s'.",
                model_id,
            )
            return ModelConfigResult(config={}, model_details=oci_model)

        logger.info(
            "Using base model '%s' to retrieve deployment config.", base_model_id
        )

        config = self.deployment_app.get_config_from_metadata(
            model_id=base_model_id,
            metadata_key=AquaModelMetadataKeys.DEPLOYMENT_CONFIGURATION,
        )
        if config and config.config:
            logger.info(
                "Successfully retrieved deployment config from base model metadata '%s'.",
                base_model_id,
            )
            return config

        config = self.deployment_app.get_config(
            model_id=base_model_id,
            config_file_name=AQUA_MODEL_DEPLOYMENT_CONFIG,
        )
        if config and config.config:
            logger.info(
                "Successfully retrieved deployment config from base model Object Storage '%s'.",
                base_model_id,
            )
            return config

        logger.warning(
            "Deployment configuration could not be found for model '%s' or its base model '%s'.",
            model_id,
            base_model_id,
        )
        return ModelConfigResult(config={}, model_details=oci_model)

    def _extract_model_shape_gpu(
        self,
        deployment_configs: Dict[str, AquaDeploymentConfig],
        shapes: List[ComputeShapeSummary],
    ):
        """Extracts shape and GPU count details from deployment configurations.
        Supported shapes for multi model deployment will be collected from `configuration` entry in deployment config.
        Supported shapes for single model deployment will be collected from `shape` entry in deployment config.
        """
        model_shape_gpu = {}
        deployment = {}
        is_single_model = len(deployment_configs) == 1

        for model_id, config in deployment_configs.items():
            # For multi model deployment, we cannot rely on .shape because some models, like Falcon-7B, can only be deployed on a single GPU card (A10.1).
            # However, Falcon can also be deployed on a single card in other A10 shapes, such as A10.2.
            # Our current configuration does not support this flexibility.
            # For single model deployment, we use `config.shape` to find the available shapes.
            multi_deployment_shape = (
                list(set(config.configuration.keys()).union(set(config.shape or [])))
                if is_single_model
                else list(config.configuration.keys())
            )

            shape_total_gpus_available_map = {
                deployment_shape.name.upper(): deployment_shape.gpu_specs.gpu_count
                or None
                for deployment_shape in shapes
                if deployment_shape and deployment_shape.gpu_specs
            }

            model_shape_gpu[model_id] = {
                shape.upper(): [
                    item.gpu_count
                    for item in config.configuration.get(
                        shape,
                        ConfigurationItem(
                            multi_model_deployment=(
                                [
                                    MultiModelConfig(
                                        gpu_count=shape_total_gpus_available_map.get(
                                            shape.upper()
                                        )
                                    )
                                ]
                                if is_single_model
                                else []
                            )
                        ),
                    ).multi_model_deployment
                ]
                for shape in multi_deployment_shape
            }

            # For single-model deployments: if the shape is listed in the `shapes` section of the config,
            # we include the maximum available GPU count for that shape in the allocation consideration.
            if is_single_model:
                for shape in model_shape_gpu[model_id]:
                    shape_total_gpu_count = shape_total_gpus_available_map.get(
                        shape.upper()
                    )
                    if (
                        shape in config.shape
                        and shape_total_gpu_count
                        and shape_total_gpu_count
                        not in model_shape_gpu[model_id][shape]
                    ):
                        model_shape_gpu[model_id][shape].append(shape_total_gpu_count)

            deployment[model_id] = {
                "shape": [shape.upper() for shape in config.shape],
                "configuration": {
                    shape.upper(): config.configuration.get(shape, ConfigurationItem())
                    for shape in multi_deployment_shape
                },
            }

        return model_shape_gpu, deployment

    def _get_common_shapes(
        self, model_shape_gpu: Dict[str, Dict[str, List[int]]]
    ) -> tuple:
        """Finds common shapes across all models."""
        common_shapes_set = []
        empty_configs = True
        for shapes in model_shape_gpu.values():
            if shapes:
                common_shapes_set.append(set(shapes.keys()))
                empty_configs = False
        if not common_shapes_set:
            return [], empty_configs
        return list(set.intersection(*(common_shapes_set))), empty_configs

    def _compute_gpu_allocation(
        self,
        shapes: List[ComputeShapeSummary],
        common_shapes: List[str],
        model_shape_gpu: Dict[str, Dict[str, List[int]]],
        primary_model_id: Optional[str],
    ) -> Dict[str, GPUShapeAllocation]:
        """Computes GPU allocation for common shapes."""

        gpu_allocation = {}

        for common_shape in common_shapes:
            total_gpus_available = 0

            # search the shape in the available shapes list
            shape_summary = next(
                (shape for shape in shapes if shape.name.upper() == common_shape),
                None,
            )
            if shape_summary and shape_summary.gpu_specs:
                total_gpus_available = shape_summary.gpu_specs.gpu_count

            # generate a list of possible gpu count from `total_gpus_available` for custom models
            # without multi model deployment config
            # model_gpu = {
            #     model: (
            #         shape_gpu[common_shape]
            #         if shape_gpu.get(common_shape, UNKNOWN)
            #         else self._generate_gpu_list(total_gpus_available)
            #     )
            #     for model, shape_gpu in model_shape_gpu.items()
            # }

            model_gpu = {
                model: (shape_gpu.get(common_shape, []) or [])
                for model, shape_gpu in model_shape_gpu.items()
            }

            is_compatible, combination = self._verify_compatibility(
                total_gpus_available=total_gpus_available,
                model_gpu_dict=model_gpu,
                primary_model_id=primary_model_id,
            )

            if is_compatible:
                gpu_allocation[common_shape] = GPUShapeAllocation(
                    models=combination, total_gpus_available=total_gpus_available
                )

        return gpu_allocation

    @staticmethod
    def _generate_gpu_list(total_gpus_available: int) -> list[int]:
        """Generates a list of powers of 2 that's smaller than or equal to `total_gpus_available`.

        Example
        -------
        input: 8
        output: [1,2,4,8]

        Parameters
        ----------
        total_gpus_available : int
            Total GPU available

        Returns
        -------
        list
            A list of powers of 2.
        """
        if total_gpus_available < 1:
            return []
        return [2**i for i in range(int(math.log2(total_gpus_available)) + 1)]

    def _verify_compatibility(
        self,
        total_gpus_available: int,
        model_gpu_dict: Dict,
        primary_model_id: str = None,
    ) -> tuple:
        """Calculates the gpu allocations for all compatible shapes.
        If no primary Aqua model id provided, gpu count for each compatible shape will be evenly allocated.
        If provided, gpu count for each compatible shape will be prioritized for primary model.

        Example
        -------

        Case 1:
        There is one compatible shape "BM.GPU.H100.8" for three models A, B, C, and each model has a gpu count as below:

        A - BM.GPU.H100.8 - 1, 2, 4, 8
        B - BM.GPU.H100.8 - 1, 2, 4, 8
        C - BM.GPU.H100.8 - 1, 2, 4, 8

        If no primary model is provided, the gpu allocation for A, B, C could be [2, 4, 2], [2, 2, 4] or [4, 2, 2]
        If B is the primary model, the gpu allocation is [2, 4, 2] as B always gets the maximum gpu count.

        Case 2:
        There is one compatible shape "BM.GPU.H100.8" for three models A, B, C, and each model has a gpu count as below:

        A - BM.GPU.H100.8 - 1
        B - BM.GPU.H100.8 - 1, 2, 4
        C - BM.GPU.H100.8 - 1, 2, 4

        If no primary model is provided, the gpu allocation for A, B, C could be [1, 1, 2] or [1, 2, 1]
        If C is the primary model, the gpu allocation is [1, 1, 2] as C always gets the maximum gpu count.

        Parameters
        ----------
        model_gpu_dict: Dict
            A dict of Aqua model and its gpu counts.
        primary_model_id: str
            The OCID of the primary Aqua model

        Returns
        -------
        tuple:
            A tuple of gpu count allocation result.
        """
        model_gpu_dict_copy = copy.deepcopy(model_gpu_dict)
        # minimal gpu count needed to satisfy all models
        minimal_gpus_needed = len(model_gpu_dict)
        if primary_model_id and minimal_gpus_needed > 1:
            primary_model_gpu_list = sorted(model_gpu_dict_copy.pop(primary_model_id))
            primary_model_gpu_list.reverse()
            combinations = self.get_combinations(model_gpu_dict_copy)
            for gpu_count in primary_model_gpu_list:
                current_gpus_available = total_gpus_available
                while (
                    current_gpus_available >= minimal_gpus_needed
                    # or current_gpus_available == 1
                ):
                    for combination in combinations:
                        if (
                            len(combination) == len(model_gpu_dict_copy)
                            and sum(combination.values())
                            == current_gpus_available - gpu_count
                        ):
                            combination[primary_model_id] = gpu_count
                            return (
                                True,
                                [
                                    GPUModelAllocation(ocid=ocid, gpu_count=gpu_count)
                                    for ocid, gpu_count in combination.items()
                                ],
                            )

                    current_gpus_available -= 1
                    # current_gpus_available = (
                    #     1 if current_gpus_available == 0 else current_gpus_available
                    # )
        else:
            combinations = self.get_combinations(model_gpu_dict_copy)
            current_gpus_available = total_gpus_available
            while (
                current_gpus_available >= minimal_gpus_needed
                # or current_gpus_available == 1
            ):
                minimal_difference = float("inf")  # gets the positive infinity
                optimal_combination = []
                for combination in combinations:
                    if (
                        len(combination) == len(model_gpu_dict_copy)
                        and sum(combination.values()) == current_gpus_available
                    ):
                        difference = max(combination.values()) - min(
                            combination.values()
                        )
                        if difference < minimal_difference:
                            minimal_difference = difference
                            optimal_combination = combination

                            # find the optimal combination, no need to continue
                            if minimal_difference == 0:
                                break

                if optimal_combination:
                    return (
                        True,
                        [
                            GPUModelAllocation(ocid=ocid, gpu_count=gpu_count)
                            for ocid, gpu_count in optimal_combination.items()
                        ],
                    )

                current_gpus_available -= 1
                # current_gpus_available = (
                #     1 if current_gpus_available == 0 else current_gpus_available
                # )

        return (False, [])

    @staticmethod
    def get_combinations(input_dict: dict):
        """Finds all unique combinations within input dict.

        The input is a dict of {model:[gpu_count]} on a specific shape and this method will
        return a list of all unique combinations of gpu allocation of each model.

        For example:

        input: {'model_a': [2, 4], 'model_b': [1, 2, 4], 'model_c': [1, 2, 8]}
        output:
        [
            {'model_a': 2, 'model_b': 1, 'model_c': 1},
            {'model_a': 2, 'model_b': 1, 'model_c': 2},
            {'model_a': 2, 'model_b': 1, 'model_c': 8},
            {'model_a': 2, 'model_b': 2, 'model_c': 1},
            {'model_a': 2, 'model_b': 2, 'model_c': 2},
            {'model_a': 2, 'model_b': 2, 'model_c': 8},
            {'model_a': 2, 'model_b': 4, 'model_c': 1},
            {'model_a': 2, 'model_b': 4, 'model_c': 2},
            {'model_a': 2, 'model_b': 4, 'model_c': 8},
            {'model_a': 4, 'model_b': 1, 'model_c': 1},
            {'model_a': 4, 'model_b': 1, 'model_c': 2},
            {'model_a': 4, 'model_b': 1, 'model_c': 8},
            {'model_a': 4, 'model_b': 2, 'model_c': 1},
            {'model_a': 4, 'model_b': 2, 'model_c': 2},
            {'model_a': 4, 'model_b': 2, 'model_c': 8},
            {'model_a': 4, 'model_b': 4, 'model_c': 1},
            {'model_a': 4, 'model_b': 4, 'model_c': 2},
            {'model_a': 4, 'model_b': 4, 'model_c': 8}
        ]

        Parameters
        ----------
        input_dict: dict
            A dict of {model:[gpu_count]} on a specific shape

        Returns
        -------
        list:
            A list of all unique combinations of gpu allocation of each model.
        """
        keys, values = zip(*input_dict.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]
