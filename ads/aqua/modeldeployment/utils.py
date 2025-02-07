#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""AQUA model deployment utils"""

import copy
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from ads.aqua.app import AquaApp
from ads.aqua.common.errors import AquaValueError
from ads.aqua.modeldeployment.entities import (
    AquaDeploymentConfig,
    GPUModelAllocation,
    GPUShapeAllocation,
    ModelDeploymentConfigSummary,
)
from ads.config import AQUA_MODEL_DEPLOYMENT_CONFIG


class MultiModelDeploymentConfigLoader:
    """
    Processes multiple model deployment configurations to determine compatible GPU shapes
    and calculate optimal GPU allocations.
    """

    MAX_WORKERS = 10

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
        self, model_ids: List[str], primary_model_id: Optional[str] = None
    ) -> ModelDeploymentConfigSummary:
        """
        Retrieves deployment configurations for multiple models and calculates compatible GPU allocations.

        Parameters
        ----------
        model_ids : List[str]
            A list of OCIDs for the Aqua models.
        primary_model_id : Optional[str], optional
            The OCID of the primary Aqua model. If provided, GPU allocation prioritizes this model.
            Otherwise, GPUs are evenly allocated.

        Returns
        -------
        ModelDeploymentConfigSummary
            A summary of the deployment configurations and GPU allocations.

        Raises
        ------
        AquaValueError
            If no compatible shapes or GPU allocations are available.
        """
        deployment_configs = self._fetch_deployment_configs_concurrently(model_ids)
        model_shape_gpu, deployment = self._extract_model_shape_gpu(deployment_configs)

        common_shapes = self._get_common_shapes(model_shape_gpu)
        if not common_shapes:
            raise AquaValueError(
                "No available shapes for selected models. Choose a different model."
            )

        gpu_allocation = self._compute_gpu_allocation(
            common_shapes, model_shape_gpu, primary_model_id
        )
        if not gpu_allocation:
            raise AquaValueError(
                "No available GPU allocations. Choose a different model."
            )

        return ModelDeploymentConfigSummary(
            deployment_config=deployment, gpu_allocation=gpu_allocation
        )

    def _fetch_deployment_configs_concurrently(
        self, model_ids: List[str]
    ) -> Dict[str, AquaDeploymentConfig]:
        """Fetches deployment configurations in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            results = executor.map(
                lambda model_id: self.deployment_app.get_config(
                    model_id, AQUA_MODEL_DEPLOYMENT_CONFIG
                ),
                model_ids,
            )

        return {
            model_id: AquaDeploymentConfig(**config)
            for model_id, config in zip(model_ids, results)
        }

    def _extract_model_shape_gpu(
        self, deployment_configs: Dict[str, AquaDeploymentConfig]
    ):
        """Extracts shape and GPU count details from deployment configurations."""
        model_shape_gpu = {}
        deployment = {}

        for model_id, config in deployment_configs.items():
            model_shape_gpu[model_id] = {
                shape: [
                    item.gpu_count
                    for item in config.configuration[shape].multi_model_deployment
                ]
                for shape in config.shape
            }
            deployment[model_id] = {
                "shape": config.shape,
                "configuration": {
                    shape: config.configuration[shape] for shape in config.shape
                },
            }

        return model_shape_gpu, deployment

    def _get_common_shapes(
        self, model_shape_gpu: Dict[str, Dict[str, List[int]]]
    ) -> List[str]:
        """Finds common shapes across all models."""
        return list(
            set.intersection(
                *(set(shapes.keys()) for shapes in model_shape_gpu.values())
            )
        )

    def _compute_gpu_allocation(
        self,
        common_shapes: List[str],
        model_shape_gpu: Dict[str, Dict[str, List[int]]],
        primary_model_id: Optional[str],
    ) -> Dict[str, GPUShapeAllocation]:
        """Computes GPU allocation for common shapes."""
        gpu_allocation = {}

        for common_shape in common_shapes:
            model_gpu = {
                model: shape_gpu[common_shape]
                for model, shape_gpu in model_shape_gpu.items()
            }
            is_compatible, max_gpu_count, combination = self._verify_compatibility(
                model_gpu, primary_model_id
            )

            if is_compatible:
                gpu_allocation[common_shape] = GPUShapeAllocation(
                    models=combination, total_gpus_available=max_gpu_count
                )

        return gpu_allocation

    def _verify_compatibility(
        self, model_gpu_dict: Dict, primary_model_id: str = None
    ) -> tuple:
        """Calculates the gpu allocations for all compatible shapes.
        If no primary Aqua model id provided, gpu count for each compatible shape will be evenly allocated.
        If provided, gpu count for each compatible shape will be prioritized for primary model.

        For example, there is one compatible shape "BM.GPU.H100.8" for three models A, B, C, and each model has a gpu count as below:

        A - BM.GPU.H100.8 - 1, 2, 4, 8
        B - BM.GPU.H100.8 - 1, 2, 4, 8
        C - BM.GPU.H100.8 - 1, 2, 4, 8

        If no primary model is provided, the gpu allocation for A, B, C could be [2, 4, 2], [2, 2, 4] or [4, 2, 2]
        If B is the primary model, the gpu allocation is [2, 4, 2] as B always gets the maximum gpu count.

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
        maximum_gpu_count = max([sorted(gpus)[-1] for gpus in model_gpu_dict.values()])
        model_gpu_dict_copy = copy.deepcopy(model_gpu_dict)
        if primary_model_id:
            primary_model_gpu_list = sorted(model_gpu_dict_copy.pop(primary_model_id))
            for gpu_count in reversed(primary_model_gpu_list):
                combinations = self.get_combinations(model_gpu_dict_copy)
                for combination in combinations:
                    if (
                        len(combination) == len(model_gpu_dict_copy)
                        and sum(combination.values()) == maximum_gpu_count - gpu_count
                    ):
                        combination[primary_model_id] = gpu_count
                        return (
                            True,
                            maximum_gpu_count,
                            [
                                GPUModelAllocation(ocid=ocid, gpu_count=gpu_count)
                                for ocid, gpu_count in combination.items()
                            ],
                        )

        else:
            combinations = self.get_combinations(model_gpu_dict_copy)
            minimal_difference = float("inf")  # gets the positive infinity
            optimal_combination = []
            for combination in combinations:
                if (
                    len(combination) == len(model_gpu_dict_copy)
                    and sum(combination.values()) == maximum_gpu_count
                ):
                    difference = max(combination.values()) - min(combination.values())
                    if difference < minimal_difference:
                        minimal_difference = difference
                        optimal_combination = combination

                        # find the optimal combination, no need to continue
                        if minimal_difference == 0:
                            break

            if optimal_combination:
                return (
                    True,
                    maximum_gpu_count,
                    [
                        GPUModelAllocation(ocid=ocid, gpu_count=gpu_count)
                        for ocid, gpu_count in optimal_combination.items()
                    ],
                )

        return (False, 0, [])

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
