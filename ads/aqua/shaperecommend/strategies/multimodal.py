#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Multimodal (Vision-Language Model) recommendation strategy.

Handles models like LLaVA, Qwen2-VL, Nemotron-VL, InternVL, etc.
Combines text+vision estimators and adds multimodal-specific vLLM flags.
"""

from typing import List

from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.common.errors import AquaValueError
from ads.aqua.shaperecommend.constants import (
    BITS_AND_BYTES_4BIT,
    BITSANDBYTES,
    TROUBLESHOOT_MSG,
    VLLM_PARAMS,
)
from ads.aqua.shaperecommend.estimator import (
    VisionMemoryEstimator,
    get_estimator,
)
from ads.aqua.shaperecommend.llm_config import ParsedModelConfig
from ads.aqua.shaperecommend.shape_report import (
    DeploymentParams,
    ModelConfig,
    ModelDetail,
    ShapeRecommendationReport,
    ShapeReport,
)
from ads.aqua.shaperecommend.strategies.base import RecommendationStrategy


class MultimodalStrategy(RecommendationStrategy):
    """
    Strategy for multimodal (vision-language) models.
    
    Combines text and vision estimators, adds image token overhead,
    and appends multimodal-specific vLLM flags:
    - --limit-mm-per-prompt {"image": 1}
    - --trust-remote-code (if auto_map present)
    """

    def recommend(
        self,
        parsed_config: ParsedModelConfig,
        shapes: List[ComputeShapeSummary],
        model_name: str,
        batch_size: int = 1,
    ) -> ShapeRecommendationReport:
        """Generate recommendations for multimodal models."""
        if not parsed_config.llm_config:
            raise AquaValueError(
                "MultimodalStrategy requires llm_config in ParsedModelConfig."
            )

        llm_config = parsed_config.llm_config
        vision_config = parsed_config.vision_config

        recommendations = []

        if not shapes:
            raise AquaValueError(
                "No GPU shapes were passed for recommendation."
            )

        # Calculate vision model memory overhead (if vision_config present)
        vision_memory_gb = 0.0
        image_token_count = 0
        if vision_config:
            vision_estimator = VisionMemoryEstimator(vision_config=vision_config)
            vision_memory_gb = vision_estimator.model_memory
            image_token_count = vision_estimator.image_token_count()

        # Pre-quantized case
        if llm_config.quantization_type:
            deployment_config = llm_config.calculate_possible_seq_len()
            for shape in shapes:
                shape_quantization = set(shape.gpu_specs.quantization)
                if llm_config.quantization_type in shape_quantization:
                    allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs
                    for max_seq_len in deployment_config:
                        # Account for image tokens reducing available text token budget
                        effective_seq_len = max(2048, max_seq_len - image_token_count)
                        estimator = get_estimator(
                            llm_config=llm_config,
                            seq_len=effective_seq_len,
                            batch_size=batch_size,
                        )
                        total_memory = estimator.total_memory + vision_memory_gb
                        if (allowed_gpu_memory * 0.9) > total_memory:
                            # Build custom ModelConfig for multimodal
                            best_config = [
                                self._build_multimodal_config(
                                    estimator, vision_memory_gb, allowed_gpu_memory
                                )
                            ]
                            recommendations.append(
                                ShapeReport(
                                    shape_details=shape, configurations=best_config
                                )
                            )
                            break

        # Unquantized case
        else:
            deployment_config = llm_config.optimal_config()
            prev_quant = None
            for shape in shapes:
                shape_quantization = set(shape.gpu_specs.quantization)
                allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs
                for quantization, max_seq_len in deployment_config:
                    if (
                        quantization == BITS_AND_BYTES_4BIT
                        and BITSANDBYTES not in shape_quantization
                    ):
                        continue
                    if quantization != prev_quant:
                        updated_config = llm_config.model_copy(
                            update={"in_flight_quantization": quantization}
                        )
                        prev_quant = quantization
                    
                    effective_seq_len = max(2048, max_seq_len - image_token_count)
                    estimator = get_estimator(
                        llm_config=updated_config,
                        seq_len=effective_seq_len,
                        batch_size=batch_size,
                    )
                    total_memory = estimator.total_memory + vision_memory_gb
                    if (allowed_gpu_memory * 0.9) > total_memory:
                        best_config = [
                            self._build_multimodal_config(
                                estimator, vision_memory_gb, allowed_gpu_memory
                            )
                        ]
                        recommendations.append(
                            ShapeReport(shape_details=shape, configurations=best_config)
                        )
                        break

        troubleshoot_msg = ""

        if len(recommendations) > 2:
            recommendations = ShapeReport.pareto_front(recommendations)

        if not recommendations:
            troubleshoot_msg += TROUBLESHOOT_MSG

            largest_shapes = (
                [(shapes[0], "fp8", False), (shapes[1], "4bit", True)]
                if len(shapes) > 1
                else []
            )

            for shape, quantization, in_flight in largest_shapes:
                if in_flight:
                    updated_config = llm_config.model_copy(
                        update={"in_flight_quantization": quantization}
                    )
                else:
                    updated_config = llm_config.model_copy(
                        update={"quantization": quantization}
                    )
                estimator = get_estimator(
                    llm_config=updated_config, seq_len=2048, batch_size=batch_size
                )
                allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs * 0.9
                best_config = [
                    self._build_multimodal_config(
                        estimator, vision_memory_gb, allowed_gpu_memory
                    )
                ]
                recommendations.append(
                    ShapeReport(shape_details=shape, configurations=best_config)
                )

        return ShapeRecommendationReport(
            display_name=model_name,
            recommendations=recommendations,
            troubleshoot=troubleshoot_msg,
        )

    def _build_multimodal_config(
        self, estimator, vision_memory_gb: float, allowed_gpu_memory: float
    ) -> ModelConfig:
        """
        Build a ModelConfig with multimodal-specific deployment params.
        
        Adds:
        - --limit-mm-per-prompt {"image": 1}
        - --trust-remote-code (if needed)
        - --enforce-eager (recommended for VLMs)
        """
        c = estimator.llm_config
        params = []
        
        # Standard vLLM params
        if estimator.seq_len < c.max_seq_len:
            params.append(VLLM_PARAMS["max_model_len"])
            params.append(str(estimator.seq_len))

        if not c.quantization and c.in_flight_quantization == "4bit":
            params.append(VLLM_PARAMS["in_flight_quant"])

        # Multimodal-specific params
        params.append(VLLM_PARAMS["limit_mm_per_prompt_image"])
        params.append(VLLM_PARAMS["enforce_eager"])
        
        if c.trust_remote_code:
            params.append(VLLM_PARAMS["trust_remote_code"])

        deployment_params = DeploymentParams(
            quantization=c.quantization or c.in_flight_quantization or c.weight_dtype,
            max_model_len=estimator.seq_len,
            params=" ".join(params) if params else "",
            weight_dtype=c.weight_dtype,
            env_var={},
        )

        model_detail = ModelDetail(
            model_size_gb=round(estimator.model_memory + vision_memory_gb, 2),
            kv_cache_size_gb=round(estimator.kv_cache_memory, 2),
            total_model_gb=round(estimator.total_memory + vision_memory_gb, 2),
        )

        return ModelConfig(
            model_details=model_detail,
            deployment_params=deployment_params,
            recommendation=estimator.limiting_factor(allowed_gpu_memory),
        )
