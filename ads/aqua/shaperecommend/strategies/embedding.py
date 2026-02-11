#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Embedding model recommendation strategy.

Handles models like BERT, RoBERTa, E5-Mistral, GTE, etc.
Embedding models are typically small and throughput-sensitive rather than memory-bound.
"""

from typing import List

from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.common.errors import AquaValueError
from ads.aqua.shaperecommend.constants import VLLM_PARAMS
from ads.aqua.shaperecommend.estimator import EmbeddingMemoryEstimator
from ads.aqua.shaperecommend.llm_config import ParsedModelConfig
from ads.aqua.shaperecommend.shape_report import (
    DeploymentParams,
    ModelConfig,
    ModelDetail,
    ShapeRecommendationReport,
    ShapeReport,
)
from ads.aqua.shaperecommend.strategies.base import RecommendationStrategy


class EmbeddingStrategy(RecommendationStrategy):
    """
    Strategy for embedding models (BERT, RoBERTa, etc.).
    
    Embedding models:
    - Are typically small (< 1GB)
    - Have minimal KV cache during inference
    - Focus on throughput rather than sequence length
    - Require --task embedding flag for vLLM
    """

    def recommend(
        self,
        parsed_config: ParsedModelConfig,
        shapes: List[ComputeShapeSummary],
        model_name: str,
        batch_size: int = 1,
    ) -> ShapeRecommendationReport:
        """Generate recommendations for embedding models."""
        if not parsed_config.embedding_config:
            raise AquaValueError(
                "EmbeddingStrategy requires embedding_config in ParsedModelConfig."
            )

        embedding_config = parsed_config.embedding_config
        estimator = EmbeddingMemoryEstimator(embedding_config=embedding_config)

        recommendations = []

        if not shapes:
            raise AquaValueError("No GPU shapes were passed for recommendation.")

        # Embedding models are small - find all shapes that fit
        for shape in shapes:
            allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs
            if estimator.validate_shape(allowed_gpu_memory):
                model_config = self._build_embedding_config(
                    estimator, embedding_config, allowed_gpu_memory
                )
                recommendations.append(
                    ShapeReport(shape_details=shape, configurations=[model_config])
                )

        # Apply pareto front if too many recommendations
        if len(recommendations) > 3:
            recommendations = ShapeReport.pareto_front(recommendations)

        troubleshoot = ""
        if not recommendations:
            troubleshoot = (
                f"The embedding model ({estimator.total_memory:.2f}GB) "
                "is larger than expected. "
                "Embedding models are typically small (< 1GB). "
                "Please verify the model is a valid embedding model."
            )

        return ShapeRecommendationReport(
            display_name=model_name,
            recommendations=recommendations,
            troubleshoot=troubleshoot,
        )

    def _build_embedding_config(
        self, estimator: EmbeddingMemoryEstimator, config, allowed_gpu_memory: float
    ) -> ModelConfig:
        """
        Build ModelConfig for embedding models.
        
        Adds:
        - --task embedding
        - --max-model-len (if different from default 512)
        """
        params = [VLLM_PARAMS["task_embedding"]]

        # Override max_embed_len if needed
        if config.max_seq_len and config.max_seq_len != 512:
            params.append(VLLM_PARAMS["max_model_len"])
            params.append(str(config.max_seq_len))

        deployment_params = DeploymentParams(
            quantization=config.quantization or config.weight_dtype,
            max_model_len=config.max_seq_len,
            params=" ".join(params),
            weight_dtype=config.weight_dtype,
            env_var={},
        )

        model_detail = ModelDetail(
            model_size_gb=round(estimator.model_memory, 2),
            kv_cache_size_gb=0.0,  # Embedding models don't use KV cache
            total_model_gb=round(estimator.total_memory, 2),
        )

        # Simple recommendation message
        required = estimator.total_memory
        if required < allowed_gpu_memory * 0.5:
            recommendation = (
                f"Model fits comfortably within GPU memory "
                f"({required:.1f}GB used / {allowed_gpu_memory:.1f}GB allowed). "
                f"This shape can handle high throughput for batch embedding tasks."
            )
        else:
            recommendation = (
                f"Model fits within GPU memory "
                f"({required:.1f}GB used / {allowed_gpu_memory:.1f}GB allowed)."
            )

        return ModelConfig(
            model_details=model_detail,
            deployment_params=deployment_params,
            recommendation=recommendation,
        )
