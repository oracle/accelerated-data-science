#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Embedding model recommendation strategy.

Handles models like BERT, RoBERTa, E5-Mistral, GTE, Jina, NomicBERT, etc.
Embedding models are typically small and throughput-sensitive rather than memory-bound.

Dynamic parameter selection:
- --task embedding: always required to put vLLM in embedding mode
- --max-model-len: added when the model's context length deviates from the BERT default (512),
  which covers large LLM-backbone embedding models (E5-Mistral: 32768, Jina-v3: 8194, etc.)
- --dtype: derived from model's torch_dtype (float16/bfloat16/float32)
- --trust-remote-code: added when auto_map is present (e.g., Jina embeddings use custom LoRA code)
- For large LLM-backbone models (hidden_size > threshold): recommendation text notes
  that these are heavier than typical BERT-style embeddings
"""

from typing import List

from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.common.errors import AquaValueError
from ads.aqua.shaperecommend.constants import (
    LARGE_EMBEDDING_HIDDEN_SIZE_THRESHOLD,
    VLLM_PARAMS,
)
from ads.aqua.shaperecommend.estimator import EmbeddingMemoryEstimator
from ads.aqua.shaperecommend.llm_config import EmbeddingConfig, ParsedModelConfig
from ads.aqua.shaperecommend.shape_report import (
    DeploymentParams,
    ModelConfig,
    ModelDetail,
    ShapeRecommendationReport,
    ShapeReport,
)
from ads.aqua.shaperecommend.strategies.base import RecommendationStrategy

# Default BERT-style max sequence length; models matching this get no explicit --max-model-len
_BERT_DEFAULT_SEQ_LEN = 512


class EmbeddingStrategy(RecommendationStrategy):
    """
    Strategy for embedding models (BERT, RoBERTa, Jina, E5, GTE, NomicBERT, etc.).

    Embedding models:
    - Are typically small (< 1GB) for BERT-style models
    - Large LLM-backbone models (E5-Mistral, GTE-Qwen2) can be 7B+ parameters
    - Have minimal KV cache during inference (no token generation)
    - Focus on throughput rather than sequence length
    - Require --task embedding flag for vLLM

    Dynamic parameter selection:
    - --max-model-len added when seq_len != 512 (covers all non-BERT-default models)
    - --dtype set from torch_dtype in config
    - --trust-remote-code added when auto_map present (e.g., Jina with custom LoRA)
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

        # Embedding models - find all shapes that fit
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
            is_large = (
                embedding_config.hidden_size >= LARGE_EMBEDDING_HIDDEN_SIZE_THRESHOLD
            )
            if is_large:
                troubleshoot = (
                    f"The embedding model ({estimator.total_memory:.2f}GB) uses a large "
                    "LLM backbone (e.g., Mistral, Qwen2). These models require more GPU "
                    "memory than typical BERT-style embeddings. "
                    "Please select a shape with at least 16GB of GPU memory."
                )
            else:
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
        self,
        estimator: EmbeddingMemoryEstimator,
        config: EmbeddingConfig,
        allowed_gpu_memory: float,
    ) -> ModelConfig:
        """
        Build ModelConfig for embedding models with dynamic vLLM parameter selection.

        Dynamic params:
        - --task embedding: always required to run vLLM in pooling/embedding mode
        - --max-model-len <n>: when seq_len != 512 (e.g., 8194 for Jina-v3, 32768 for E5-Mistral)
        - --dtype <dtype>: explicit dtype from model config (float16/bfloat16/float32)
        - --trust-remote-code: when auto_map is present (e.g., Jina custom LoRA implementation)
        """
        params = [VLLM_PARAMS["task_embedding"]]

        # Add explicit --max-model-len when context length differs from BERT default (512)
        # This covers:
        # - Long-context BERT-style: NomicBERT (8192), Jina-v3 (8194)
        # - LLM-backbone embeddings: E5-Mistral (32768), GTE-Qwen2 (32768+)
        if config.max_seq_len and config.max_seq_len != _BERT_DEFAULT_SEQ_LEN:
            params.append(VLLM_PARAMS["max_model_len"])
            params.append(str(config.max_seq_len))

        # Dynamic dtype: use model's declared weight type
        # BERT-style models are typically float32; LLM-backbone models use float16/bfloat16
        weight_dtype = (config.weight_dtype or "float32").lower()
        if weight_dtype in ("float16", "bfloat16", "float32"):
            params.append(VLLM_PARAMS["dtype"])
            params.append(weight_dtype)

        # Trust remote code only if the model has custom auto_map modules
        # Example: Jina-embeddings-v3 uses custom XLM-RoBERTa-LoRA implementation
        if config.trust_remote_code:
            params.append(VLLM_PARAMS["trust_remote_code"])

        deployment_params = DeploymentParams(
            quantization=config.quantization or config.weight_dtype,
            max_model_len=config.max_seq_len,
            params=" ".join(params),
            weight_dtype=config.weight_dtype,
            env_var={},
        )

        model_detail = ModelDetail(
            model_size_gb=round(estimator.model_memory, 2),
            kv_cache_size_gb=0.0,  # Embedding models don't use KV cache for generation
            total_model_gb=round(estimator.total_memory, 2),
        )

        # Determine if this is a large LLM-backbone embedding model
        is_large_backbone = (
            config.hidden_size >= LARGE_EMBEDDING_HIDDEN_SIZE_THRESHOLD
            and config.max_seq_len is not None
            and config.max_seq_len > _BERT_DEFAULT_SEQ_LEN
        )

        required = estimator.total_memory
        backbone_note = (
            " (large LLM-backbone embedding model)" if is_large_backbone else ""
        )

        if required < allowed_gpu_memory * 0.5:
            recommendation = (
                f"Model fits comfortably within GPU memory"
                f"{backbone_note} "
                f"({required:.1f}GB used / {allowed_gpu_memory:.1f}GB allowed). "
                f"This shape can handle high throughput for batch embedding tasks."
            )
        else:
            recommendation = (
                f"Model fits within GPU memory"
                f"{backbone_note} "
                f"({required:.1f}GB used / {allowed_gpu_memory:.1f}GB allowed)."
            )

        return ModelConfig(
            model_details=model_detail,
            deployment_params=deployment_params,
            recommendation=recommendation,
        )
