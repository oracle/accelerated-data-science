#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Audio (Whisper ASR) recommendation strategy.

Handles Whisper models for automatic speech recognition.
Whisper has fixed architecture sizes and requires audio-specific vLLM flags.
"""

from typing import List

from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.common.errors import AquaValueError
from ads.aqua.shaperecommend.constants import VLLM_PARAMS
from ads.aqua.shaperecommend.estimator import WhisperMemoryEstimator
from ads.aqua.shaperecommend.llm_config import ParsedModelConfig
from ads.aqua.shaperecommend.shape_report import (
    DeploymentParams,
    ModelConfig,
    ModelDetail,
    ShapeRecommendationReport,
    ShapeReport,
)
from ads.aqua.shaperecommend.strategies.base import RecommendationStrategy


class AudioStrategy(RecommendationStrategy):
    """
    Strategy for audio/ASR models (Whisper).
    
    Whisper models:
    - Have fixed encoder-decoder architecture
    - Use CPU for audio pre-processing (mel-spectrograms)
    - Require --limit-mm-per-prompt {"audio": 1}
    - max_model_len applies only to decoder (typically 448 tokens)
    """

    def recommend(
        self,
        parsed_config: ParsedModelConfig,
        shapes: List[ComputeShapeSummary],
        model_name: str,
        batch_size: int = 1,
    ) -> ShapeRecommendationReport:
        """Generate recommendations for Whisper/ASR models."""
        if not parsed_config.whisper_config:
            raise AquaValueError(
                "AudioStrategy requires whisper_config in ParsedModelConfig."
            )

        whisper_config = parsed_config.whisper_config
        estimator = WhisperMemoryEstimator(whisper_config=whisper_config)

        recommendations = []

        if not shapes:
            raise AquaValueError("No GPU shapes were passed for recommendation.")

        # Whisper models are typically small - find all shapes that fit
        for shape in shapes:
            allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs
            # Whisper also needs CPU memory for audio buffers - check if CPU memory is sufficient
            cpu_memory_gb = shape.memory_in_gbs or 0
            cpu_required = estimator.total_memory * 0.3  # Rough estimate: 30% of total for CPU buffers

            if estimator.validate_shape(allowed_gpu_memory) and cpu_memory_gb > cpu_required:
                model_config = self._build_audio_config(
                    estimator, whisper_config, allowed_gpu_memory, cpu_memory_gb
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
                f"The Whisper model ({estimator.total_memory:.2f}GB GPU memory) "
                "requires both GPU memory and sufficient CPU memory for audio pre-processing. "
                "Please select a shape with adequate CPU memory (typically 32GB+)."
            )

        return ShapeRecommendationReport(
            display_name=model_name,
            recommendations=recommendations,
            troubleshoot=troubleshoot,
        )

    def _build_audio_config(
        self,
        estimator: WhisperMemoryEstimator,
        config,
        allowed_gpu_memory: float,
        cpu_memory_gb: float,
    ) -> ModelConfig:
        """
        Build ModelConfig for Whisper/ASR models.
        
        Adds:
        - --limit-mm-per-prompt {"audio": 1}
        - --task transcribe (optional, vLLM default for Whisper)
        - --max-model-len 448 (decoder only, not encoder)
        """
        params = [
            VLLM_PARAMS["limit_mm_per_prompt_audio"],
            # task_transcribe is optional - vLLM auto-detects for Whisper
        ]

        # max_target_positions is the decoder max length (typically 448)
        if config.max_target_positions:
            params.append(VLLM_PARAMS["max_model_len"])
            params.append(str(config.max_target_positions))

        deployment_params = DeploymentParams(
            quantization=config.quantization or config.weight_dtype,
            max_model_len=config.max_target_positions,
            params=" ".join(params),
            weight_dtype=config.weight_dtype,
            env_var={},
        )

        model_detail = ModelDetail(
            model_size_gb=round(estimator.model_memory, 2),
            kv_cache_size_gb=0.0,  # Whisper has minimal KV cache (decoder only, fixed length)
            total_model_gb=round(estimator.total_memory, 2),
        )

        # Recommendation message includes CPU memory check
        required_gpu = estimator.total_memory
        required_cpu = required_gpu * 0.3

        if required_gpu < allowed_gpu_memory * 0.5 and cpu_memory_gb > required_cpu * 2:
            recommendation = (
                f"Model fits comfortably within GPU memory "
                f"({required_gpu:.1f}GB GPU / {allowed_gpu_memory:.1f}GB allowed, "
                f"~{required_cpu:.1f}GB CPU / {cpu_memory_gb:.1f}GB available). "
                f"This shape can handle high throughput for audio transcription tasks."
            )
        else:
            recommendation = (
                f"Model fits within GPU memory "
                f"({required_gpu:.1f}GB GPU / {allowed_gpu_memory:.1f}GB allowed). "
                f"CPU memory ({cpu_memory_gb:.1f}GB) is sufficient for audio pre-processing."
            )

        return ModelConfig(
            model_details=model_detail,
            deployment_params=deployment_params,
            recommendation=recommendation,
        )
