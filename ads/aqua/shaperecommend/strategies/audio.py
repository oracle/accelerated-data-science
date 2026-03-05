#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Audio (Whisper ASR) recommendation strategy.

Handles Whisper models for automatic speech recognition.
Whisper has fixed architecture sizes and requires audio-specific vLLM flags.

Dynamic parameter selection:
- --max-model-len: set from max_target_positions (decoder length, typically 448)
- --dtype: derived from model's torch_dtype (float16 vs bfloat16)
- --trust-remote-code: added when auto_map is present in config
- For distil-Whisper (decoder_layers < threshold): lighter configuration since
  the distilled decoder is much smaller, reducing memory pressure

All Whisper variants share the same audio pre-processing pipeline:
- --limit-mm-per-prompt {"audio": 1} is always required (Whisper processes one audio
  segment at a time; the 30-second context window is enforced by the mel-spectrogram)
"""

from typing import List

from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.common.errors import AquaValueError
from ads.aqua.shaperecommend.constants import (
    VLLM_PARAMS,
    WHISPER_DISTILLED_DECODER_LAYERS_THRESHOLD,
)
from ads.aqua.shaperecommend.estimator import WhisperMemoryEstimator
from ads.aqua.shaperecommend.llm_config import ParsedModelConfig, WhisperConfig
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

    Dynamic parameter selection:
    - torch_dtype from config drives --dtype flag (float16/bfloat16)
    - auto_map presence drives --trust-remote-code
    - Distilled variants (few decoder layers) get lighter recommendations
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
            # Prefer gpu_specs.cpu_memory_in_gbs (always populated from GPU index);
            # fall back to shape.memory_in_gbs (top-level field, sometimes None).
            cpu_memory_gb = (
                getattr(shape.gpu_specs, "cpu_memory_in_gbs", None)
                or shape.memory_in_gbs
                or 0
            )
            cpu_required = (
                estimator.total_memory * 0.3
            )  # Rough estimate: 30% of total for CPU buffers

            if (
                estimator.validate_shape(allowed_gpu_memory)
                and cpu_memory_gb > cpu_required
            ):
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
        config: WhisperConfig,
        allowed_gpu_memory: float,
        cpu_memory_gb: float,
    ) -> ModelConfig:
        """
        Build ModelConfig for Whisper/ASR models with dynamic vLLM parameter selection.

        Dynamic params:
        - --limit-mm-per-prompt {"audio": 1}: always required for all Whisper variants
        - --max-model-len <max_target_positions>: decoder context length (typically 448)
        - --dtype <torch_dtype>: float16 or bfloat16 based on model's torch_dtype
        - --trust-remote-code: only when auto_map is present in config
        """
        params = [
            VLLM_PARAMS["limit_mm_per_prompt_audio"],
        ]

        # max_target_positions is the decoder max length (typically 448)
        if config.max_target_positions:
            params.append(VLLM_PARAMS["max_model_len"])
            params.append(str(config.max_target_positions))

        # Dynamic dtype: use the model's declared weight type
        # float16 is Whisper's standard; bfloat16 is used by some fine-tunes
        weight_dtype = (config.weight_dtype or "float16").lower()
        if weight_dtype in ("float16", "bfloat16", "float32"):
            # Only add explicit --dtype for non-default cases or when clearly specified
            # vLLM defaults to auto-detect; we add it explicitly to match model's intent
            params.append(VLLM_PARAMS["dtype"])
            params.append(weight_dtype)

        # Trust remote code only if the model has custom auto_map modules
        if config.trust_remote_code:
            params.append(VLLM_PARAMS["trust_remote_code"])

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

        # Build recommendation message, noting if this is a distilled variant
        required_gpu = estimator.total_memory
        required_cpu = required_gpu * 0.3
        is_distilled = (
            config.decoder_layers < WHISPER_DISTILLED_DECODER_LAYERS_THRESHOLD
        )

        distilled_note = (
            " (distil-Whisper variant: smaller decoder for faster inference)"
            if is_distilled
            else ""
        )

        if required_gpu < allowed_gpu_memory * 0.5 and cpu_memory_gb > required_cpu * 2:
            recommendation = (
                f"Model fits comfortably within GPU memory"
                f"{distilled_note} "
                f"({required_gpu:.1f}GB GPU / {allowed_gpu_memory:.1f}GB allowed, "
                f"~{required_cpu:.1f}GB CPU / {cpu_memory_gb:.1f}GB available). "
                f"This shape can handle high throughput for audio transcription tasks."
            )
        else:
            recommendation = (
                f"Model fits within GPU memory"
                f"{distilled_note} "
                f"({required_gpu:.1f}GB GPU / {allowed_gpu_memory:.1f}GB allowed). "
                f"CPU memory ({cpu_memory_gb:.1f}GB) is sufficient for audio pre-processing."
            )

        return ModelConfig(
            model_details=model_detail,
            deployment_params=deployment_params,
            recommendation=recommendation,
        )
