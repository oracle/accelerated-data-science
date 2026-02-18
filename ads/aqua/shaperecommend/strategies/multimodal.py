#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Multimodal (Vision-Language Model) recommendation strategy.

Handles models like LLaVA, Qwen2-VL, Nemotron-VL, InternVL, LLaVA-OneVision, mLLaMA, etc.
Combines text+vision estimators and adds multimodal-specific vLLM flags.

Dynamic parameter selection:
- --limit-mm-per-prompt {"image": N}: N=1 for basic VLMs (LLaVA-1.5, Phi-3-Vision);
  N=4 for multi-image/tiling models (LLaVA-OneVision, Qwen2-VL, mLLaMA).
  Presence of image_grid_pinpoints or specific model_type drives the higher count.
- --limit-mm-per-prompt {"video": 1}: added when video_token_index is in config
  (e.g., LLaVA-OneVision, Qwen2-VL support video input).
- --enforce-eager: only added for model architectures known to have CUDA graph issues
  (phi3_v, idefics2, paligemma). NOT added for all VLMs—many work fine without it.
- --trust-remote-code: added when auto_map is present in config (e.g., Nemotron-VL).
"""

import json
from typing import List

from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.common.errors import AquaValueError
from ads.aqua.shaperecommend.constants import (
    BITS_AND_BYTES_4BIT,
    BITSANDBYTES,
    ENFORCE_EAGER_MODEL_TYPES,
    MULTI_IMAGE_MODEL_TYPES,
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

# Image count for models that support tiling / multi-image natively
_MULTI_IMAGE_PROMPT_COUNT = 4
# Image count for single-image VLMs
_SINGLE_IMAGE_PROMPT_COUNT = 1


def _build_mm_per_prompt_flag(image_count: int, has_video: bool) -> str:
    """
    Build the --limit-mm-per-prompt flag value as a JSON dict string.

    Examples:
    - image_count=1, has_video=False  -> '{"image": 1}'
    - image_count=4, has_video=True   -> '{"image": 4, "video": 1}'
    """
    mm_dict = {"image": image_count}
    if has_video:
        mm_dict["video"] = 1
    return f"--limit-mm-per-prompt {json.dumps(mm_dict)}"


class MultimodalStrategy(RecommendationStrategy):
    """
    Strategy for multimodal (vision-language) models.

    Combines text and vision estimators, adds image token overhead,
    and appends multimodal-specific vLLM flags.

    Dynamic parameter selection:
    - --limit-mm-per-prompt: image count based on model capabilities (1 or 4),
      plus video=1 when model supports video tokens
    - --enforce-eager: only for architectures known to require it
    - --trust-remote-code: only when auto_map present
    """

    def recommend(
        self,
        parsed_config: ParsedModelConfig,
        shapes: List[ComputeShapeSummary],
        model_name: str,
        batch_size: int = 1,
    ) -> ShapeRecommendationReport:
        """Generate recommendations for multimodal models."""
        if not parsed_config.llm_config and not parsed_config.vision_config:
            raise AquaValueError(
                "MultimodalStrategy requires at least llm_config or vision_config in ParsedModelConfig."
            )

        llm_config = parsed_config.llm_config
        vision_config = parsed_config.vision_config

        # For vision-only configs (e.g., LLaVA-1.5 with incomplete text_config),
        # we can only recommend based on vision memory; no seq-len iteration possible.
        if not llm_config:
            return self._recommend_vision_only(
                parsed_config=parsed_config,
                vision_config=vision_config,
                shapes=shapes,
                model_name=model_name,
            )

        recommendations = []

        if not shapes:
            raise AquaValueError("No GPU shapes were passed for recommendation.")

        # Determine multimodal capabilities from parsed config metadata
        model_type = (parsed_config.model_type or "").lower()
        has_video = parsed_config.has_video_tokens
        has_tiling = (
            parsed_config.has_image_grid_pinpoints
            or model_type in MULTI_IMAGE_MODEL_TYPES
        )
        # trust_remote_code is read from ParsedModelConfig (top-level field) which
        # combines top-level auto_map (e.g., Nemotron-VL) with nested llm_config auto_map.
        trust_remote_code = parsed_config.trust_remote_code

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
                                    estimator,
                                    vision_memory_gb,
                                    allowed_gpu_memory,
                                    model_type=model_type,
                                    has_video=has_video,
                                    has_tiling=has_tiling,
                                    trust_remote_code=trust_remote_code,
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
                                estimator,
                                vision_memory_gb,
                                allowed_gpu_memory,
                                model_type=model_type,
                                has_video=has_video,
                                has_tiling=has_tiling,
                                trust_remote_code=trust_remote_code,
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
                        estimator,
                        vision_memory_gb,
                        allowed_gpu_memory,
                        model_type=model_type,
                        has_video=has_video,
                        has_tiling=has_tiling,
                        trust_remote_code=trust_remote_code,
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

    def _recommend_vision_only(
        self,
        parsed_config: ParsedModelConfig,
        vision_config,
        shapes: List[ComputeShapeSummary],
        model_name: str,
    ) -> ShapeRecommendationReport:
        """
        Fallback recommendation path for multimodal models where llm_config is None.

        This handles VLMs (e.g., LLaVA-1.5) whose text_config section is a
        reference to an external model and cannot be parsed into a full LLMConfig.
        In this case we estimate only the vision encoder memory and recommend
        shapes that can fit it, using conservative multimodal vLLM params.
        """
        if not vision_config:
            raise AquaValueError(
                "MultimodalStrategy requires vision_config when llm_config is absent."
            )

        vision_estimator = VisionMemoryEstimator(vision_config=vision_config)
        vision_memory_gb = vision_estimator.model_memory

        model_type = (parsed_config.model_type or "").lower()
        has_video = parsed_config.has_video_tokens
        has_tiling = (
            parsed_config.has_image_grid_pinpoints
            or model_type in MULTI_IMAGE_MODEL_TYPES
        )
        trust_remote_code = parsed_config.trust_remote_code

        recommendations = []
        for shape in shapes:
            allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs
            if (allowed_gpu_memory * 0.9) > vision_memory_gb:
                image_count = (
                    _MULTI_IMAGE_PROMPT_COUNT
                    if has_tiling
                    else _SINGLE_IMAGE_PROMPT_COUNT
                )
                params_list = [_build_mm_per_prompt_flag(image_count, has_video)]
                if model_type in ENFORCE_EAGER_MODEL_TYPES:
                    params_list.append(VLLM_PARAMS["enforce_eager"])
                if trust_remote_code:
                    params_list.append(VLLM_PARAMS["trust_remote_code"])

                deployment_params = DeploymentParams(
                    quantization=None,
                    max_model_len=None,
                    params=" ".join(params_list),
                    weight_dtype=None,
                    env_var={},
                )
                model_detail = ModelDetail(
                    model_size_gb=round(vision_memory_gb, 2),
                    kv_cache_size_gb=0.0,
                    total_model_gb=round(vision_memory_gb, 2),
                )
                config = ModelConfig(
                    model_details=model_detail,
                    deployment_params=deployment_params,
                    recommendation=f"Vision encoder fits in {allowed_gpu_memory} GB GPU memory.",
                )
                recommendations.append(
                    ShapeReport(shape_details=shape, configurations=[config])
                )
                break

        troubleshoot_msg = ""
        if not recommendations:
            troubleshoot_msg = (
                "No GPU shape could fit the vision encoder. "
                "Consider using a smaller model or a shape with more GPU memory."
            )

        return ShapeRecommendationReport(
            display_name=model_name,
            recommendations=recommendations,
            troubleshoot=troubleshoot_msg,
        )

    def _build_multimodal_config(
        self,
        estimator,
        vision_memory_gb: float,
        allowed_gpu_memory: float,
        model_type: str = "",
        has_video: bool = False,
        has_tiling: bool = False,
        trust_remote_code: bool = False,
    ) -> ModelConfig:
        """
        Build a ModelConfig with dynamic multimodal-specific deployment params.

        Dynamic params:
        - --limit-mm-per-prompt {"image": N[, "video": 1]}:
            N=4 for tiling/multi-image models (LLaVA-OneVision, Qwen2-VL, mLLaMA);
            N=1 for single-image VLMs (LLaVA-1.5, LLaVA-v1.6-mistral).
            Video slot added when model supports video_token_index.
        - --enforce-eager: only for architectures with known CUDA graph limitations
            (phi3_v, idefics2, paligemma). NOT added by default.
        - --trust-remote-code: passed from ParsedModelConfig.trust_remote_code,
            which combines top-level auto_map with nested llm_config auto_map.
        - --max-model-len, --quantization: inherited from text strategy logic.
        """
        c = estimator.llm_config
        params = []

        # Standard sequence length and quantization params
        if estimator.seq_len < c.max_seq_len:
            params.append(VLLM_PARAMS["max_model_len"])
            params.append(str(estimator.seq_len))

        if not c.quantization and c.in_flight_quantization == "4bit":
            params.append(VLLM_PARAMS["in_flight_quant"])

        # --- Dynamic multimodal params ---

        # Determine image slot count based on model capabilities
        if has_tiling:
            # High-resolution tiling models process images as multiple tiles:
            # LLaVA-OneVision, Qwen2-VL, mLLaMA support up to N tiles per image
            image_count = _MULTI_IMAGE_PROMPT_COUNT
        else:
            # Basic VLMs: one image per prompt
            # LLaVA-1.5, LLaVA-v1.6-mistral, basic Phi-3-Vision
            image_count = _SINGLE_IMAGE_PROMPT_COUNT

        params.append(_build_mm_per_prompt_flag(image_count, has_video))

        # --enforce-eager: only for architectures known to need it
        # Many VLMs (LLaVA, Qwen2-VL, InternVL) work fine with CUDA graphs.
        # phi3_v, idefics2, paligemma have custom ops that conflict with graph capture.
        if model_type in ENFORCE_EAGER_MODEL_TYPES:
            params.append(VLLM_PARAMS["enforce_eager"])

        # --trust-remote-code when model uses custom auto_map code.
        # This is passed from ParsedModelConfig.trust_remote_code which correctly
        # combines top-level auto_map (e.g., Nemotron-VL) with nested llm_config auto_map.
        if trust_remote_code:
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
