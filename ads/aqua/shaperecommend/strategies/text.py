#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Text-generation (decoder-only LLM) recommendation strategy.

Handles standard text-generation models like Llama, Mistral, Qwen, Falcon.
This is the default strategy that uses the existing logic from recommend.py.
"""

from typing import List

from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.common.errors import AquaValueError
from ads.aqua.shaperecommend.constants import (
    BITS_AND_BYTES_4BIT,
    BITSANDBYTES,
    TROUBLESHOOT_MSG,
)
from ads.aqua.shaperecommend.estimator import get_estimator
from ads.aqua.shaperecommend.llm_config import ParsedModelConfig
from ads.aqua.shaperecommend.shape_report import (
    ModelConfig,
    ShapeRecommendationReport,
    ShapeReport,
)
from ads.aqua.shaperecommend.strategies.base import RecommendationStrategy


class TextGenerationStrategy(RecommendationStrategy):
    """
    Strategy for text-generation (decoder-only LLM) models.
    
    Uses the existing logic from recommend.py::_summarize_shapes_for_seq_lens().
    Supports quantized and unquantized models, iterates through sequence lengths
    and quantization options to find compatible shapes.
    """

    def recommend(
        self,
        parsed_config: ParsedModelConfig,
        shapes: List[ComputeShapeSummary],
        model_name: str,
        batch_size: int = 1,
    ) -> ShapeRecommendationReport:
        """
        Generate recommendations for text-generation models.
        
        This method is extracted from the original recommend.py::_summarize_shapes_for_seq_lens().
        """
        if not parsed_config.llm_config:
            raise AquaValueError(
                "TextGenerationStrategy requires llm_config to be set in ParsedModelConfig."
            )

        config = parsed_config.llm_config
        recommendations = []

        if not shapes:
            raise AquaValueError(
                "No GPU shapes were passed for recommendation. Ensure shape parsing succeeded."
            )

        # Pre-quantized: only consider different max-seq-len
        if config.quantization_type:
            deployment_config = config.calculate_possible_seq_len()
            for shape in shapes:
                shape_quantization = set(shape.gpu_specs.quantization)
                if config.quantization_type in shape_quantization:
                    allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs
                    for max_seq_len in deployment_config:
                        estimator = get_estimator(
                            llm_config=config,
                            seq_len=max_seq_len,
                            batch_size=batch_size,
                        )
                        if estimator.validate_shape(allowed_gpu_memory):
                            best_config = [
                                ModelConfig.constuct_model_config(
                                    estimator, allowed_gpu_memory
                                )
                            ]
                            recommendations.append(
                                ShapeReport(
                                    shape_details=shape, configurations=best_config
                                )
                            )
                            break

        # unquantized: consider inflight quantization (4bit)
        else:
            deployment_config = config.optimal_config()
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
                        updated_config = config.model_copy(
                            update={"in_flight_quantization": quantization}
                        )
                        prev_quant = quantization
                    estimator = get_estimator(
                        llm_config=updated_config,
                        seq_len=max_seq_len,
                        batch_size=batch_size,
                    )
                    if estimator.validate_shape(allowed_gpu_memory):
                        best_config = [
                            ModelConfig.constuct_model_config(
                                estimator, allowed_gpu_memory
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
            # Troubleshooting advice if nothing fits
            # Assumes shapes is sorted largest to smallest and quantizations 'fp8'/'4bit' exist
            troubleshoot_msg += TROUBLESHOOT_MSG

            largest_shapes = (
                [(shapes[0], "fp8", False), (shapes[1], "4bit", True)]
                if len(shapes) > 1
                else []
            )  # shape, quantization, in_flight_quantization

            for shape, quantization, in_flight in largest_shapes:
                if in_flight:
                    updated_config = config.model_copy(
                        update={"in_flight_quantization": quantization}
                    )
                else:
                    updated_config = config.model_copy(
                        update={"quantization": quantization}
                    )
                estimator = get_estimator(
                    llm_config=updated_config, seq_len=2048, batch_size=batch_size
                )
                allowed_gpu_memory = shape.gpu_specs.gpu_memory_in_gbs * 0.9
                best_config = [
                    ModelConfig.constuct_model_config(estimator, allowed_gpu_memory)
                ]
                recommendations.append(
                    ShapeReport(shape_details=shape, configurations=best_config)
                )

        return ShapeRecommendationReport(
            display_name=model_name,
            recommendations=recommendations,
            troubleshoot=troubleshoot_msg,
        )
