#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.shaperecommend.constants
~~~~~~~~~~~~~~

This module contains constants used in Aqua GPU Recommendation for Models.

LLAMA_REQUIRED_FIELDS refer to fields necessary for calculating model memory for GQA Architecture Models

MOE_REQUIRED_FIELDS refer to fields necessary for Mixture of Experts (MoE) Architecture Models

NEXT_QUANT suggests the next quantization level based on the current quantization (if applied) or the model weights (if no quantization yet)
"""
LLAMA_REQUIRED_FIELDS = [
    "num_hidden_layers", "hidden_size", "num_attention_heads",
    "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size"
]

MOE_REQUIRED_FIELDS = LLAMA_REQUIRED_FIELDS + [
    "num_local_experts", "intermediate_size"
]

NEXT_QUANT = {
    "float32": ["4bit", "8bit"], # bits and bytes does not support bfloat16, pytorch responsibility
    "bfloat16": ["4bit", "8bit"],
    "float16": ["4bit", "8bit"],
    "int8": ["4bit"],
    "fp8":  ["4bit", "8bit"],
    "8bit": ["4bit"],
    "int4": ["No smaller quantization available"],
    "4bit": ["No smaller quantization available"]
}

#TODO:
SHAPES_METADATA = "/Users/elizjo/tmp/accelerated-data-science/ads/aqua/resources/shapes.json"

TEXT_MODEL = "text-generation"

QUANT_MAPPING = {
            "float32": 4,
            "bfloat16": 2,
            "float16": 2,
            "fp16": 2,
            "half": 2,
            "int8": 1,
            "fp8": 1,
            "8bit": 1,
            "4bit": 0.5,
            "int4": 0.5,
        }


