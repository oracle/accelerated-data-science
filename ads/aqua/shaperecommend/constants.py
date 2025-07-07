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
    "float32": ["bfloat16", "float16", "int8"],
    "bfloat16": ["float16", "int8"],
    "float16": ["int8"],
    "int8": ["8bit", "4bit (Not Recommended)"],
    "8bit": ["4bit (Not Recommended)"],
    "4bit": ["No smaller quantization available"]
}
