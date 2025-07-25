#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import re
from typing import Optional

from pydantic import BaseModel, Field

from ads.aqua.common.errors import AquaRecommendationError, AquaValueError
from ads.aqua.shaperecommend.constants import NEXT_QUANT, QUANT_MAPPING


class LLMConfig(BaseModel):
    """
    Standardized configuration object for evaluating the size of Large Language Models (LLMs)
    based on their architecture and quantization.
    """

    num_hidden_layers: int = Field(
        ...,
        description="Number of transformer blocks (layers) in the model’s neural network stack.",
    )
    hidden_size: int = Field(
        ..., description="Embedding dimension or hidden size of each layer."
    )
    vocab_size: int = Field(..., description="Vocabulary size for input/output tokens.")
    num_attention_heads: int = Field(
        ...,
        description="Number of attention heads (used for queries and to determine head_dim).",
    )

    head_dim: int = Field(
        ...,
        description="Dimension of each attention head. Typically hidden_size // num_attention_heads.",
    )
    max_seq_len: Optional[int] = Field(
        8192, description="Maximum input sequence length (context window)."
    )
    weight_dtype: Optional[str] = Field(
        "float32", description="Parameter data type: 'float32', 'float16', etc."
    )
    quantization: Optional[str] = Field(
        None,
        description="Quantization weight (e.g., '8bit', '4bit') or None if unquantized.",
    )
    quantization_type: Optional[str] = Field(
        None,
        description="Quantization method (e.g., '8bit', '4bit', 'gptq', 'awq') or None if unquantized.",
    )

    num_key_value_heads: Optional[int] = Field(
        None,
        description="Number of key/value heads (for GQA architectures: Llama, Mistral, Falcon, Qwen, etc.). Used to determine KV cache size",
    )

    num_local_experts: Optional[int] = Field(
        None, description="For MoE architectures, the number of experts per MoE layer"
    )
    intermediate_size: Optional[int] = Field(
        None, description="For MoE architectures, size of the MLP activation layer."
    )

    tie_word_embeddings: Optional[bool] = Field(None)

    @property
    def bytes_per_parameter(self) -> float:
        """
        Returns the number of bytes used to store a model parameter,
        accounting for quantization or weight storage type.
        """
        # Quantization takes precedence
        q = (self.quantization or "").lower()

        # Direct match in mapping
        if q in QUANT_MAPPING:
            return QUANT_MAPPING[q]

        # Dynamic bit-width detection
        m = re.match(r"(\d+)\s*bit", q)
        if m:
            bits = int(m[1])
            return bits / 8  # bytes per parameter

        # Fallback to dtype mapping
        dtype = (self.weight_dtype or "float32").lower()
        return QUANT_MAPPING.get(dtype, QUANT_MAPPING["float32"])

    @classmethod
    def detect_quantization_type(cls, raw: dict) -> Optional[str]:
        """
        Detects quantization type (e.g., 'gptq', 'bitsandbytes', 'awq', etc.) from Hugging Face config dict.
        """
        qcfg = raw.get("quantization_config", {})
        if raw.get("load_in_8bit") or raw.get("load_in_4bit"):
            return "bitsandbytes"
        for key in [
            "gptq",
            "awq",
            "marlin",
            "bitblas",
            "aqlm",
            "deepspeedfp",
            "gguf",
            "fp8",
        ]:
            if key in str(qcfg).lower() or key in str(raw).lower():
                return key
        return None

    @classmethod
    def detect_quantization_bits(cls, raw: dict) -> Optional[str]:
        """
        Detects quantization bit-width as a string (e.g., '4bit', '8bit') from Hugging Face config dict.
        """
        if raw.get("load_in_8bit"):
            return "8bit"
        if raw.get("load_in_4bit"):
            return "4bit"
        if "quantization_config" in raw:
            qcfg = raw["quantization_config"]
            bits = qcfg.get("bits") or qcfg.get("wbits")
            if bits:
                return f"{bits}bit"
        return None

    @property
    def suggested_quantizations(self):
        """
        Suggests the next lower quantization options based on the current quantization level/ weight size.

        If model is un-quantized, uses the weight size.
        If model is pre-quantized, uses the quantization level.
        """
        key = (self.quantization or self.weight_dtype or "float32").lower()
        return NEXT_QUANT.get(key, [])

    def calculate_possible_seq_len(self, min_len=2048):
        """
        Calculates a list of possible sequence lengths (in tokens).
        [2048, ... max-length] (max-length found in model's config.json file)
        """
        vals = []
        curr = min_len
        max_seq_len = 16384 if not self.max_seq_len else self.max_seq_len
        while curr <= max_seq_len:
            vals.append(curr)
            curr *= 2
        if vals and vals[-1] != max_seq_len:
            vals.append(max_seq_len)
        return vals

    def optimal_config(self):
        """
        Builds a list of optimal configuration parameters (sorted descending). Combination of:
            - Quantization / weight sizes: bfloat16 weight size -> 8bit -> 4bit
            - max-model-len: power-of-two model lengths from max length (config.json of model) to 2048 tokens.

        Example:
        [('bfloat16', max_model_len supported by model) ('bfloat16', 1/2 of max_model_len) ... ('int8', 2048), ('int4', 4096), ('int4', 2048)]

        """
        # Create a copy of the suggested_quantizations list
        quantizations = self.suggested_quantizations[:]
        quantizations.append("bfloat16")

        lengths = self.calculate_possible_seq_len()

        configs = []
        for quantization in quantizations:
            for length in lengths:
                configs.append((quantization, length))

        configs.sort(
            key=lambda x: (-QUANT_MAPPING.get(x[0], 0), -x[1])
        )  # (-quant_priority, -max_seq_len)
        return configs

    @classmethod
    def validate_model_support(cls, raw: dict) -> ValueError:
        """
        Validates if model is decoder-only. Check for text-generation model occurs at DataScienceModel level.
        """
        excluded_models = {"t5", "gemma", "bart", "bert", "roberta", "albert"}
        if (
            raw.get("is_encoder_decoder", False) # exclude encoder-decoder models
            or (raw.get("is_decoder") is False) # exclude explicit encoder-only models (altho no text-generation task ones, just dbl check)
            or raw.get("model_type", "").lower() # exclude by known model types
            in excluded_models
        ):
            raise AquaRecommendationError(
                "Please provide a decoder-only text-generation model (ex. Llama, Falcon, etc). "
                "Encoder-decoder models (ex. T5, Gemma) and encoder-only (BERT) are not supported in this tool at this time."
            )

    @classmethod
    def from_raw_config(cls, raw: dict) -> "LLMConfig":
        """
        Instantiates an LLMConfig from a raw Hugging Face config.json file,
        using robust key detection and fallback for architecture.
        """
        cls.validate_model_support(raw)

        # Field mappings with fallback
        num_hidden_layers = (
            raw.get("num_hidden_layers") or raw.get("n_layer") or raw.get("num_layers")
        )
        hidden_size = raw.get("hidden_size") or raw.get("n_embd") or raw.get("d_model")
        vocab_size = raw.get("vocab_size")
        weight_dtype = str(raw.get("torch_dtype", "float32"))
        quantization = cls.detect_quantization_bits(raw)
        quantization_type = cls.detect_quantization_type(raw)

        if not quantization and quantization_type in QUANT_MAPPING:
            quantization = quantization_type

        num_key_value_heads = (
            raw.get("num_key_value_heads")  # GQA models (ex. Llama-type)
        )

        num_attention_heads = (
            raw.get("num_attention_heads") or raw.get("n_head") or raw.get("num_heads")
        )

        head_dim = raw.get("head_dim") or (
            int(hidden_size) // int(num_attention_heads)
            if hidden_size and num_attention_heads
            else None
        )
        max_seq_len = (
            raw.get("max_position_embeddings")
            or raw.get("n_positions")
            or raw.get("max_seq_len")
            or 2048
        )

        num_local_experts = (
            raw.get("num_local_experts")
            or raw.get("n_routed_experts")
            or raw.get("num_experts")
        )
        intermediate_size = raw.get("moe_intermediate_size") or raw.get(
            "intermediate_size"
        )

        # Type safety: minimal assertion
        if None in [
            num_hidden_layers,
            hidden_size,
            vocab_size,
            num_attention_heads,
            head_dim,
        ]:
            raise ValueError("Missing required value in model config.")

        return cls(
            num_hidden_layers=int(num_hidden_layers),
            hidden_size=int(hidden_size),
            num_attention_heads=int(num_attention_heads),
            num_key_value_heads=num_key_value_heads,
            head_dim=int(head_dim),
            vocab_size=int(vocab_size),
            weight_dtype=weight_dtype,
            quantization=quantization,
            quantization_type=quantization_type,
            max_seq_len=int(max_seq_len),
            num_local_experts=num_local_experts,
            intermediate_size=intermediate_size,
        )
