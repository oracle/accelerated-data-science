#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import re
from typing import List, Optional

from pydantic import BaseModel, Field

from ads.aqua.shaperecommend.constants import NEXT_QUANT


class LLMConfig(BaseModel):
    """
    Standardized configuration object for evaluating the size of Large Language Models (LLMs)
    based on their config.json file.
        Architecture is determined by which non-required fields are defined below.
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
    max_seq_len: int = Field(
        ..., description="Maximum input sequence length (context window)."
    )
    weight_dtype: Optional[str] = Field(
        "float32", description="Parameter data type: 'float32', 'float16', etc."
    )
    quantization: Optional[str] = Field(
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
        mapping = {
            "float32": 4,
            "bfloat16": 2,
            "float16": 2,
            "fp16": 2,
            "half": 2,
            "int8": 1,
            "8bit": 1,
            "4bit": 0.5,
            "awq": 0.5,
            "gptq": 0.5,
        }
        # Quantization takes precedence
        q = (self.quantization or "").lower()
        if q in mapping:
            return mapping[q]
        if "bit" in q:
            m = re.match(r"(\d+)bit", q)
            if m:
                bits = int(m[1])
                return bits / 8  # bytes per parameter
            # Unknown bit type: fallback
            return 1
        # Fallback to weight_dtype mapping
        dtype = (self.weight_dtype or "float32").lower()
        if dtype in mapping:
            return mapping[dtype]
        return mapping["float32"]  # Default

    @classmethod
    def detect_quantization(cls, raw: dict) -> Optional[str]:
        """
        Detects main quantization types from Hugging Face config dict.
        """
        if raw.get("load_in_8bit"):
            return "8bit"
        if raw.get("load_in_4bit"):
            return "4bit"
        if "quantization_config" in raw:
            qcfg = raw["quantization_config"]
            if "gptq" in str(qcfg).lower():
                return "gptq"
            if "awq" in str(qcfg).lower():
                return "awq"
            bits = qcfg.get("bits") or qcfg.get("wbits")
            if bits:
                return f"{bits}bit"
            return "custom-quant"
        return None

    @property
    def suggested_quantizations(self) -> List[str]:
        """
        Suggests the next quantization level to use based on the current quantization level if available.
        Model weights as fallback if no quantization is currently applied.
        """
        key = (self.quantization or self.weight_dtype or "float32").lower()
        return NEXT_QUANT.get(key, [])

    @classmethod
    def from_raw_config(cls, raw: dict) -> "LLMConfig":
        """
        Instantiates an LLMConfig from a raw Hugging Face config.json file,
        using robust key detection (considers multiple possibilities for keys referring to the same model attribute).
        """

        # Field mappings with fallback
        num_hidden_layers = (
            raw.get("num_hidden_layers") or raw.get("n_layer") or raw.get("num_layers")
        )
        hidden_size = raw.get("hidden_size") or raw.get("n_embd") or raw.get("d_model")
        vocab_size = raw.get("vocab_size")
        weight_dtype = str(raw.get("torch_dtype", "float32"))
        quantization = cls.detect_quantization(raw)
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
            max_seq_len,
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
            max_seq_len=int(max_seq_len),
            num_local_experts=num_local_experts,
            intermediate_size=intermediate_size,
        )
