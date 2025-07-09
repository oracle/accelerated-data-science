#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from typing import Optional

from pydantic import BaseModel, Field

from ads.aqua.app import logger
from ads.aqua.shaperecommend.constants import LLAMA_REQUIRED_FIELDS, MOE_REQUIRED_FIELDS
from ads.aqua.shaperecommend.llm_config import LLMConfig


class MemoryEstimator(BaseModel):
    """
    The generic estimator for Transformer Architecture models (OPT/ Bloom)
    Used as a fallback estimator if model identified is not a MoE or GQA Architecture Model.
    Has properties to estimate the KV Cache size, Model size, and total footprint (KV Cache + Model size)
    """

    llm_config: LLMConfig = Field(
        ...,
        description="The model's config.json file with the necessary parameters for model size and KV cache estimation/",
    )
    batch_size: int = (
        1  # we assume that estimation for batch sizes are not supported yet
    )
    seq_len: Optional[int] = Field(
        4096, description="The max-seq-len to estimate the size of the KV cache."
    )

    @property
    def kv_cache_memory(self) -> float:
        """
        Estimates the KV cache size (in GB) using the LLM config.json parameters.

        Uses num_attention_heads (assumes no GQA, each attention head has its own query, key, value) for estimation
        """
        seq_len = self.seq_len or self.llm_config.max_seq_len
        c = self.llm_config
        kv_cache_dtype_bytes = (
            c.bytes_per_parameter
        )  # vLLM uses model's weight/quantization applied to KV cache

        total_bytes = (
            self.batch_size
            * c.num_hidden_layers
            * 2
            * c.num_attention_heads
            * seq_len
            * c.head_dim
            * kv_cache_dtype_bytes
        )
        return total_bytes / 1e9

    @property
    def model_memory(self) -> float:
        """
        Estimates the model size (in GB) based on estimating the model parameter size and model weights

        Model Parameter estimation: Standard decoder-only, untied/tied embeddings possible
        """
        c = self.llm_config
        embedding_count = 1 if getattr(c, "tie_word_embeddings", True) else 2
        embedding_params = (
            embedding_count * c.vocab_size * c.hidden_size
        )  # input and output untied
        layer_params = 12 * c.num_hidden_layers * (c.hidden_size**2)  # GPT-style
        num_params = layer_params + embedding_params

        return num_params * c.bytes_per_parameter / 1e9

    # @property
    # def model_overhead(self) -> float:
    #     overhead = max(1, math.ceil(0.0 * self.model_memory))
    #     return overhead

    @property
    def total_memory(self) -> float:
        """
        Computes the total memory footprint of the model (KV cache & model size from estimated parameters)
        """
        return self.model_memory + self.kv_cache_memory


# Specialized estimators:
class LlamaMemoryEstimator(MemoryEstimator):
    """
    Estimator for GQA-type architectures. Handles tied (memory savings) and untied embeddings,
    and uses grouped attention (GQA) for more efficient KV cache memory estimation.

    KV cache: Use num_attention_heads (assumes GQA)
    Model Parameter estimation: Standard decoder-only, untied/tied embeddings possible
    """

    @property
    def model_memory(self) -> float:
        """
        Returns estimated model parameter memory (in GB), accurately accounting
        for Llama-style attention and MLP, and tied or untied embeddings.
        """
        c = self.llm_config

        embedding_params, attn_params = self._calc_attn_embed_params()

        # MLP params
        gate_proj = c.hidden_size * c.intermediate_size
        up_proj = c.hidden_size * c.intermediate_size
        down_proj = c.intermediate_size * c.hidden_size
        mlp_params = gate_proj + up_proj + down_proj

        # Total per-layer
        layer_params = attn_params + mlp_params
        # Total params
        num_params = c.num_hidden_layers * layer_params + embedding_params
        return num_params * c.bytes_per_parameter / 1e9

    @property
    def kv_cache_memory(self) -> float:
        """
        Returns estimated KV cache memory in GB for GQA models.

        Grouped Query Attention uses num_key_value_heads, which groups of Q heads share a K and V projection.
        num_key_value_heads < num_attention_heads, which reduces the KV Cache size.
        """
        c = self.llm_config
        seq_len = self.seq_len or getattr(c, "max_seq_len", 2048)
        kv_cache_dtype_bytes = c.bytes_per_parameter
        kv_heads = c.num_key_value_heads

        total_bytes = (
            self.batch_size
            * c.num_hidden_layers
            * 2
            * kv_heads
            * seq_len
            * c.head_dim
            * kv_cache_dtype_bytes
        )
        return total_bytes / 1e9

    def _calc_attn_embed_params(self) -> tuple:
        """
        Returns the embedding parameter count and attention parameter count for Llama-family (GQA) models.
        """
        c = self.llm_config

        # Embedding parameters
        # assume tied embeddings unless tie_word_embeddings = False
        embedding_count = 1 if getattr(c, "tie_word_embeddings", True) else 2
        embedding_params = embedding_count * c.vocab_size * c.hidden_size

        q_proj = c.hidden_size * c.hidden_size
        k_proj = c.hidden_size * (c.num_key_value_heads * c.head_dim)
        v_proj = c.hidden_size * (c.num_key_value_heads * c.head_dim)
        o_proj = c.hidden_size * c.hidden_size
        attn_params = q_proj + k_proj + v_proj + o_proj

        return embedding_params, attn_params


class MixtureMemoryEstimator(LlamaMemoryEstimator):
    """
    Estimator for Mixture-of-Experts (MoE) architectures (e.g., Mixtral, MoE Llama).
    Adds extra expert parallelism block parameter count to LlamaMemoryEstimator logic.
    """

    @property
    def model_memory(self) -> float:
        """
        Accounts for the increase in model parameters due to additional expert MLP blocks in MoE Models.

        Returns the estimated memory size of the MoE Model (in GB).
        """
        c = self.llm_config
        # Attention parameter count (Llama-style)
        embedding_params, attn_params = self._calc_attn_embed_params()

        # MoE MLP params per layer
        moe_params_per_layer = (
            c.num_local_experts * 3 * c.hidden_size * c.intermediate_size
        )
        total_params = (
            c.num_hidden_layers * (attn_params + moe_params_per_layer)
            + embedding_params
        )

        # Convert to GB
        return total_params * c.bytes_per_parameter / 1e9


def get_estimator(llm_config, **kwargs) -> MemoryEstimator:
    """
    Extracts the correct estimator based on the defined parameters in the config.json
    See constants.py for LLMConfig parameters necessary for specific estimators.
    Uses MemoryEstimator as a fallback if parameters needed for GQA and MoE Architectures are missing.

    Returns the appropriate MemoryEstimator based on the fields defined by the model's config.json (as represented by LLMConfig).
    """
    if all(
        hasattr(llm_config, f) and getattr(llm_config, f) is not None
        for f in MOE_REQUIRED_FIELDS
    ):
        return MixtureMemoryEstimator(llm_config=llm_config, **kwargs)
    elif all(
        hasattr(llm_config, f) and getattr(llm_config, f) is not None
        for f in LLAMA_REQUIRED_FIELDS
    ):
        return LlamaMemoryEstimator(llm_config=llm_config, **kwargs)
    else:
        logger.warning(
            "Falling back to generic GPT estimator: required fields missing from config.json file in model."
        )
        return MemoryEstimator(llm_config=llm_config, **kwargs)
