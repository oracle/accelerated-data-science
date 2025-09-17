#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from typing import Optional

from pydantic import BaseModel, Field

from ads.aqua.app import logger
from ads.aqua.shaperecommend.constants import (
    IN_FLIGHT_QUANTIZATION,
    LLAMA_REQUIRED_FIELDS,
    MOE_REQUIRED_FIELDS,
    NEXT_QUANT,
    QUANT_MAPPING,
    VLLM_PARAMS,
)
from ads.aqua.shaperecommend.llm_config import LLMConfig


class MemoryEstimator(BaseModel):
    """
    The generic estimator for Transformer Architecture models (OPT/ Bloom)
    Used as a fallback estimator if model identified is not a MoE or GQA Architecture Model.
    Has properties to estimate the KV Cache size, Model size, and total footprint (KV Cache + Model size)

    KV cache: Use num_attention_heads (all heads, no GQA)
    Parameter estimation: Standard decoder-only, untied embeddings possible
    """

    llm_config: LLMConfig = Field(
        ...,
        description="The model's config.json file with the necessary parameters for model size and KV cache estimation.",
    )
    batch_size: Optional[int] = (
        1  # we assume that estimation for batch sizes are not supported yet
    )
    seq_len: int = Field(
        ..., description="The max-seq-len to estimate the size of the KV cache."
    )

    @property
    def kv_cache_memory(self) -> float:
        """
        Estimates the KV cache size (in GB) using the LLM config.json parameters.

        Uses num_attention_heads (assumes no GQA, each attention head has its own query, key, value) for estimation.
        """
        seq_len = self.seq_len or self.llm_config.max_seq_len
        llm_config = self.llm_config
        kv_cache_dtype_bytes = QUANT_MAPPING.get(
            llm_config.weight_dtype, 2
        )  # vLLM uses model's weight applied to KV cache

        total_bytes = (
            self.batch_size
            * llm_config.num_hidden_layers
            * 2
            * llm_config.num_attention_heads
            * seq_len
            * llm_config.head_dim
            * kv_cache_dtype_bytes
        )
        return total_bytes / 1e9

    @property
    def model_memory(self) -> float:
        """
        Estimates the model size (in GB) based on estimating the model parameter size and model weights.

        Model Parameter estimation: Standard decoder-only, untied/tied embeddings possible.
        """
        llm_config = self.llm_config
        embedding_count = 1 if llm_config.tie_word_embeddings else 2
        embedding_params = (
            embedding_count * llm_config.vocab_size * llm_config.hidden_size
        )  # input and output untied
        layer_params = (
            12 * llm_config.num_hidden_layers * (llm_config.hidden_size**2)
        )  # GPT-style
        num_params = layer_params + embedding_params

        return num_params * llm_config.bytes_per_parameter / 1e9

    @property
    def total_memory(self) -> float:
        """
        Computes the total memory footprint of the model (KV cache & model size from estimated parameters).
        """
        return self.model_memory + self.kv_cache_memory

    def validate_shape(
        self, allowed_gpu_memory: float, gpu_utilization: float = 0.9
    ) -> bool:
        """
        Validates if a given model estimator fits within the allowed GPU memory budget, using a fixed utilization margin.

        Parameters
        ----------
        estimator : MemoryEstimator
            The estimator with current shape/memory needs.
        allowed_gpu_memory : float
            The maximum allowed GPU memory.

        Returns
        -------
        bool
            True if estimator uses less than adjusted GPU memory, else False.
        """
        return (allowed_gpu_memory * gpu_utilization) > self.total_memory

    def construct_deployment_params(self) -> str:
        """
        Constructs a deployment parameter string for the model.

        This method assembles runtime configuration parameters to be passed
        during model deployment. It:
        - Overrides the max sequence length if a shorter length is provided.
        - Suggests in-flight quantization **only if the model is unquantized**
            and in-flight quantization (such as '4bit') is requested in config.

        Returns
        -------
            str: Parameter string for model deployment.
        """
        llm_config = self.llm_config
        params = []
        if self.seq_len < llm_config.max_seq_len:
            params.append(VLLM_PARAMS["max_model_len"])
            params.append(str(self.seq_len))

        # Only suggest in-flight quantization for unquantized models when such quantization is requested
        if (
            not llm_config.quantization
            and llm_config.in_flight_quantization in IN_FLIGHT_QUANTIZATION
        ):
            # vLLM only supports 4bit in-flight quantization
            params.append(VLLM_PARAMS["in_flight_quant"])

        # add trust-remote-code if custom modules are specified
        if llm_config.trust_remote_code:
            params.append(VLLM_PARAMS["trust_remote_code"])

        params = " ".join(params) if params else ""
        return params

    def suggest_param_advice(self, allowed: float) -> str:
        """
        Suggests parameter modifications to help a model fit within GPU memory limits.

        Parameters
        ----------
        estimator : MemoryEstimator
            The memory estimator object.
        allowed : float
            Allowed GPU memory in GB.

        Returns
        -------
        str
            Advice message with suggestions.
        """
        kv_gb = self.kv_cache_memory
        wt_gb = self.model_memory
        batch_size = self.batch_size
        seq_len = self.seq_len
        weight_size = self.llm_config.weight_dtype
        config = self.llm_config

        suggested_quant_msg = None
        quant_advice = ", ".join(config.suggested_quantizations)
        quantization = config.quantization

        advice = []

        if config.suggested_quantizations:
            to_do = f", which is smaller than the current {quantization if quantization in NEXT_QUANT else weight_size} format."
            if "No" in quant_advice:
                suggested_quant_msg = "No smaller quantized version exists. Use a model with fewer parameters."
            elif not quant_advice:
                suggested_quant_msg = (
                    "Use a quantized version of the same model (e.g., INT8 or other)"
                    + to_do
                )
            else:
                suggested_quant_msg = (
                    f"Either use a pre-quantized model at {quant_advice}, or apply in-flight {quant_advice} quantization"
                    + to_do
                )

        kv_advice = [f"Reduce maximum context length (set --max-model-len < {seq_len})"]

        if batch_size != 1:
            kv_advice.append(f"Reduce batch size to less than {batch_size}.")

        wt_advice = [
            "Use a model with fewer parameters.",
            f"{suggested_quant_msg}" if suggested_quant_msg else "",
        ]

        if kv_gb > wt_gb and kv_gb > allowed * 0.5:
            main = "KV cache memory usage is the main limiting factor"
            advice = kv_advice
        elif wt_gb > kv_gb and wt_gb > allowed * 0.5:
            main = "Model weights are the main limiting factor"
            advice = wt_advice
        else:
            main = "Both model weights and KV cache are significant contributors to memory use"
            advice = kv_advice
            advice.extend(wt_advice)

        advice_str = "\n".join(f"{i}. {item}" for i, item in enumerate(advice, 1))

        return (
            f"{advice_str}\n\n{main} (KV cache: {kv_gb:.1f}GB, Weights: {wt_gb:.1f}GB)."
        )

    def limiting_factor(
        self, allowed_gpu_memory: float, warn_delta: float = 0.85
    ) -> str:
        """
        Determines the memory limiting factor for a model deployment and returns advice.

        Parameters
        ----------
        estimator : MemoryEstimator
            The memory estimator object with current model configuration.
        allowed_gpu_memory : float
            The maximum allowed GPU memory (in GBs).
        warn_delta : float, optional
            The threshold (fraction) of allowed GPU memory to trigger a warning (default=0.85).

        Returns
        -------
        str
            Advice message about model fit and limiting factors.
        """
        required = self.total_memory

        # Warn if required is close to but under allowed
        if allowed_gpu_memory > required > allowed_gpu_memory * warn_delta:
            model_params = self.suggest_param_advice(allowed_gpu_memory)
            advice = (
                f"While the selected compute shape is estimated to work "
                f"({required:.1f}GB used / {allowed_gpu_memory:.1f}GB allowed), "
                f"the model configuration is close to the GPU memory limit.\n\n"
                "If you encounter issues with this shape, consider the following options to reduce memory usage:\n\n"
                f"{model_params.lstrip()}"
            )
        elif required > allowed_gpu_memory:
            model_params = self.suggest_param_advice(allowed_gpu_memory)
            advice = (
                f"Model does not fit within GPU memory budget. "
                "Consider the following options to reduce memory usage:\n\n"
                f"{model_params.lstrip()}"
            )
        else:
            advice = (
                f"Model fits well within the allowed compute shape "
                f"({required:.1f}GB used / {allowed_gpu_memory:.1f}GB allowed)."
            )
        return advice


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
        llm_config = self.llm_config

        embedding_params, attn_params = self._calc_attn_embed_params()

        # MLP params
        gate_proj = llm_config.hidden_size * llm_config.intermediate_size
        up_proj = llm_config.hidden_size * llm_config.intermediate_size
        down_proj = llm_config.intermediate_size * llm_config.hidden_size
        mlp_params = gate_proj + up_proj + down_proj

        # Total per-layer
        layer_params = attn_params + mlp_params
        # Total params
        num_params = llm_config.num_hidden_layers * layer_params + embedding_params

        return num_params * llm_config.bytes_per_parameter / 1e9

    @property
    def kv_cache_memory(self) -> float:
        """
        Returns estimated KV cache memory in GB for GQA models.

        Grouped Query Attention uses num_key_value_heads, which groups of Q heads share a K and V projection.
        num_key_value_heads < num_attention_heads, which reduces the KV Cache size.
        """
        llm_config = self.llm_config
        seq_len = self.seq_len or llm_config.max_seq_len
        kv_cache_dtype_bytes = QUANT_MAPPING.get(llm_config.weight_dtype, 2)
        kv_heads = llm_config.num_key_value_heads

        total_bytes = (
            self.batch_size
            * llm_config.num_hidden_layers
            * 2
            * kv_heads
            * seq_len
            * llm_config.head_dim
            * kv_cache_dtype_bytes
        )
        return total_bytes / 1e9

    def _calc_attn_embed_params(self) -> tuple:
        """
        Returns the embedding parameter count and attention parameter count for Llama-family (GQA) models.
        """
        llm_config = self.llm_config

        # Embedding parameters
        # assume tied embeddings unless tie_word_embeddings = False
        embedding_count = 1 if llm_config.tie_word_embeddings else 2
        embedding_params = (
            embedding_count * llm_config.vocab_size * llm_config.hidden_size
        )

        q_proj = llm_config.hidden_size * llm_config.hidden_size
        k_proj = llm_config.hidden_size * (
            llm_config.num_key_value_heads * llm_config.head_dim
        )
        v_proj = llm_config.hidden_size * (
            llm_config.num_key_value_heads * llm_config.head_dim
        )
        o_proj = llm_config.hidden_size * llm_config.hidden_size
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
        llm_config = self.llm_config
        # Attention parameter count (Llama-style)
        embedding_params, attn_params = self._calc_attn_embed_params()

        # MoE MLP params per layer
        moe_params_per_layer = (
            llm_config.num_local_experts
            * 3
            * llm_config.hidden_size
            * llm_config.intermediate_size
        )
        total_params = (
            llm_config.num_hidden_layers * (attn_params + moe_params_per_layer)
            + embedding_params
        )

        # Convert to GB
        return total_params * llm_config.bytes_per_parameter / 1e9


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
