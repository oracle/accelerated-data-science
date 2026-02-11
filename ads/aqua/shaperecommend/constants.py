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

EXCLUDED_MODELS contains a set of model identifiers that are known to be unsupported for shape recommendation.

ARCHITECTURE_TYPE identifies the detected model architecture category for strategy selection.

SUPPORTED_TASKS defines the set of model task types that the recommender can handle.
"""

# ---------------------------------------------------------------------------
# Architecture type identifiers (used by StrategyFactory)
# ---------------------------------------------------------------------------
ARCH_TEXT_GENERATION = "text_generation"
ARCH_MULTIMODAL = "multimodal"
ARCH_EMBEDDING = "embedding"
ARCH_AUDIO = "audio"
ARCH_UNSUPPORTED = "unsupported"

# ---------------------------------------------------------------------------
# Supported task tags (from HF / OCI freeform_tags)
# ---------------------------------------------------------------------------
SUPPORTED_TASKS = {
    "text_generation",
    "text-generation",
    "image_text_to_text",
    "image-text-to-text",
    "feature_extraction",
    "feature-extraction",
    "automatic_speech_recognition",
    "automatic-speech-recognition",
}

# ---------------------------------------------------------------------------
# Model types that map to specific architecture strategies
# ---------------------------------------------------------------------------
MULTIMODAL_MODEL_TYPES = {
    "llava",
    "llava_next",
    "llava_onevision",
    "qwen2_vl",
    "internvl",
    "phi3_v",
    "pixtral",
    "idefics2",
    "idefics3",
    "mllama",
    "paligemma",
}

EMBEDDING_MODEL_TYPES = {
    "bert",
    "roberta",
    "xlm-roberta",
    "modernbert",
    "nomic_bert",
}

AUDIO_MODEL_TYPES = {
    "whisper",
}

# Architecture keywords in HF 'architectures' list that indicate multimodal
MULTIMODAL_ARCHITECTURE_KEYWORDS = {
    "llava",
    "vila",
    "nemotron_vl",
    "nemotron_nano_vl",
    "qwen2vl",
    "internvl",
    "phi3v",
    "pixtral",
    "idefics",
    "paligemma",
    "mllama",
}

LLAMA_REQUIRED_FIELDS = [
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "vocab_size",
]

MOE_REQUIRED_FIELDS = LLAMA_REQUIRED_FIELDS + ["num_local_experts", "intermediate_size"]

NEXT_QUANT = {
    "float32": ["8bit", "4bit"],
    "bfloat16": ["8bit", "4bit"],
    "float16": ["8bit", "4bit"],
    "int8": ["4bit"],
    "fp8": ["4bit"],
    "8bit": ["4bit"],
    "int4": ["No smaller quantization available"],
    "4bit": ["No smaller quantization available"],
}

RUNTIME_WEIGHTS = {
    "use_bfloat16": "bfloat16",
    "use_fp16": "float16",
    "use_fp32": "float32",
    "use_int8": "int8",
    "use_int4": "int4",
    "use_bfloat32": "bfloat32",
}

TEXT_GENERATION = "text_generation"
SAFETENSORS = "safetensors"

QUANT_METHODS = [
    "aqlm",
    "awq",
    "deepspeedfp",
    "tpu_int8",
    "fp8",
    "ptpc_fp8",
    "fbgemm_fp8",
    "modelopt",
    "modelopt_fp4",
    "marlin",
    "bitblas",
    "gguf",
    "gptq_marlin_24",
    "gptq_marlin",
    "gptq_bitblas",
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "qqq",
    "hqq",
    "experts_int8",
    "neuron_quant",
    "ipex",
    "quark",
    "moe_wna16",
    "torchao",
    "auto-round",
    "rtn",
    "inc",
    "mxfp4",
]

IN_FLIGHT_QUANTIZATION = {"4bit"}  # vLLM only supports 4bit in-flight-quantization

VLLM_PARAMS_FAMILY = "VLLM_PARAMS"
VLLM_ENV = "VLLM"

QUANT_FLAG = "--quantization"
WEIGHT_DTYPE_FLAG = "--dtype"
MAX_MODEL_LEN_FLAG = "--max-model-len"

TROUBLESHOOT_MSG = "The selected model is too large to fit on standard GPU shapes with the current configuration.\nAs troubleshooting, we have suggested the two largest available GPU shapes using the smallest quantization level ('4bit') to maximize chances of fitting the model. "

VLLM_PARAMS = {
    "max_model_len": "--max-model-len",
    "in_flight_quant": "--quantization bitsandbytes --load-format bitsandbytes",
    "trust_remote_code": "--trust-remote-code",
    "task_embedding": "--task embedding",
    "task_transcribe": "--task transcribe",
    "limit_mm_per_prompt_image": '--limit-mm-per-prompt {"image": 1}',
    "limit_mm_per_prompt_audio": '--limit-mm-per-prompt {"audio": 1}',
    "enforce_eager": "--enforce-eager",
}

DEFAULT_WEIGHT_SIZE = "float32"
DEFAULT_MAX_SEQ_LEN = 4096

BITS_AND_BYTES_8BIT = "8bit"
BITS_AND_BYTES_4BIT = "4bit"

BITSANDBYTES = "bitsandbytes"


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

SHAPE_MAP = {
    "NVIDIA_GPU": "GPU",
    "AMD_ROME": "CPU",
    "GENERIC": "CPU",
    "LEGACY": "CPU",
    "ARM": "CPU",
    "UNKNOWN_ENUM_VALUE": "N/A",
}
# Models that are truly unsupported (encoder-decoder text gen, no vLLM support)
EXCLUDED_MODELS = {
    "t5",
    "bart",
    "albert",
    "t5gemma",
}

# Encoder-decoder text models that cannot be served via standard vLLM text generation
ENCODER_DECODER_TEXT_MODELS = {
    "t5",
    "bart",
    "albert",
    "t5gemma",
    "ul2",
    "longt5",
    "pegasus",
}