#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import re
from typing import Optional, Any, Dict

from pydantic import BaseModel, Field

from ads.aqua.common.errors import AquaRecommendationError
from ads.aqua.shaperecommend.constants import (
    ARCH_AUDIO,
    ARCH_EMBEDDING,
    ARCH_MULTIMODAL,
    ARCH_TEXT_GENERATION,
    ARCH_UNSUPPORTED,
    AUDIO_MODEL_TYPES,
    BITS_AND_BYTES_4BIT,
    BITS_AND_BYTES_8BIT,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_WEIGHT_SIZE,
    EMBEDDING_MODEL_TYPES,
    ENCODER_DECODER_TEXT_MODELS,
    EXCLUDED_MODELS,
    MULTIMODAL_ARCHITECTURE_KEYWORDS,
    MULTIMODAL_MODEL_TYPES,
    NEXT_QUANT,
    QUANT_MAPPING,
    QUANT_METHODS,
    RUNTIME_WEIGHTS,
)
from ads.common.utils import parse_bool


class GeneralConfig(BaseModel):
    num_hidden_layers: int = Field(
        ...,
        description="Number of transformer blocks (layers) in the model's neural network stack.",
    )
    hidden_size: int = Field(
        ..., description="Embedding dimension or hidden size of each layer."
    )
    quantization: Optional[str] = Field(
        None,
        description="Quantization weight (e.g., '8bit', '4bit') or None if unquantized.",
    )
    quantization_type: Optional[str] = Field(
        None,
        description="Quantization method (e.g., '8bit', '4bit', 'gptq', 'awq') or None if unquantized.",
    )
    in_flight_quantization: Optional[str] = Field(
        None,
        description="By setting this, enables recalculation of model footprint using 4bit in-flight quantization",
    )
    weight_dtype: Optional[str] = Field(
        DEFAULT_WEIGHT_SIZE,
        description="Parameter data type: 'float32', 'float16', etc.",
    )

    @staticmethod
    def _get_required_int(raw: dict[str, Any], keys: list[str], field_name: str) -> int:
        """
        Helper to safely extract a required integer field from multiple possible keys.
        Raises AquaRecommendationError if the value is missing or None.
        """
        for key in keys:
            val = raw.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass  # If value exists but isn't a number, keep looking or fail later
        
        # If we reach here, no valid key was found
        raise AquaRecommendationError(
            f"Could not determine '{field_name}' from the model configuration. "
            f"Checked keys: {keys}. "
            "This indicates the model architecture might not be supported or uses a non-standard config structure."
        )

    @classmethod
    def get_weight_dtype(cls, raw: dict) -> str:
        # some configs use a different weight dtype at runtime
        # for runtime weight keys, see RUNTIME_WEIGHTS
        runtime_flags = False
        for flag, dtype in RUNTIME_WEIGHTS.items():
            value = raw.get(flag)
            # only permit use_bfloat16 : true
            if value is True or (isinstance(value, str) and value.lower() == "true"):
                return dtype
            if value is False or (isinstance(value, str) and value.lower() == "false"):
                runtime_flags = True

        # Fallback to torch_dtype if present & no runtime weight dtype
        if not runtime_flags:
            torch_dtype = raw.get("torch_dtype")
            if torch_dtype:
                return str(torch_dtype).lower()

        # if runtime flag present (ex. use_bfloat16: false) or torch_dtype not present
        return DEFAULT_WEIGHT_SIZE

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

        # consider in-flight quantization
        if self.in_flight_quantization in QUANT_MAPPING:
            return QUANT_MAPPING[self.in_flight_quantization]

        # Fallback to dtype mapping
        dtype = (self.weight_dtype or DEFAULT_WEIGHT_SIZE).lower()
        return QUANT_MAPPING.get(dtype, QUANT_MAPPING[DEFAULT_WEIGHT_SIZE])

    @classmethod
    def detect_quantization_type(cls, raw: dict) -> Optional[str]:
        """
        Detects quantization type (e.g., 'gptq', 'bitsandbytes', 'awq', etc.) from Hugging Face config dict.
        """
        qcfg = raw.get("quantization_config", {})
        if raw.get("load_in_8bit") or raw.get("load_in_4bit"):
            return "bitsandbytes"
        for key in QUANT_METHODS:
            if key in str(qcfg).lower() or key in str(raw).lower():
                return key
        return None

    @classmethod
    def detect_quantization_bits(cls, raw: dict) -> Optional[str]:
        """
        Detects quantization bit-width as a string (e.g., '4bit', '8bit') from Hugging Face config dict.
        """
        if raw.get("load_in_8bit"):
            return BITS_AND_BYTES_8BIT
        if raw.get("load_in_4bit"):
            return BITS_AND_BYTES_4BIT
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
        key = (
            self.quantization
            or self.in_flight_quantization
            or self.weight_dtype
            or DEFAULT_WEIGHT_SIZE
        ).lower()
        return NEXT_QUANT.get(key, [])


class VisionConfig(GeneralConfig):
    """
    For transformer-based vision encoder models (part of the image-text-to-text task models),
    parses the module responsible for the vision model.
    """

    mlp_dim: int = Field(
        None,
        description="Size of the MLP/feedforward sub-block in each transformer layer.",
    )
    patch_size: int = (
        Field(
            None,
            description="Image is divided into (patch_size x patch_size) pixel squares.",
        ),
    )
    num_hidden_layers: int = (Field(...),)
    hidden_size: int = Field(...)
    image_size: Optional[int] = (
        Field(
            None,
            description="Input image resolution, affects memory consumption in KV cache.",
        ),
    )
    num_attention_heads: Optional[int] = Field(
        None,
        description="Number of attention heads, impacts the size of attention parameters (model size).",
    )

    @classmethod
    def from_raw_config(cls, vision_section: dict) -> "VisionConfig":
        weight_dtype = cls.get_weight_dtype(vision_section)
        
        num_layers = cls._get_required_int(
            vision_section, 
            ["num_layers", "vision_layers", "num_hidden_layers", "n_layer"], 
            "num_hidden_layers"
        )

        hidden_size = cls._get_required_int(
            vision_section,
            ["hidden_size", "embed_dim"],
            "hidden_size"
        )

        mlp_dim = cls._get_required_int(
            vision_section,
            ["mlp_dim", "intermediate_size"],
            "mlp_dim"
        )

        # Optional fields can use standard .get()
        num_attention_heads = (
            vision_section.get("num_attention_heads")
            or vision_section.get("vision_num_attention_heads")
            or vision_section.get("n_head")
        )

        image_size = vision_section.get("image_size") or vision_section.get(
            "image_resolution"
        )

        patch_size = vision_section.get("patch_size")
        weight_dtype = str(cls.get_weight_dtype(vision_section))

        return cls(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            patch_size=int(patch_size) if patch_size else 0,
            num_attention_heads=int(num_attention_heads)
            if num_attention_heads
            else None,
            weight_dtype=weight_dtype,
            image_size=int(image_size) if image_size else None,
        )


class EmbeddingConfig(GeneralConfig):
    """
    Configuration for embedding models (BERT, RoBERTa, E5-Mistral, etc.).
    Embedding models are typically smaller and throughput-sensitive rather than memory-bound.
    """

    vocab_size: int = Field(..., description="Vocabulary size for input/output tokens.")
    num_attention_heads: Optional[int] = Field(
        None,
        description="Number of attention heads.",
    )
    max_seq_len: Optional[int] = Field(
        512,
        description="Maximum input sequence length (typically 512 for BERT-style models).",
    )
    intermediate_size: Optional[int] = Field(
        None, description="Size of the feedforward layer."
    )
    pooling_type: Optional[str] = Field(
        None, description="Pooling strategy: 'cls', 'mean', etc."
    )

    @classmethod
    def from_raw_config(cls, raw: dict) -> "EmbeddingConfig":
        """Instantiates an EmbeddingConfig from a raw HF config.json."""
        num_hidden_layers = cls._get_required_int(
            raw,
            ["num_hidden_layers", "n_layer", "num_layers"],
            "num_hidden_layers",
        )
        hidden_size = cls._get_required_int(
            raw,
            ["hidden_size", "n_embd", "d_model"],
            "hidden_size",
        )
        vocab_size = cls._get_required_int(raw, ["vocab_size"], "vocab_size")

        num_attention_heads = (
            raw.get("num_attention_heads")
            or raw.get("n_head")
            or raw.get("num_heads")
        )
        intermediate_size = raw.get("intermediate_size")
        max_seq_len = (
            raw.get("max_position_embeddings")
            or raw.get("n_positions")
            or raw.get("max_seq_len")
            or 512
        )
        weight_dtype = cls.get_weight_dtype(raw)
        quantization = cls.detect_quantization_bits(raw)
        quantization_type = cls.detect_quantization_type(raw)

        return cls(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_attention_heads=int(num_attention_heads) if num_attention_heads else None,
            intermediate_size=int(intermediate_size) if intermediate_size else None,
            max_seq_len=int(max_seq_len),
            weight_dtype=weight_dtype,
            quantization=quantization,
            quantization_type=quantization_type,
        )

    @property
    def estimated_params(self) -> int:
        """Rough parameter count for embedding models."""
        embed_params = self.vocab_size * self.hidden_size
        layer_params = 12 * self.num_hidden_layers * (self.hidden_size ** 2)
        return embed_params + layer_params


class WhisperConfig(GeneralConfig):
    """
    Configuration for Whisper-style ASR (Automatic Speech Recognition) models.
    Whisper uses an encoder-decoder architecture with fixed audio input sizes.
    """

    vocab_size: int = Field(..., description="Vocabulary size for decoder tokens.")
    encoder_layers: int = Field(..., description="Number of encoder transformer layers.")
    decoder_layers: int = Field(..., description="Number of decoder transformer layers.")
    d_model: int = Field(..., description="Model dimension (shared between encoder/decoder).")
    encoder_attention_heads: Optional[int] = Field(
        None, description="Number of attention heads in the encoder."
    )
    decoder_attention_heads: Optional[int] = Field(
        None, description="Number of attention heads in the decoder."
    )
    encoder_ffn_dim: Optional[int] = Field(
        None, description="FFN dimension in encoder layers."
    )
    decoder_ffn_dim: Optional[int] = Field(
        None, description="FFN dimension in decoder layers."
    )
    max_source_positions: Optional[int] = Field(
        1500, description="Maximum audio frames (30s of audio at 50 frames/s)."
    )
    max_target_positions: Optional[int] = Field(
        448, description="Maximum decoder output tokens."
    )
    num_mel_bins: Optional[int] = Field(
        128, description="Number of mel-spectrogram frequency bins."
    )

    @classmethod
    def from_raw_config(cls, raw: dict) -> "WhisperConfig":
        """Instantiates a WhisperConfig from a raw HF config.json."""
        vocab_size = cls._get_required_int(raw, ["vocab_size"], "vocab_size")
        d_model = cls._get_required_int(raw, ["d_model"], "d_model")

        encoder_layers = cls._get_required_int(
            raw, ["encoder_layers", "num_hidden_layers"], "encoder_layers"
        )
        decoder_layers = cls._get_required_int(
            raw, ["decoder_layers"], "decoder_layers"
        )

        weight_dtype = cls.get_weight_dtype(raw)

        return cls(
            num_hidden_layers=encoder_layers + decoder_layers,
            hidden_size=d_model,
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_attention_heads=raw.get("encoder_attention_heads"),
            decoder_attention_heads=raw.get("decoder_attention_heads"),
            encoder_ffn_dim=raw.get("encoder_ffn_dim"),
            decoder_ffn_dim=raw.get("decoder_ffn_dim"),
            max_source_positions=raw.get("max_source_positions", 1500),
            max_target_positions=raw.get("max_target_positions", 448),
            num_mel_bins=raw.get("num_mel_bins", 128),
            weight_dtype=weight_dtype,
        )

    @property
    def estimated_params(self) -> int:
        """Rough parameter count for Whisper models."""
        # Encoder + Decoder: each layer ~12 * d_model^2, plus embeddings
        layer_params = 12 * (self.encoder_layers + self.decoder_layers) * (self.d_model ** 2)
        embed_params = self.vocab_size * self.d_model
        return layer_params + embed_params


class LLMConfig(GeneralConfig):
    """
    Standardized configuration object for evaluating the size of Large Language Models (LLMs)
    based on their architecture and quantization.
    """

    vocab_size: int = Field(..., description="Vocabulary size for input/output tokens.")
    num_attention_heads: int = Field(
        ...,
        description="Number of attention heads (used for queries and to determine head_dim).",
    )
    num_hidden_layers: int = Field(...)
    hidden_size: int = Field(...)

    head_dim: int = Field(
        ...,
        description="Dimension of each attention head. Typically hidden_size // num_attention_heads.",
    )
    max_seq_len: Optional[int] = Field(
        DEFAULT_MAX_SEQ_LEN,
        description="Maximum input sequence length (context window).",
    )
    weight_dtype: Optional[str] = Field(
        DEFAULT_WEIGHT_SIZE,
        description="Parameter data type: 'float32', 'float16', etc.",
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

    tie_word_embeddings: Optional[bool] = Field(
        True,
        description="If True, input and output embedding matrices share the same parameters in memory.",
    )

    trust_remote_code: Optional[bool] = Field(
        False, description="If True, the model requires custom code to operate."
    )

    def calculate_possible_seq_len(self, min_len=2048):
        """
        Calculates a list of possible sequence lengths (in tokens).
        [2048, ... max-length] (max-length found in model's config.json file)
        """
        vals = []
        curr = min_len
        while curr <= self.max_seq_len:
            vals.append(curr)
            curr *= 2
        if vals and vals[-1] != self.max_seq_len:
            vals.append(self.max_seq_len)
        return vals

    def optimal_config(self):
        """
        Builds a list of optimal configuration parameters (sorted descending). Combination of:
            - Quantization / weight sizes: bfloat16 weight size -> 8bit -> 4bit
            - max-model-len: power-of-two model lengths from max length (config.json of model) to 2048 tokens.

        Example:
        [('bfloat16', max_model_len supported by model) ('bfloat16', 1/2 of max_model_len) ... ('4bit', 4096), ('4bit', 2048)]

        """
        # use later-Create a copy of the suggested_quantizations list
        # quantizations = self.suggested_quantizations[:]
        quantizations = ["bfloat16", "4bit"]

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
    def validate_model_support(cls, raw: dict):
        """
        Validates if model is decoder-only text generation.
        
        Note: This validation is only called when the model has already been
        routed to the text-generation strategy. Audio, embedding, and multimodal
        models are handled by their respective strategies via ParsedModelConfig.detect_architecture().
        """
        # Known unsupported model architectures or types
        excluded_models = EXCLUDED_MODELS
        
        model_type = raw.get("model_type", "").lower()
        
        if model_type in excluded_models:
            raise AquaRecommendationError(
                f"The model type '{model_type}' is not supported. "
                "Please provide a decoder-only text-generation model (ex. Llama, Falcon, etc). "
                "Encoder-decoder models (ex. T5, Gemma), encoder-only (BERT), and audio models (Whisper) are not supported at this time."
            )

        if (
            raw.get("is_encoder_decoder", False)  # exclude encoder-decoder models
            or (
                raw.get("is_decoder") is False
            )  # exclude explicit encoder-only models
        ):
            raise AquaRecommendationError(
                "Please provide a decoder-only text-generation model (ex. Llama, Falcon, etc). "
                "Encoder-decoder models (ex. T5, Gemma) and encoder-only (BERT) are not supported at this time."
            )

    @classmethod
    def from_raw_config(cls, raw: dict) -> "LLMConfig":
        """
        Instantiates an LLMConfig from a raw Hugging Face config.json file,
        using robust key detection and fallback for architecture.
        """
        cls.validate_model_support(raw)

        # Field mappings with fallback using safe extraction
        num_hidden_layers = cls._get_required_int(
            raw, 
            ["num_hidden_layers", "n_layer", "num_layers"], 
            "num_hidden_layers"
        )

        hidden_size = cls._get_required_int(
            raw,
            ["hidden_size", "n_embd", "d_model"],
            "hidden_size"
        )
        
        num_attention_heads = cls._get_required_int(
            raw,
            ["num_attention_heads", "n_head", "num_heads"],
            "num_attention_heads"
        )
        
        # Vocab size might be missing in some architectures, but usually required for memory calc
        vocab_size = cls._get_required_int(
            raw,
            ["vocab_size"],
            "vocab_size"
        )

        weight_dtype = cls.get_weight_dtype(raw)
        quantization = cls.detect_quantization_bits(raw)
        quantization_type = cls.detect_quantization_type(raw)

        if not quantization and quantization_type in QUANT_MAPPING:
            quantization = quantization_type

        num_key_value_heads = (
            raw.get("num_key_value_heads")  # GQA models (ex. Llama-type)
        )

        head_dim = raw.get("head_dim") or (
            int(hidden_size) // int(num_attention_heads)
            if hidden_size and num_attention_heads
            else None
        )
        
        # Ensure head_dim is not None if calculation failed
        if head_dim is None:
            raise AquaRecommendationError(
                "Could not determine 'head_dim' and it could not be calculated from 'hidden_size' and 'num_attention_heads'."
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

        raw_tie_word_embeddings = raw.get("tie_word_embeddings", True)
        tie_word_embeddings = parse_bool(raw_tie_word_embeddings)

        trust_remote_code = (
            "auto_map" in raw
        )  # trust-remote-code is always needed when this key is present

        return cls(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=int(head_dim),
            vocab_size=vocab_size,
            weight_dtype=weight_dtype,
            quantization=quantization,
            quantization_type=quantization_type,
            max_seq_len=int(max_seq_len),
            num_local_experts=num_local_experts,
            intermediate_size=intermediate_size,
            tie_word_embeddings=tie_word_embeddings,
            trust_remote_code=trust_remote_code,
        )


class ParsedModelConfig(BaseModel):
    """
    Represents the configuration for a model, supporting text-only, vision-only,
    multimodal (text + vision), embedding, or audio architectures.

    Attributes
    ----------
    architecture_type : str
        Detected architecture type (one of ARCH_* constants).
    llm_config : Optional[LLMConfig]
        Parsed configuration for the text-generation (language) model, if present.
    vision_config : Optional[VisionConfig]
        Parsed configuration for the vision/image encoder, if present.
    embedding_config : Optional[EmbeddingConfig]
        Parsed configuration for embedding models, if present.
    whisper_config : Optional[WhisperConfig]
        Parsed configuration for Whisper/ASR models, if present.

    Notes
    -----
    If both `llm_config` and `vision_config` are defined, this represents a multimodal model.
    If only `llm_config` is defined, this represents a text-generation model.
    If only `embedding_config` is defined, this represents an embedding model.
    If only `whisper_config` is defined, this represents an audio model.
    """

    architecture_type: str = Field(
        ARCH_TEXT_GENERATION,
        description="Detected architecture type for strategy selection.",
    )
    llm_config: Optional[LLMConfig] = Field(
        None,
        description="Parsed configuration of the text-generation model if present.",
    )
    vision_config: Optional[VisionConfig] = Field(
        None, description="Parsed configuration of the vision model if present."
    )
    embedding_config: Optional[EmbeddingConfig] = Field(
        None, description="Parsed configuration of the embedding model if present."
    )
    whisper_config: Optional[WhisperConfig] = Field(
        None, description="Parsed configuration of the Whisper/ASR model if present."
    )

    @classmethod
    def detect_architecture(cls, raw: dict, task_hint: Optional[str] = None) -> str:
        """
        Detects the model architecture type from a raw config.json dictionary.

        Parameters
        ----------
        raw : dict
            The raw config.json dictionary.
        task_hint : Optional[str]
            Optional task tag from model metadata (e.g., from OCI freeform_tags).

        Returns
        -------
        str
            One of ARCH_TEXT_GENERATION, ARCH_MULTIMODAL, ARCH_EMBEDDING, ARCH_AUDIO, ARCH_UNSUPPORTED.
        """
        model_type = raw.get("model_type", "").lower()
        architectures = [a.lower() for a in raw.get("architectures", [])]
        task = (task_hint or "").lower().replace("-", "_")

        # 1. Audio / Whisper detection (highest specificity)
        if model_type in AUDIO_MODEL_TYPES:
            return ARCH_AUDIO
        if any("whisper" in a for a in architectures):
            return ARCH_AUDIO

        # 2. Encoder-decoder text models (unsupported)
        if model_type in ENCODER_DECODER_TEXT_MODELS:
            return ARCH_UNSUPPORTED
        if raw.get("is_encoder_decoder", False) and model_type not in AUDIO_MODEL_TYPES:
            return ARCH_UNSUPPORTED

        # 3. Multimodal detection
        if model_type in MULTIMODAL_MODEL_TYPES:
            return ARCH_MULTIMODAL
        if raw.get("vision_config") or raw.get("vision_encoder_config"):
            return ARCH_MULTIMODAL
        # Check nested keys that hint at vision
        has_vision_key = any(
            "vision" in k and isinstance(v, dict)
            for k, v in raw.items()
        )
        has_text_key = any(
            k in raw and isinstance(raw[k], dict)
            for k in ("text_config", "llm_config", "language_model")
        )
        if has_vision_key and has_text_key:
            return ARCH_MULTIMODAL
        # Check architecture keywords
        for arch in architectures:
            for keyword in MULTIMODAL_ARCHITECTURE_KEYWORDS:
                if keyword in arch:
                    return ARCH_MULTIMODAL
        # Task-based multimodal detection
        if task in ("image_text_to_text",):
            return ARCH_MULTIMODAL

        # 4. Embedding detection
        if model_type in EMBEDDING_MODEL_TYPES:
            return ARCH_EMBEDDING
        if task in ("feature_extraction",):
            return ARCH_EMBEDDING
        if any("embeddingmodel" in a or "formaskedlm" in a for a in architectures):
            return ARCH_EMBEDDING

        # 5. Default: text generation (decoder-only)
        return ARCH_TEXT_GENERATION

    @classmethod
    def get_model_config(cls, raw: dict, task_hint: Optional[str] = None) -> "ParsedModelConfig":
        """
        Instantiates a ParsedModelConfig by parsing a raw config dictionary.

        Parameters
        ----------
        raw : dict
            Raw configuration dictionary to parse.
        task_hint : Optional[str]
            Optional task tag from model metadata.

        Returns
        -------
        ParsedModelConfig
            An instance with the relevant sub-configurations set based on detected architecture.

        Raises
        ------
        AquaRecommendationError
            If the configuration cannot be parsed for the detected architecture.
        """
        arch_type = cls.detect_architecture(raw, task_hint)

        # --- Audio (Whisper) ---
        if arch_type == ARCH_AUDIO:
            whisper_config = WhisperConfig.from_raw_config(raw)
            return cls(architecture_type=arch_type, whisper_config=whisper_config)

        # --- Unsupported ---
        if arch_type == ARCH_UNSUPPORTED:
            model_type = raw.get("model_type", "unknown")
            raise AquaRecommendationError(
                f"The model type '{model_type}' is not supported for shape recommendation. "
                "Encoder-decoder text generation models (e.g., T5, BART) are not supported at this time."
            )

        # --- Embedding ---
        if arch_type == ARCH_EMBEDDING:
            embedding_config = EmbeddingConfig.from_raw_config(raw)
            return cls(architecture_type=arch_type, embedding_config=embedding_config)

        # --- Multimodal ---
        if arch_type == ARCH_MULTIMODAL:
            # Find nested text section
            text_section = (
                raw.get("text_config")
                or raw.get("llm_config")
                or raw.get("language_model")
                or raw.get("language_model_config")
                or raw.get("decoder_config")
                or raw.get("model_config")
                or raw.get("base_model")
                or raw.get("gpt_config")
                or next(
                    (
                        v
                        for k, v in raw.items()
                        if ("text" in k or "llm" in k or "gpt" in k) and isinstance(v, dict)
                    ),
                    None,
                )
            )
            # Find nested vision section
            vision_section = (
                raw.get("vision_config")
                or raw.get("vision_encoder_config")
                or next(
                    (v for k, v in raw.items() if "vision" in k and isinstance(v, dict)),
                    None,
                )
            )

            llm_config = None
            vision_config = None

            if text_section:
                llm_config = LLMConfig.from_raw_config(text_section)
            if vision_section:
                vision_config = VisionConfig.from_raw_config(vision_section)

            if not llm_config and not vision_config:
                raise AquaRecommendationError(
                    "Detected multimodal model but could not parse text or vision sub-configs. "
                    "Ensure config.json contains 'text_config'/'llm_config' and/or 'vision_config'."
                )

            return cls(
                architecture_type=arch_type,
                llm_config=llm_config,
                vision_config=vision_config,
            )

        # --- Text Generation (default) ---
        # Try nested text section first, then flat
        text_section = (
            raw.get("text_config")
            or raw.get("llm_config")
            or raw.get("language_model")
            or raw.get("language_model_config")
            or raw.get("decoder_config")
            or raw.get("model_config")
            or raw.get("base_model")
            or raw.get("gpt_config")
            or next(
                (
                    v
                    for k, v in raw.items()
                    if ("text" in k or "llm" in k or "gpt" in k) and isinstance(v, dict)
                ),
                None,
            )
        )

        if text_section:
            llm_config = LLMConfig.from_raw_config(text_section)
        else:
            llm_config = LLMConfig.from_raw_config(raw)

        return cls(architecture_type=arch_type, llm_config=llm_config)


# Keep backward compatibility alias
ModelConfig = ParsedModelConfig