#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import re
from unittest.mock import MagicMock

import pytest

from ads.aqua.common.entities import ComputeShapeSummary
from ads.aqua.common.errors import AquaRecommendationError
from ads.aqua.shaperecommend.estimator import (
    LlamaMemoryEstimator,
    MemoryEstimator,
    MixtureMemoryEstimator,
    get_estimator,
)
from ads.aqua.shaperecommend.llm_config import LLMConfig
from ads.aqua.shaperecommend.recommend import AquaShapeRecommend
from ads.aqua.shaperecommend.shape_report import (
    DeploymentParams,
    ModelConfig,
    ModelDetail,
    RequestRecommend,
    ShapeRecommendationReport,
    ShapeReport,
)
from ads.model.model_metadata import ModelCustomMetadata, ModelProvenanceMetadata

CONFIG_ROOT = os.path.join(os.path.dirname(__file__), "test_data/recommend/")


def load_config(filename):
    with open(os.path.join(CONFIG_ROOT, filename)) as f:
        return json.load(f)


# --- Tests for estimator.py ---
class TestMemoryEstimator:
    def test_memory_estimator_properties(self):
        config = LLMConfig(
            num_hidden_layers=2,
            hidden_size=64,
            vocab_size=1000,
            num_attention_heads=4,
            head_dim=16,
            weight_dtype="float32",
        )
        estimator = MemoryEstimator(llm_config=config, seq_len=128, batch_size=2)
        assert estimator.model_memory > 0
        assert estimator.kv_cache_memory > 0
        assert estimator.total_memory == pytest.approx(
            estimator.model_memory + estimator.kv_cache_memory
        )

    def test_get_estimator_llama_and_moe_fields(self):
        base_args = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "vocab_size": 1000,
            "num_attention_heads": 4,
            "head_dim": 16,
            "weight_dtype": "float32",
            "num_key_value_heads": 2,
            "intermediate_size": 256,
        }
        config_reg = LLMConfig(**base_args)
        base_args["num_key_value_heads"] = 2
        config_llama = LLMConfig(**base_args)
        base_args["num_local_experts"] = 4
        config_moe = LLMConfig(**base_args)

        assert isinstance(
            get_estimator(config_reg, seq_len=128, batch_size=1), MemoryEstimator
        )
        assert isinstance(
            get_estimator(config_llama, seq_len=128, batch_size=1), LlamaMemoryEstimator
        )
        assert isinstance(
            get_estimator(config_moe, seq_len=128, batch_size=1), MixtureMemoryEstimator
        )

    @pytest.mark.parametrize(
        "config_file,should_raise",
        [
            ("Devstral-Small-2507-GQA.json", False),
            ("Kimi-K2-Instruct-MOE.json", False),
            ("Qwen3-235B-A22B-Instruct-2507-FP8.json", False),
            ("t5gemma-ml-ml-prefixlm.json", True),  # This one is expected to raise
        ],
    )
    def test_memory_estimator_properties_from_file(self, config_file, should_raise):
        raw = load_config(config_file)
        if should_raise:
            with pytest.raises(AquaRecommendationError):
                config = LLMConfig.from_raw_config(raw)
                MemoryEstimator(llm_config=config, seq_len=128, batch_size=2)
        else:
            config = LLMConfig.from_raw_config(raw)
            estimator = MemoryEstimator(llm_config=config, seq_len=128, batch_size=2)
            assert estimator.model_memory > 0
            assert estimator.kv_cache_memory > 0
            assert estimator.total_memory == pytest.approx(
                estimator.model_memory + estimator.kv_cache_memory
            )

    @pytest.mark.parametrize(
        "config_file, expected_estimator_cls",
        [
            ("Devstral-Small-2507-GQA.json", LlamaMemoryEstimator),
            ("Kimi-K2-Instruct-MOE.json", MixtureMemoryEstimator),
            ("Qwen3-235B-A22B-Instruct-2507-FP8.json", MixtureMemoryEstimator),
        ],
    )
    def test_get_estimator_types_from_config_file(
        self, config_file, expected_estimator_cls
    ):
        raw = load_config(config_file)
        config = LLMConfig.from_raw_config(raw)
        estimator = get_estimator(config, seq_len=128, batch_size=1)
        assert isinstance(estimator, expected_estimator_cls)


# --- Tests for llm_config.py ---
class TestLLMConfig:
    def test_llm_config_from_raw_config(self):
        raw = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "vocab_size": 1000,
            "num_attention_heads": 4,
            "head_dim": 16,
            "torch_dtype": "float16",
            "max_position_embeddings": 2048,
        }
        config = LLMConfig.from_raw_config(raw)
        assert config.hidden_size == 64
        assert config.max_seq_len == 2048
        assert config.weight_dtype.lower() == "float16"
        assert config.quantization is None

    @pytest.mark.parametrize(
        "config_file, expected_hidden_size, expected_max_seq_len, expected_dtype, exp_num_key_value_heads, exp_num_local_experts, expected_head_dim, expected_quant",
        [
            (
                "Devstral-Small-2507-GQA.json",
                5120,
                131072,
                "bfloat16",
                8,
                None,
                128,
                None,
            ),
            (
                "Kimi-K2-Instruct-MOE.json",
                7168,
                131072,
                "bfloat16",
                64,
                384,
                112,
                "fp8",
            ),
            (
                "Qwen3-235B-A22B-Instruct-2507-FP8.json",
                4096,
                262144,
                "bfloat16",
                4,
                128,
                128,
                "fp8",
            ),
        ],
    )
    def test_llm_config_from_raw_config_file(
        self,
        config_file,
        expected_hidden_size,
        expected_max_seq_len,
        expected_dtype,
        exp_num_key_value_heads,
        exp_num_local_experts,
        expected_head_dim,
        expected_quant,
    ):
        raw = load_config(config_file)
        config = LLMConfig.from_raw_config(raw)
        assert config.hidden_size == expected_hidden_size
        assert config.max_seq_len == expected_max_seq_len
        assert config.num_key_value_heads == exp_num_key_value_heads
        assert config.num_local_experts == exp_num_local_experts
        assert config.weight_dtype.lower() == expected_dtype
        assert config.head_dim == expected_head_dim
        assert config.quantization == expected_quant

    def test_suggested_quantizations(self):
        c = LLMConfig(
            num_hidden_layers=2,
            hidden_size=64,
            vocab_size=1000,
            num_attention_heads=4,
            head_dim=16,
            weight_dtype="bfloat16",
            max_seq_len=2048,
        )
        suggestions = c.suggested_quantizations
        assert "4bit" in suggestions

    @pytest.mark.parametrize(
        "config_file, expected_quantizations",
        [
            ("Devstral-Small-2507-GQA.json", {"4bit"}),
            ("Kimi-K2-Instruct-MOE.json", {"4bit"}),
            ("Qwen3-235B-A22B-Instruct-2507-FP8.json", {"4bit"}),
        ],
    )
    def test_suggested_quantizations_from_file(
        self, config_file, expected_quantizations
    ):
        raw = load_config(config_file)
        config = LLMConfig.from_raw_config(raw)
        suggestions = set(config.suggested_quantizations)
        assert expected_quantizations.issubset(suggestions)


# --- Tests for recommend.py ---
class GPUShapesIndexMock:
    def __init__(self):
        # local_path = os.path.join(os.path.dirname(__file__), "../../resources", "gpu_shapes_index.json")
        local_path = "ads/aqua/resources/gpu_shapes_index.json"
        with open(local_path) as f:
            local_data = json.load(f)

        local_shapes = local_data.get("shapes", {})
        self.shapes = local_shapes


class MockDataScienceModel:
    @staticmethod
    def create(config_file=""):
        mock_model = MagicMock()
        mock_model.model_file_description = {"test_key": "test_value"}
        mock_model.display_name = re.sub(r"\.json$", "", config_file)
        mock_model.description = "test_description"
        mock_model.freeform_tags = {
            "OCI_AQUA": "ACTIVE",
            "license": "test_license",
            "organization": "test_organization",
            "task": "text-generation",
            "model_format": "SAFETENSORS",
            "ready_to_fine_tune": "true",
            "aqua_custom_base_model": "true",
        }
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            **{"key": "test_metadata_item_key", "value": "test_metadata_item_value"}
        )
        mock_model.custom_metadata_list = custom_metadata_list
        mock_model.provenance_metadata = ModelProvenanceMetadata(
            training_id="test_training_id"
        )
        return mock_model


class TestAquaShapeRecommend:
    @pytest.mark.parametrize(
        "config, expected_recs, expected_troubleshoot",
        [
            (  # decoder-only model
                {
                    "num_hidden_layers": 2,
                    "hidden_size": 64,
                    "vocab_size": 1000,
                    "num_attention_heads": 4,
                    "head_dim": 16,
                    "max_position_embeddings": 2048,
                },
                [],
                "",
            ),
            (  # encoder-decoder model
                {
                    "num_hidden_layers": 2,
                    "hidden_size": 64,
                    "vocab_size": 1000,
                    "num_attention_heads": 4,
                    "head_dim": 16,
                    "max_position_embeddings": 2048,
                    "is_encoder_decoder": True,
                },
                [],
                "Please provide a decoder-only text-generation model (ex. Llama, Falcon, etc). Encoder-decoder models (ex. T5, Gemma) and encoder-only (BERT) are not supported at this time.",
            ),
        ],
    )
    def test_which_shapes_valid(
        self, monkeypatch, config, expected_recs, expected_troubleshoot
    ):
        app = AquaShapeRecommend()
        mock_model = MockDataScienceModel.create()

        monkeypatch.setattr(
            "ads.aqua.app.DataScienceModel.from_id", lambda _: mock_model
        )

        expected_result = ShapeRecommendationReport(
            recommendations=expected_recs, troubleshoot=expected_troubleshoot
        )
        app._get_model_config = MagicMock(return_value=config)
        app.valid_compute_shapes = MagicMock(return_value=[])
        app._summarize_shapes_for_seq_lens = MagicMock(return_value=expected_result)

        request = RequestRecommend(
            model_id="ocid1.datasciencemodel.oc1.TEST", generate_table=False
        )
        result = app.which_shapes(request)
        assert result == expected_result

        # If troubleshoot is populated (error case), _summarize_shapes_for_seq_lens should not have been called
        if expected_troubleshoot:
            app._summarize_shapes_for_seq_lens.assert_not_called()
        else:
            # For non-error case, summarize should have been called
            llm_config = LLMConfig.from_raw_config(config)
            app._summarize_shapes_for_seq_lens.assert_called_once_with(
                llm_config, [], ""
            )

    @pytest.mark.parametrize(
        "config_file, result_file",
        [
            ("Devstral-Small-2507-GQA.json", "result-Devstral-Small-2507-GQA.json"),
            ("Kimi-K2-Instruct-MOE.json", "result-Kimi-K2-Instruct-MOE.json"),
            (
                "Qwen3-235B-A22B-Instruct-2507-FP8.json",
                "result-Qwen3-235B-A22B-Instruct-2507-FP8.json",
            ),
        ],
    )
    def test_which_shapes_valid_from_file(
        self, monkeypatch, config_file, result_file, **kwargs
    ):
        raw = load_config(config_file)
        app = AquaShapeRecommend()
        mock_model = MockDataScienceModel.create(config_file)
        monkeypatch.setattr(
            "ads.aqua.app.DataScienceModel.from_id", lambda _: mock_model
        )
        monkeypatch.setattr(app, "_get_model_config", lambda _: raw)

        shapes_index = GPUShapesIndexMock()
        real_shapes = [
            ComputeShapeSummary(name=name, shape_series="GPU", gpu_specs=spec)
            for name, spec in shapes_index.shapes.items()
        ]
        monkeypatch.setattr(
            app, "valid_compute_shapes", lambda *args, **kwargs: real_shapes
        )

        request = RequestRecommend(
            model_id="ocid1.datasciencemodel.oc1.TEST", generate_table=False
        )
        result = app.which_shapes(request=request)

        expected_result = load_config(result_file)
        print(result.model_dump_json())
        assert result.model_dump() == expected_result


# --- Tests for shape_report.py ---
class TestShapeReport:
    def test_shape_report_pareto_front(self):
        # worse recommendation- higher cost and lower performance -> should be filtered out
        mock_shape_a = ComputeShapeSummary(
            name="VM.GPU2.1",
            shape_series="GPU",
            gpu_specs={"ranking": {"cost": 15, "performance": 10}},
        )

        mock_shape_b = ComputeShapeSummary(
            name="VM.GPU.A10.1",
            shape_series="GPU",
            gpu_specs={"ranking": {"cost": 10, "performance": 12}},
        )

        a = ShapeReport(
            shape_details=mock_shape_a,
            configurations=[
                ModelConfig(
                    model_details=ModelDetail(
                        model_size_gb=1, kv_cache_size_gb=1, total_model_gb=2
                    ),
                    deployment_params=DeploymentParams(
                        quantization="8bit", max_model_len=2048, params=""
                    ),
                    recommendation="ok",
                )
            ],
        )
        b = ShapeReport(
            shape_details=mock_shape_b,
            configurations=[
                ModelConfig(
                    model_details=ModelDetail(
                        model_size_gb=1, kv_cache_size_gb=1, total_model_gb=2
                    ),
                    deployment_params=DeploymentParams(
                        quantization="8bit", max_model_len=2048, params=""
                    ),
                    recommendation="ok",
                )
            ],
        )
        c = ShapeReport(
            shape_details=mock_shape_b,
            configurations=[
                ModelConfig(
                    model_details=ModelDetail(
                        model_size_gb=1, kv_cache_size_gb=1, total_model_gb=2
                    ),
                    deployment_params=DeploymentParams(
                        quantization="bfloat16", max_model_len=2048, params=""
                    ),
                    recommendation="ok",
                )
            ],
        )
        d = ShapeReport(
            shape_details=mock_shape_b,
            configurations=[
                ModelConfig(
                    model_details=ModelDetail(
                        model_size_gb=1, kv_cache_size_gb=1, total_model_gb=2
                    ),
                    deployment_params=DeploymentParams(
                        quantization="8bit", max_model_len=4096, params=""
                    ),
                    recommendation="ok",
                )
            ],
        )
        pf = ShapeReport.pareto_front([a, b, c, d])
        assert c and d in pf
        assert a and b not in pf
        assert len(pf) == 2
