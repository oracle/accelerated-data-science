#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import json
import os

import pytest

from ads.aqua.config.evaluation.evaluation_service_config import EvaluationServiceConfig


class TestEvaluationServiceConfig:
    """Unit tests for EvaluationServiceConfig class."""

    def setup_class(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))
        cls.artifact_dir = os.path.join(cls.curr_dir, "test_data", "config")

    def teardown_class(cls): ...

    def setup_method(self):
        self.mock_config: EvaluationServiceConfig = EvaluationServiceConfig.from_json(
            uri=os.path.join(self.artifact_dir, "evaluation_config.json")
        )

    def test_init(self):
        """Ensures the config can be instantiated with the default params"""
        test_config = EvaluationServiceConfig()

        with open(
            os.path.join(
                self.artifact_dir, "evaluation_config_with_default_params.json"
            )
        ) as file:
            expected_config = json.load(file)

        assert test_config.to_dict() == expected_config

    def test_read_config(self):
        """Ensures the config can be read from the JSON file."""

        with open(os.path.join(self.artifact_dir, "evaluation_config.json")) as file:
            expected_config = json.load(file)

        assert self.mock_config.to_dict() == expected_config

    @pytest.mark.parametrize(
        "container_name, extra_params",
        [
            ("odsc-vllm-serving", {}),
            ("odsc-vllm-serving", {}),
            ("odsc-tgi-serving", {}),
            (
                "odsc-llama-cpp-serving",
                {"inference_max_threads": 1, "inference_delay": 1},
            ),
            ("none-exist", {}),
        ],
    )
    def test_get_merged_inference_params(self, container_name, extra_params):
        """Tests merging default inference params with those specific to the given framework."""

        test_result = self.mock_config.get_merged_inference_params(
            container_name=container_name
        )
        expected_result = {
            "inference_rps": 25,
            "inference_timeout": 120,
            "inference_max_threads": 10,
            "inference_retries": 3,
            "inference_backoff_factor": 3.0,
            "inference_delay": 0.0,
        }
        expected_result.update(extra_params)

        assert test_result.to_dict() == expected_result

    @pytest.mark.parametrize(
        "container_name, version, exclude, include",
        [
            ("odsc-vllm-serving", "0.5.3.post1", ["add_generation_prompt"], {}),
            (
                "odsc-vllm-serving",
                "0.5.1",
                ["max_tokens", "frequency_penalty"],
                {"some_other_param": "some_other_param_value"},
            ),
            ("odsc-vllm-serving", "none-exist", [], {}),
            ("odsc-tgi-serving", None, [], {}),
            ("odsc-tgi-serving", "none-exist", [], {}),
            (
                "odsc-tgi-serving",
                "2.0.1.4",
                ["max_tokens", "frequency_penalty"],
                {"some_other_param": "some_other_param_value"},
            ),
            ("odsc-llama-cpp-serving", "0.2.78.0", [], {}),
            ("none-exist", "none-exist", [], {}),
        ],
    )
    def test_get_merged_inference_model_params(
        self, container_name, version, exclude, include
    ):
        expected_result = {"some_default_param": "some_default_param"}
        expected_result.update(
            {
                "model": "odsc-llm",
                "add_generation_prompt": False,
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "stop": [],
            }
        )
        expected_result.update(include)
        for key in exclude:
            expected_result.pop(key, None)

        test_result = self.mock_config.get_merged_inference_model_params(
            container_name=container_name, version=version
        )

        assert test_result == expected_result

    @pytest.mark.parametrize(
        "evaluation_container, evaluation_target, shapes_found",
        [
            (None, None, 5),
            ("odsc-llm-evaluate", None, 5),
            ("odsc-llm-evaluate", "datasciencemodeldeployment", 4),
            ("odsc-llm-evaluate", "datasciencemodel", 1),
            ("none", None, 0),
            (None, "none", 0),
        ],
    )
    def test_search_shapes(self, evaluation_container, evaluation_target, shapes_found):
        """Ensures searching shapes that match the given filters."""
        test_result = self.mock_config.ui_config.search_shapes(
            evaluation_container=evaluation_container,
            evaluation_target=evaluation_target,
        )

        assert len(test_result) == shapes_found
