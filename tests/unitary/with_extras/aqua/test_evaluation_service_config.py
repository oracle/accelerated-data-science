#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
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
