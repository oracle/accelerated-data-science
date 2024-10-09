#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
from unittest.mock import patch

from ads.aqua.common.entities import ContainerSpec
from ads.aqua.config.config import get_evaluation_service_config


class TestConfig:
    """Unit tests for AQUA common configurations."""

    def setup_class(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))
        cls.artifact_dir = os.path.join(cls.curr_dir, "test_data", "config")

    @patch("ads.aqua.config.config.get_container_config")
    def test_evaluation_service_config(self, mock_get_container_config):
        """Ensures that the common evaluation configuration can be successfully retrieved."""

        with open(
            os.path.join(
                self.artifact_dir, "evaluation_config_with_default_params.json"
            )
        ) as file:
            expected_result = {
                ContainerSpec.CONTAINER_SPEC: {"test_container": json.load(file)}
            }

        mock_get_container_config.return_value = expected_result

        test_result = get_evaluation_service_config(container="test_container")
        assert (
            test_result.to_dict()
            == expected_result[ContainerSpec.CONTAINER_SPEC]["test_container"]
        )
