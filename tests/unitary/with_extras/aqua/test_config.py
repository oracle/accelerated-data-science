#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from unittest.mock import MagicMock, patch

from ads.aqua.common.utils import service_config_path
from ads.aqua.config.config import evaluation_config
from ads.aqua.config.evaluation.evaluation_service_config import EvaluationServiceConfig
from ads.aqua.constants import EVALUATION_SERVICE_CONFIG


class TestConfig:
    """Unit tests for AQUA common configurations."""

    @patch.object(EvaluationServiceConfig, "from_json")
    def test_evaluation_service_config(self, mock_from_json):
        """Ensures that the common evaluation configuration can be successfully retrieved."""

        expected_result = MagicMock()
        mock_from_json.return_value = expected_result

        test_result = evaluation_config()

        mock_from_json.assert_called_with(
            uri=f"{service_config_path()}/{EVALUATION_SERVICE_CONFIG}"
        )
        assert test_result == expected_result
