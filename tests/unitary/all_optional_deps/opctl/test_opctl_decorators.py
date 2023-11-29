#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from ads.opctl.decorator.common import validate_environment, OpctlEnvironmentError
from unittest.mock import patch, MagicMock


class TestOpctlDecorators:
    """Tests the all OPCTL common decorators."""

    def test_validate_environment_success(self):
        """Tests validating environment decorator."""

        @validate_environment
        def mock_function():
            return "SUCCESS"

        assert mock_function() == "SUCCESS"

    def test_validate_environment_fail(self):
        """Tests validating environment decorator fails."""

        @validate_environment
        def mock_function():
            return "SUCCESS"

        import docker

        with patch.object(
            docker,
            "from_env",
            return_value=MagicMock(
                "version",
                return_value=MagicMock(side_effect=ValueError("Something went wrong")),
            ),
        ):
            with pytest.raises(OpctlEnvironmentError):
                assert mock_function()
