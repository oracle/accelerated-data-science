#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from ads.opctl.decorator import common
from ads.opctl.decorator.common import validate_environment, OpctlEnvironmentError
from unittest.mock import patch


class TestOpctlDecorators:
    """Tests the all OPCTL common decorators."""

    @patch("ads.opctl.decorator.common.NB_SESSION_OCID", None)
    @patch("ads.opctl.decorator.common.JOB_RUN_OCID", None)
    @patch("ads.opctl.decorator.common.PIPELINE_RUN_OCID", None)
    @patch("ads.opctl.decorator.common.DATAFLOW_RUN_OCID", None)
    @patch("ads.opctl.decorator.common.MD_OCID", None)
    def test_validate_environment_success(self):
        """Tests validating environment decorator."""

        @validate_environment
        def mock_function():
            return "SUCCESS"

        assert mock_function() == "SUCCESS"

    @patch("ads.opctl.decorator.common.NB_SESSION_OCID", "TEST")
    def test_validate_environment_fail(self):
        """Tests validating environment decorator fails."""

        @validate_environment
        def mock_function():
            return "SUCCESS"

        with pytest.raises(OpctlEnvironmentError):
            assert mock_function()
