#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import logging
import subprocess
import pytest
from unittest import TestCase
from unittest.mock import patch
from importlib import reload
from parameterized import parameterized

import ads.aqua
import ads.config
from ads.aqua.cli import AquaCommand


class TestAquaCLI(TestCase):
    """Tests the AQUA CLI."""

    DEFAUL_AQUA_CLI_LOGGING_LEVEL = "ERROR"
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(module)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    SERVICE_COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"

    def setUp(self):
        os.environ["ODSC_MODEL_COMPARTMENT_OCID"] = TestAquaCLI.SERVICE_COMPARTMENT_ID
        reload(ads.aqua)
        reload(ads.aqua.cli)

    def test_entrypoint(self):
        """Tests CLI entrypoint."""
        result = subprocess.run(["ads", "aqua", "--help"], capture_output=True)
        self.logger.info(f"{self._testMethodName}\n" + result.stderr.decode("utf-8"))
        assert result.returncode == 0

    @parameterized.expand(
        [
            ("default", None, DEFAUL_AQUA_CLI_LOGGING_LEVEL),
            ("set logging level", "info", "info"),
        ]
    )
    @patch("ads.aqua.cli.set_log_level")
    def test_aquacommand(self, name, arg, expected, mock_setting_log):
        """Tests aqua command initailzation."""
        if arg:
            AquaCommand(arg)
        else:
            AquaCommand()
        mock_setting_log.assert_called_with(expected)

    def test_aqua_command_without_compartment_env_var(self):
        os.environ.pop("ODSC_MODEL_COMPARTMENT_OCID", None)
        reload(ads.aqua)
        reload(ads.aqua.cli)
        with pytest.raises(SystemExit):
            AquaCommand()
