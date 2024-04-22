#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import subprocess
from unittest import TestCase
from unittest.mock import patch

from parameterized import parameterized

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
