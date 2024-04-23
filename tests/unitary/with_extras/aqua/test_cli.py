#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import logging
import subprocess
from unittest import TestCase
from unittest.mock import patch
from importlib import reload
from parameterized import parameterized

import ads.aqua
import ads.config
from ads.aqua.cli import AquaCommand


class TestAquaCLI(TestCase):
    """Tests the AQUA CLI."""

    DEFAULT_AQUA_CLI_LOGGING_LEVEL = "ERROR"
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(module)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    SERVICE_COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"

    def test_entrypoint(self):
        """Tests CLI entrypoint."""
        result = subprocess.run(["ads", "aqua", "--help"], capture_output=True)
        self.logger.info(f"{self._testMethodName}\n" + result.stderr.decode("utf-8"))
        assert result.returncode == 0

    @parameterized.expand(
        [
            ("default", None, DEFAULT_AQUA_CLI_LOGGING_LEVEL),
            ("set logging level", "info", "info"),
        ]
    )
    def test_aquacommand(self, name, arg, expected):
        """Tests aqua command initialization."""
        with patch.dict(
            os.environ,
            {"ODSC_MODEL_COMPARTMENT_OCID": TestAquaCLI.SERVICE_COMPARTMENT_ID},
        ):
            reload(ads.config)
            reload(ads.aqua)
            reload(ads.aqua.cli)
            with patch("ads.aqua.cli.set_log_level") as mock_setting_log:
                if arg:
                    AquaCommand(arg)
                else:
                    AquaCommand()
                mock_setting_log.assert_called_with(expected)

    @parameterized.expand(
        [
            ("default", None),
            ("using jupyter instance", "nb-session-ocid"),
        ]
    )
    def test_aqua_command_without_compartment_env_var(self, name, session_ocid):
        """Test whether exit is called when ODSC_MODEL_COMPARTMENT_OCID is not set. Also check if NB_SESSION_OCID is
        set then log the appropriate message."""

        with patch("sys.exit") as mock_exit:
            env_dict = {"ODSC_MODEL_COMPARTMENT_OCID": ""}
            if session_ocid:
                env_dict.update({"NB_SESSION_OCID": session_ocid})
            with patch.dict(os.environ, env_dict):
                reload(ads.config)
                reload(ads.aqua)
                reload(ads.aqua.cli)
                with patch("ads.aqua.cli.set_log_level") as mock_setting_log:
                    with patch("ads.aqua.logger.error") as mock_logger_error:
                        AquaCommand()
                        mock_setting_log.assert_called_with(
                            TestAquaCLI.DEFAULT_AQUA_CLI_LOGGING_LEVEL
                        )
                        mock_logger_error.assert_any_call(
                            "ODSC_MODEL_COMPARTMENT_OCID environment variable is not set for Aqua."
                        )
                        if session_ocid:
                            mock_logger_error.assert_any_call(
                                f"Aqua is not available for the notebook session {session_ocid}."
                            )
                        mock_exit.assert_called_with(1)
