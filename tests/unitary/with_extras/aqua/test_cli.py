#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import os
import subprocess
import uuid
from importlib import reload
from unittest import TestCase
from unittest.mock import call, patch

from parameterized import parameterized

import ads.aqua
import ads.config
from ads.aqua.cli import AquaCommand
from ads.aqua.common.errors import AquaCLIError, AquaConfigError


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
            ("set logging level", dict(log_level="info"), "INFO"),
            ("debug", dict(debug=True), "DEBUG"),
            ("verbose", dict(verbose=True), "INFO"),
            ("flag_priority", dict(debug=True, log_level="info"), "DEBUG"),
        ]
    )
    @patch.dict(
        os.environ, {"ODSC_MODEL_COMPARTMENT_OCID": SERVICE_COMPARTMENT_ID}, clear=True
    )
    def test_aquacommand(self, name, arg, expected):
        """Tests aqua command initialization."""

        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.cli)
        with patch("ads.aqua.cli.set_log_level") as mock_setting_log:
            if arg:
                AquaCommand(**arg)
            else:
                AquaCommand()
            mock_setting_log.assert_called_with(expected)

    @parameterized.expand(
        [
            ("conflict", dict(debug=True, verbose=True)),
            ("invalid_value", dict(debug="abc")),
            ("invalid_value", dict(verbose="abc")),
        ]
    )
    @patch.dict(
        os.environ, {"ODSC_MODEL_COMPARTMENT_OCID": SERVICE_COMPARTMENT_ID}, clear=True
    )
    def test_aquacommand_flag(self, name, arg):
        """Tests aqua command initialization with wrong flag."""

        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.cli)
        with self.assertRaises(AquaCLIError):
            AquaCommand(**arg)

    @patch("sys.argv", ["ads", "aqua", "--some-option"])
    @patch("ads.cli.serialize")
    @patch("fire.Fire")
    @patch("ads.aqua.cli.AquaCommand")
    @patch("ads.aqua.logger")
    def test_aqua_cli(self, mock_logger, mock_aqua_command, mock_fire, mock_serialize):
        """Tests when Aqua Cli being invoked."""
        from ads.cli import cli

        cli()
        mock_fire.assert_called_once()
        mock_fire.assert_called_with(
            mock_aqua_command,
            command=["--some-option"],
            name="ads aqua",
            serialize=mock_serialize,
        )

    @parameterized.expand(
        [
            (
                "with_defined_exit_code",
                AquaConfigError("test error"),
                AquaConfigError.exit_code,
                "test error",
            ),
            (
                "without_defined_exit_code",
                ValueError("general error"),
                1,
                "general error",
            ),
        ]
    )
    @patch("sys.argv", ["ads", "aqua", "--error-option"])
    @patch("uuid.uuid4")
    @patch("fire.Fire")
    @patch("ads.aqua.cli.AquaCommand")
    @patch("ads.aqua.logger.error")
    @patch("sys.exit")
    def test_aqua_cli_with_error(
        self,
        name,
        mock_side_effect,
        expected_code,
        expected_logging_message,
        mock_exit,
        mock_logger_error,
        mock_aqua_command,
        mock_fire,
        mock_uuid,
    ):
        """Tests when Aqua Cli gracefully exit when error raised."""
        mock_fire.side_effect = mock_side_effect
        from ads.cli import cli

        uuid_value = "12345678-1234-5678-1234-567812345678"
        mock_uuid.return_value = uuid.UUID(uuid_value)
        expected_logging_message = type(expected_logging_message)(
            f"Error Request ID: {uuid_value}\nError: {str(expected_logging_message)}"
        )
        cli()
        calls = [
            call(expected_logging_message),
            call(f"Exit code: {expected_code}"),
        ]
        mock_logger_error.assert_has_calls(calls)
        mock_exit.assert_called_with(expected_code)
