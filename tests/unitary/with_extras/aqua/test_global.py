#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
from unittest.mock import MagicMock, patch

from ads.aqua import configure_aqua_logger, get_logger_level, set_log_level


class TestAquaLogging(unittest.TestCase):
    DEFAULT_AQUA_LOG_LEVEL = "INFO"

    @patch.dict("os.environ", {})
    def test_get_logger_level_default(self):
        """Test default log level when environment variable is not set."""
        self.assertEqual(get_logger_level(), self.DEFAULT_AQUA_LOG_LEVEL)

    @patch.dict("os.environ", {"ADS_AQUA_LOG_LEVEL": "DEBUG"})
    def test_get_logger_level_from_env(self):
        """Test log level is correctly read from environment variable."""
        self.assertEqual(get_logger_level(), "DEBUG")

    @patch("logging.getLogger")
    @patch("logging.StreamHandler")
    def test_configure_aqua_logger(self, mock_handler, mock_get_logger):
        """Test that logger is correctly configured."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        logger = configure_aqua_logger()

        mock_get_logger.assert_called_once_with("ads.aqua")
        mock_logger.setLevel.assert_called_with(self.DEFAULT_AQUA_LOG_LEVEL)

    @patch("ads.aqua.logger", create=True)
    def test_set_log_level(self, mock_logger):
        """Test that the log level of the logger is set correctly."""
        mock_handler = MagicMock()
        mock_logger.handlers = [mock_handler]

        set_log_level("warning")

        mock_logger.setLevel.assert_called_with("WARNING")
