#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest
from importlib import reload
from unittest.mock import MagicMock, patch

from notebook.base.handlers import IPythonHandler

import ads.aqua
import ads.config

from ads.aqua.extension.common_handler import (
    CompatibilityCheckHandler,
    AquaVersionHandler,
)


class TestDataset:
    SERVICE_COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"


class TestCompatibilityCheckHandler(unittest.TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.common_handler = CompatibilityCheckHandler(MagicMock(), MagicMock())
        self.common_handler.request = MagicMock()

    def test_get_ok(self):
        """Test to check if ok is returned."""
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.extension.utils)
        reload(ads.aqua.extension.common_handler)

        with patch(
            "ads.aqua.extension.base_handler.AquaAPIhandler.finish"
        ) as mock_finish:
            mock_finish.side_effect = lambda x: x
            self.common_handler.request.path = "aqua/hello"
            result = self.common_handler.get()
            assert result["status"] == "ok"


class TestAquaVersionHandler(unittest.TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.version_handler = AquaVersionHandler(MagicMock(), MagicMock())
        self.version_handler.request = MagicMock()

    @patch("ads.common.utils.read_file")
    def test_get(self, mock_read_file):
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.extension.utils)
        reload(ads.aqua.extension.common_handler)
        mock_read_file.return_value = '{"installed":{"aqua":"1.0.4","ads":"2.13.12"}}'
        with patch(
            "ads.aqua.extension.base_handler.AquaAPIhandler.finish"
        ) as mock_finish:
            mock_finish.side_effect = lambda x: x
            self.version_handler.request.path = "aqua/aqua_version"
            result = self.version_handler.get()
            assert result == {"installed": {"aqua": "1.0.4", "ads": "2.13.12"}}
