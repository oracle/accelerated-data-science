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
from ads.aqua.constants import AQUA_GA_LIST
from ads.aqua.extension.common_handler import CompatibilityCheckHandler
from ads.aqua.extension.utils import ui_compatability_check


class TestDataset:
    SERVICE_COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"


class TestEvaluationHandler(unittest.TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.common_handler = CompatibilityCheckHandler(MagicMock(), MagicMock())
        self.common_handler.request = MagicMock()

    def tearDown(self) -> None:
        ui_compatability_check.cache_clear()

    def test_get_ok(self):
        """Test to check if ok is returned when ODSC_MODEL_COMPARTMENT_OCID is set."""
        with patch.dict(
            os.environ,
            {"ODSC_MODEL_COMPARTMENT_OCID": TestDataset.SERVICE_COMPARTMENT_ID},
        ):
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
