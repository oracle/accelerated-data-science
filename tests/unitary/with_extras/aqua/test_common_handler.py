#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest
from unittest.mock import MagicMock, patch
from importlib import reload
from notebook.base.handlers import IPythonHandler

import ads.config
import ads.aqua
from ads.aqua.utils import AQUA_GA_LIST
from ads.aqua.extension.common_handler import CompatibilityCheckHandler


class TestDataset:
    SERVICE_COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"


class TestEvaluationHandler(unittest.TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.common_handler = CompatibilityCheckHandler(MagicMock(), MagicMock())
        self.common_handler.request = MagicMock()

    def test_get_ok(self):
        """Test to check if ok is returned when ODSC_MODEL_COMPARTMENT_OCID is set."""
        with patch.dict(
            os.environ,
            {"ODSC_MODEL_COMPARTMENT_OCID": TestDataset.SERVICE_COMPARTMENT_ID},
        ):
            reload(ads.config)
            reload(ads.aqua)
            reload(ads.aqua.extension.common_handler)

            with patch(
                "ads.aqua.extension.base_handler.AquaAPIhandler.finish"
            ) as mock_finish:
                mock_finish.side_effect = lambda x: x
                self.common_handler.request.path = "aqua/hello"
                result = self.common_handler.get()
                assert result["status"] == "ok"

    def test_get_compatible_status(self):
        """Test to check if compatible is returned when ODSC_MODEL_COMPARTMENT_OCID is not set
        but CONDA_BUCKET_NS is one of the namespaces from the GA list."""
        with patch.dict(
            os.environ,
            {"ODSC_MODEL_COMPARTMENT_OCID": "", "CONDA_BUCKET_NS": AQUA_GA_LIST[0]},
        ):
            reload(ads.config)
            reload(ads.aqua)
            reload(ads.aqua.extension.common_handler)
            with patch(
                "ads.aqua.extension.base_handler.AquaAPIhandler.finish"
            ) as mock_finish:
                with patch(
                    "ads.aqua.extension.common_handler.fetch_service_compartment"
                ) as mock_fetch_service_compartment:
                    mock_fetch_service_compartment.return_value = None
                    mock_finish.side_effect = lambda x: x
                    self.common_handler.request.path = "aqua/hello"
                    result = self.common_handler.get()
                    assert result["status"] == "compatible"

    def test_raise_not_compatible_error(self):
        """Test to check if error is returned when ODSC_MODEL_COMPARTMENT_OCID is not set
        and CONDA_BUCKET_NS is not one of the namespaces from the GA list."""
        with patch.dict(
            os.environ,
            {"ODSC_MODEL_COMPARTMENT_OCID": "", "CONDA_BUCKET_NS": "test-namespace"},
        ):
            reload(ads.config)
            reload(ads.aqua)
            reload(ads.aqua.extension.common_handler)
            with patch(
                "ads.aqua.extension.base_handler.AquaAPIhandler.finish"
            ) as mock_finish:
                with patch(
                    "ads.aqua.extension.common_handler.fetch_service_compartment"
                ) as mock_fetch_service_compartment:
                    mock_fetch_service_compartment.return_value = None
                    mock_finish.side_effect = lambda x: x
                    self.common_handler.write_error = MagicMock()
                    self.common_handler.request.path = "aqua/hello"
                    self.common_handler.get()

                    assert self.common_handler.write_error.call_args[1].get(
                        "reason"
                    ) == (
                        "The AI Quick actions extension is not "
                        "compatible in the given region."
                    ), "Incorrect error message."
                    assert (
                        self.common_handler.write_error.call_args[1].get("status_code")
                        == 404
                    ), "Incorrect status code."
