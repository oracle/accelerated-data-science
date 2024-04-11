#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest
from importlib import metadata, reload
from unittest.mock import MagicMock, patch

from notebook.base.handlers import IPythonHandler
from parameterized import parameterized

import ads.aqua
import ads.aqua.exception
import ads.aqua.extension
import ads.aqua.extension.common_handler
import ads.config
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.extension.common_handler import (
    ADSVersionHandler,
    CompatibilityCheckHandler,
)
from ads.aqua.extension.evaluation_handler import (
    AquaEvaluationConfigHandler,
    AquaEvaluationMetricsHandler,
    AquaEvaluationReportHandler,
    AquaEvaluationStatusHandler,
)
from ads.aqua.extension.model_handler import AquaModelHandler, AquaModelLicenseHandler
from ads.aqua.model import AquaModelApp
from tests.unitary.with_extras.aqua.utils import HandlerTestDataset as TestDataset


class TestHandlers(unittest.TestCase):
    """Contains test cases for the following handlers which have GET:

    AquaEvaluationConfigHandler,
    AquaEvaluationMetricsHandler,
    AquaEvaluationReportHandler,
    AquaEvaluationStatusHandler,
    ADSVersionHandler,
    CompatibilityCheckHandler,
    AquaModelLicenseHandler,
    AquaModelHandler,
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ["ODSC_MODEL_COMPARTMENT_OCID"] = "mytemp"

        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.extension.common_handler)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.environ.pop("ODSC_MODEL_COMPARTMENT_OCID", None)

        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.extension.common_handler)

    @parameterized.expand(
        [
            (
                AquaEvaluationConfigHandler,
                AquaEvaluationApp,
                "load_evaluation_config",
                TestDataset.MOCK_OCID,
                TestDataset.MOCK_OCID,
            ),
            (
                AquaEvaluationMetricsHandler,
                AquaEvaluationApp,
                "load_metrics",
                TestDataset().mock_url("metrics"),
                TestDataset.MOCK_OCID,
            ),
            (
                AquaEvaluationReportHandler,
                AquaEvaluationApp,
                "download_report",
                TestDataset().mock_url("report"),
                TestDataset.MOCK_OCID,
            ),
            (
                AquaEvaluationStatusHandler,
                AquaEvaluationApp,
                "get_status",
                TestDataset().mock_url("status"),
                TestDataset.MOCK_OCID,
            ),
            (
                ADSVersionHandler,
                None,
                None,
                None,
                {"data": metadata.version("oracle_ads")},
            ),
            (
                CompatibilityCheckHandler,
                None,
                None,
                None,
                dict(status="ok"),
            ),
            (
                AquaModelLicenseHandler,
                AquaModelApp,
                "load_license",
                TestDataset().mock_url("license"),
                TestDataset.MOCK_OCID,
            ),
            (
                AquaModelHandler,
                AquaModelApp,
                "get",
                TestDataset.MOCK_OCID,
                TestDataset.MOCK_OCID,
            ),
            (AquaModelHandler, AquaModelApp, "list", "", (None, None)),
        ]
    )
    @patch.object(IPythonHandler, "__init__")
    def test_get(
        self,
        target_handler,
        target_app,
        associated_api,
        url,
        expected_call,
        ipython_init_mock,
    ):
        """Tests invoking GET method successfully."""

        ipython_init_mock.return_value = None
        test_instance = target_handler(MagicMock(), MagicMock())
        test_instance.finish = MagicMock()
        test_instance.request = MagicMock()

        if associated_api:
            with patch.object(target_app, associated_api) as mock_api:
                test_instance.get(url)

                if associated_api:
                    test_instance.finish.assert_called_with(mock_api.return_value)

                    (
                        mock_api.assert_called_with(expected_call)
                        if isinstance(expected_call, str)
                        else mock_api.assert_called
                    )
        else:
            test_instance.get()
            test_instance.finish.assert_called_with(expected_call)
