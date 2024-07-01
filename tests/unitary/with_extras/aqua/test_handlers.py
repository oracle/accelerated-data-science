#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import unittest
from dataclasses import asdict
from importlib import metadata, reload
from unittest.mock import MagicMock, patch

from notebook.base.handlers import APIHandler, IPythonHandler
from oci.exceptions import ServiceError
from parameterized import parameterized
from tornado.httpserver import HTTPRequest
from tornado.httputil import HTTPServerRequest
from tornado.web import Application, HTTPError

import ads.aqua
import ads.aqua.common.errors
import ads.aqua.extension
import ads.aqua.extension.common_handler
import ads.config
from ads.aqua.common.errors import AquaError
from ads.aqua.data import AquaResourceIdentifier
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.extension.base_handler import AquaAPIhandler
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


class TestBaseHandlers(unittest.TestCase):
    """Contains test cases for base handler."""

    @patch.object(APIHandler, "__init__")
    def setUp(self, mock_init) -> None:
        mock_init.return_value = None
        application = Application()
        request = HTTPServerRequest(method="GET", uri="/test", connection=MagicMock())
        self.test_instance = AquaAPIhandler(application, request)
        self.test_instance.telemetry.record_event_async = MagicMock()

    @parameterized.expand(
        [
            ("with None", None, None),
            ("with list", [1, 2, 3], {"data": [1, 2, 3]}),
            (
                "with DataClassSerializable",
                AquaResourceIdentifier(id="123", name="myname"),
                {"id": "123", "name": "myname", "url": ""},
            ),
            (
                "with dataclass",
                TestDataset.mock_dataclass_obj,
                asdict(TestDataset.mock_dataclass_obj),
            ),
        ]
    )
    @patch.object(APIHandler, "finish")
    def test_finish(self, name, payload, expected_call, mock_super_finish):
        """Tests AquaAPIhandler.finish"""
        mock_super_finish.return_value = None

        self.test_instance.finish(payload)
        if expected_call:
            mock_super_finish.assert_called_with(expected_call)
        else:
            mock_super_finish.assert_called_with()

    @parameterized.expand(
        [
            [
                "HTTPError",
                dict(
                    status_code=400,
                    exc_info=(None, HTTPError(400, "Bad Request"), None),
                ),
                "Bad Request",
            ],
            [
                "ADS Error",
                dict(
                    status_code=500,
                    reason="Testing ADS Internal Error.",
                    exc_info=(None, ValueError("Invalid parameter."), None),
                ),
                "Internal Server Error",
            ],
            [
                "AQUA Error",
                dict(
                    status_code=404,
                    reason="Testing AquaError happen during create operation.",
                    service_payload=TestDataset.mock_service_payload_create,
                    exc_info=(
                        None,
                        AquaError(
                            status=404,
                            reason="Testing AquaError happen during create operation.",
                            service_payload=TestDataset.mock_service_payload_create,
                        ),
                        None,
                    ),
                ),
                "Authorization Failed: Could not create resource. Operation Name: create_resources.",
            ],
            [
                "oci ServiceError",
                dict(
                    status_code=404,
                    reason="Testing ServiceError happen when get_job_run.",
                    service_payload=TestDataset.mock_service_payload_get,
                    exc_info=(
                        None,
                        ServiceError(
                            headers={},
                            **TestDataset.mock_service_payload_get,
                        ),
                        None,
                    ),
                ),
                "Authorization Failed: The resource you're looking for isn't accessible. Operation Name: get_job_run.",
            ],
        ]
    )
    @patch("ads.aqua.extension.base_handler.logger")
    @patch("uuid.uuid4")
    def test_write_error(self, name, input, expected_msg, mock_uuid, mock_logger):
        """Tests AquaAPIhandler.write_error"""
        mock_uuid.return_value = "1234"
        self.test_instance.set_header = MagicMock()
        self.test_instance.set_status = MagicMock()
        self.test_instance.finish = MagicMock()

        self.test_instance.write_error(**input)

        self.test_instance.set_header.assert_called_once_with(
            "Content-Type", "application/json"
        )
        self.test_instance.set_status.assert_called_once_with(
            input.get("status_code"), reason=input.get("reason")
        )
        expected_reply = {
            "status": input.get("status_code"),
            "message": expected_msg,
            "service_payload": input.get("service_payload", {}),
            "reason": input.get("reason"),
            "request_id": "1234",
        }
        self.test_instance.finish.assert_called_once_with(json.dumps(expected_reply))
        self.test_instance.telemetry.record_event_async.assert_called_with(
            category="aqua/error",
            action=str(
                input.get("status_code"),
            ),
            value=input.get("reason"),
        )

        mock_logger.warning.assert_called_with(expected_msg)


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
                "AquaEvaluationConfigHandler",
                AquaEvaluationConfigHandler,
                AquaEvaluationApp,
                "load_evaluation_config",
                TestDataset.MOCK_OCID,
                TestDataset.MOCK_OCID,
            ),
            (
                "AquaEvaluationMetricsHandler",
                AquaEvaluationMetricsHandler,
                AquaEvaluationApp,
                "load_metrics",
                TestDataset().mock_url("metrics"),
                TestDataset.MOCK_OCID,
            ),
            (
                "AquaEvaluationReportHandler",
                AquaEvaluationReportHandler,
                AquaEvaluationApp,
                "download_report",
                TestDataset().mock_url("report"),
                TestDataset.MOCK_OCID,
            ),
            (
                "AquaEvaluationStatusHandler",
                AquaEvaluationStatusHandler,
                AquaEvaluationApp,
                "get_status",
                TestDataset().mock_url("status"),
                TestDataset.MOCK_OCID,
            ),
            (
                "ADSVersionHandler",
                ADSVersionHandler,
                None,
                None,
                None,
                {"data": metadata.version("oracle_ads")},
            ),
            (
                "CompatibilityCheckHandler",
                CompatibilityCheckHandler,
                None,
                None,
                None,
                dict(status="ok"),
            ),
            (
                "AquaModelLicenseHandler",
                AquaModelLicenseHandler,
                AquaModelApp,
                "load_license",
                TestDataset().mock_url("license"),
                TestDataset.MOCK_OCID,
            ),
            (
                "AquaModelHandler",
                AquaModelHandler,
                AquaModelApp,
                "get",
                TestDataset.MOCK_OCID,
                TestDataset.MOCK_OCID,
            ),
            (
                "AquaModelHandler_list",
                AquaModelHandler,
                AquaModelApp,
                "list",
                "",
                (None, None),
            ),
        ]
    )
    @patch.object(IPythonHandler, "__init__")
    def test_get(
        self,
        name,
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
