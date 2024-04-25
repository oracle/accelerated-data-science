#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from unittest import TestCase
from unittest.mock import MagicMock, patch

from notebook.base.handlers import IPythonHandler
from oci.exceptions import (
    UPLOAD_MANAGER_DEBUG_INFORMATION_LOG,
    CompositeOperationError,
    ConfigFileNotFound,
    ConnectTimeout,
    MissingEndpointForNonRegionalServiceClientError,
    MultipartUploadError,
    RequestException,
    ServiceError,
)
from parameterized import parameterized
from tornado.web import HTTPError

from ads.aqua.exception import AquaError
from ads.aqua.extension.base_handler import AquaAPIhandler


class TestDataset:
    mock_request_id = "1234"


class TestAquaDecorators(TestCase):
    """Tests the all aqua common decorators."""

    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.test_instance = AquaAPIhandler(MagicMock(), MagicMock())
        self.test_instance.finish = MagicMock()
        self.test_instance.set_header = MagicMock()
        self.test_instance.set_status = MagicMock()

    @parameterized.expand(
        [
            [
                "oci ServiceError",
                ServiceError(
                    status=500,
                    code="InternalError",
                    message="An internal error occurred.",
                    headers={},
                ),
                {
                    "status": 500,
                    "message": "An internal error occurred.",
                    "service_payload": {
                        "target_service": None,
                        "status": 500,
                        "code": "InternalError",
                        "opc-request-id": None,
                        "message": "An internal error occurred.",
                        "operation_name": None,
                        "timestamp": None,
                        "client_version": None,
                        "request_endpoint": None,
                        "logging_tips": "To get more info on the failing request, refer to https://docs.oracle.com/en-us/iaas/tools/python/latest/logging.html for ways to log the request/response details.",
                        "troubleshooting_tips": "See https://docs.oracle.com/iaas/Content/API/References/apierrors.htm#apierrors_500__500_internalerror for more information about resolving this error. If you are unable to resolve this None issue, please contact Oracle support and provide them this full error message.",
                    },
                    "reason": "An internal error occurred.",
                    "request_id": TestDataset.mock_request_id,
                },
            ],
            [
                "oci ClientError",
                ConfigFileNotFound("Could not find config file at the given path."),
                {
                    "status": 400,
                    "message": "Something went wrong with your request.",
                    "service_payload": {},
                    "reason": "ConfigFileNotFound: Could not find config file at the given path.",
                    "request_id": TestDataset.mock_request_id,
                },
            ],
            [
                "oci MissingEndpointForNonRegionalServiceClientError",
                MissingEndpointForNonRegionalServiceClientError(
                    "An endpoint must be provided for a non-regional service client"
                ),
                {
                    "status": 400,
                    "message": "Something went wrong with your request.",
                    "service_payload": {},
                    "reason": "MissingEndpointForNonRegionalServiceClientError: An endpoint must be provided for a non-regional service client",
                    "request_id": TestDataset.mock_request_id,
                },
            ],
            [
                "oci RequestException",
                RequestException("An exception occurred when making the request"),
                {
                    "status": 400,
                    "message": "Something went wrong with your request.",
                    "service_payload": {},
                    "reason": "RequestException: An exception occurred when making the request",
                    "request_id": TestDataset.mock_request_id,
                },
            ],
            [
                "oci ConnectTimeout",
                ConnectTimeout(
                    "The request timed out while trying to connect to the remote server."
                ),
                {
                    "status": 408,
                    "message": "Server is taking too long to response, please try again.",
                    "service_payload": {},
                    "reason": "ConnectTimeout: The request timed out while trying to connect to the remote server.",
                    "request_id": TestDataset.mock_request_id,
                },
            ],
            [
                "oci MultipartUploadError",
                MultipartUploadError(),
                {
                    "status": 500,
                    "message": "Internal Server Error",
                    "service_payload": {},
                    "reason": f"MultipartUploadError: MultipartUploadError exception has occured. {UPLOAD_MANAGER_DEBUG_INFORMATION_LOG}",
                    "request_id": TestDataset.mock_request_id,
                },
            ],
            [
                "oci CompositeOperationError",
                CompositeOperationError(),
                {
                    "status": 500,
                    "message": "Internal Server Error",
                    "service_payload": {},
                    "reason": "CompositeOperationError: ",
                    "request_id": TestDataset.mock_request_id,
                },
            ],
            [
                "AquaError",
                AquaError(reason="Mocking AQUA error.", status=403, service_payload={}),
                {
                    "status": 403,
                    "message": "We're having trouble processing your request with the information provided.",
                    "service_payload": {},
                    "reason": "Mocking AQUA error.",
                    "request_id": TestDataset.mock_request_id,
                },
            ],
            [
                "HTTPError",
                HTTPError(400, "The request `/test` is invalid."),
                {
                    "status": 400,
                    "message": "The request `/test` is invalid.",
                    "service_payload": {},
                    "reason": "The request `/test` is invalid.",
                    "request_id": TestDataset.mock_request_id,
                },
            ],
            [
                "ADS Error",
                ValueError("Mocking ADS internal error."),
                {
                    "status": 500,
                    "message": "Internal Server Error",
                    "service_payload": {},
                    "reason": "ValueError: Mocking ADS internal error.",
                    "request_id": TestDataset.mock_request_id,
                },
            ],
        ]
    )
    @patch("uuid.uuid4")
    def test_handle_exceptions(self, name, error, expected_reply, mock_uuid):
        """Tests handling error decorator."""
        from ads.aqua.decorator import handle_exceptions

        mock_uuid.return_value = TestDataset.mock_request_id
        expected_call = json.dumps(expected_reply)

        @handle_exceptions
        def mock_function(self):
            raise error

        mock_function(self.test_instance)

        self.test_instance.finish.assert_called_with(expected_call)
