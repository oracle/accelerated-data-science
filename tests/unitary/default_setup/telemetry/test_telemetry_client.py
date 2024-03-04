#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from unittest.mock import patch, PropertyMock

from ads.telemetry.client import TelemetryClient

class TestTelemetryClient:
    """Contains unittests for TelemetryClient."""

    endpoint = "https://objectstorage.us-ashburn-1.oraclecloud.com"

    def mocked_requests_head(*args, **kwargs):
        class MockResponse:
            def __init__(self, status_code):
                self.status_code = status_code

        return MockResponse(200)

    @patch('requests.head', side_effect=mocked_requests_head)
    @patch('ads.telemetry.client.TelemetryClient.service_endpoint', new_callable=PropertyMock,
           return_value=endpoint)
    def test_telemetry_client_record_event(self, mock_endpoint, mock_head):
        """Tests TelemetryClient.record_event() with category/action and path, respectively.
        """
        data = {
            "cmd": "ads aqua model list",
            "category": "telemetry/aqua/service/model",
            "action": "list",
            "bucket": "test_bucket",
            "namespace": "test_namespace",
            "value": {
                "keyword": "test_service_model_name_or_id"
            }
        }
        category = data["category"]
        action = data["action"]
        bucket = data["bucket"]
        namespace = data["namespace"]
        value = data["value"]
        path = f"{category}/{action}"
        expected_endpoint = f"{self.endpoint}/n/{namespace}/b/{bucket}/o/{path}"

        telemetry = TelemetryClient(bucket=bucket, namespace=namespace)
        telemetry.record_event(category=category, action=action)
        telemetry.record_event(path=path)
        telemetry.record_event(path=path, **value)

        expected_headers = [
            {'User-Agent': ''},
            {'User-Agent': ''},
            {'User-Agent': 'keyword=test_service_model_name_or_id'}
        ]
        i = 0
        for call_args in mock_head.call_args_list:
            args, kwargs = call_args
            assert all(endpoint == expected_endpoint for endpoint in args)
            assert kwargs['headers'] == expected_headers[i]
            i += 1
