#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from unittest.mock import patch

import oci

from ads.telemetry.client import TelemetryClient

TEST_CONFIG = {
    "tenancy": "ocid1.tenancy.oc1..unique_ocid",
    "user": "ocid1.user.oc1..unique_ocid",
    "fingerprint": "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
    "key_file": "<path>/<to>/<key_file>",
    "region": "test_region",
}

EXPECTED_ENDPOINT = "https://objectstorage.test_region.oraclecloud.com"


class TestTelemetryClient:
    """Contains unittests for TelemetryClient."""

    @patch("oci.base_client.BaseClient.request")
    @patch("oci.signer.Signer")
    def test_telemetry_client_record_event(self, signer, request_call):
        """Tests TelemetryClient.record_event() with category/action and path, respectively."""
        data = {
            "cmd": "ads aqua model list",
            "category": "aqua/service/model",
            "action": "list",
            "bucket": "test_bucket",
            "namespace": "test_namespace",
            "value": {"keyword": "test_service_model_name_or_id"},
        }
        category = data["category"]
        action = data["action"]
        bucket = data["bucket"]
        namespace = data["namespace"]
        value = data["value"]

        with patch("oci.config.from_file", return_value=TEST_CONFIG):
            telemetry = TelemetryClient(bucket=bucket, namespace=namespace)
        telemetry.record_event(category=category, action=action)
        telemetry.record_event(category=category, action=action, **value)

        expected_agent_headers = [
            "",
            "keyword=test_service_model_name_or_id",
        ]

        assert len(request_call.call_args_list) == 2
        expected_url = f"{EXPECTED_ENDPOINT}/n/{namespace}/b/{bucket}/o/telemetry/{category}/{action}"

        # Event #1, no user-agent
        args, _ = request_call.call_args_list[0]
        request: oci.request.Request = args[0]
        operation = args[2]
        assert request.url == expected_url
        assert operation == "head_object"
        assert request.header_params["user-agent"] == expected_agent_headers[0]

        # Event #2, with user-agent
        args, _ = request_call.call_args_list[1]
        request: oci.request.Request = args[0]
        operation = args[2]
        assert request.url == expected_url
        assert operation == "head_object"
        assert request.header_params["user-agent"] == expected_agent_headers[1]
