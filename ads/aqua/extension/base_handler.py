#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from notebook.base.handlers import APIHandler
from tornado import httputil
from tornado.web import Application

from ads.aqua.common.utils import is_pydantic_model
from ads.aqua.extension.utils import construct_error
from ads.config import AQUA_TELEMETRY_BUCKET, AQUA_TELEMETRY_BUCKET_NS
from ads.telemetry.client import TelemetryClient


class AquaAPIhandler(APIHandler):
    """Base handler for Aqua REST APIs."""

    def __init__(
        self,
        application: "Application",
        request: httputil.HTTPServerRequest,
        **kwargs: Any,
    ):
        super().__init__(application, request, **kwargs)

        try:
            self.telemetry = TelemetryClient(
                bucket=AQUA_TELEMETRY_BUCKET, namespace=AQUA_TELEMETRY_BUCKET_NS
            )
        except Exception:
            pass

    def prepare(self, *args, **kwargs):
        """The base class prepare is not required for Aqua"""
        pass

    @staticmethod
    def serialize(obj: Any):
        """Serialize the object.
        If the object is a dataclass, convert it to dictionary. Otherwise, convert it to string.
        """
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()

        if is_dataclass(obj):
            return asdict(obj)

        if is_pydantic_model(obj):
            return obj.model_dump()

        return str(obj)

    def finish(self, payload=None):  # pylint: disable=W0221
        """Ending the HTTP request by returning a payload and status code.

        Tornado finish() only takes one argument.
        Calling finish() with more than one arguments will cause error.
        """
        if payload is None:
            return super().finish()
        # If the payload is a list, put into a dictionary with key=data
        if isinstance(payload, list):
            payload = {"data": payload}
        # Convert the payload to a JSON serializable object
        payload = json.loads(json.dumps(payload, default=self.serialize))
        return super().finish(payload)

    def write_error(self, status_code, **kwargs):
        """AquaAPIhandler errors are JSON, not human pages."""

        reply_details = construct_error(status_code, **kwargs)

        self.set_header("Content-Type", "application/json")
        self.set_status(status_code, reason=reply_details.reason)

        # telemetry may not be present if there is an error while initializing
        if hasattr(self, "telemetry"):
            aqua_api_details = kwargs.get("aqua_api_details", {})
            self.telemetry.record_event_async(
                category="aqua/error",
                action=str(status_code),
                value=reply_details.reason,
                **aqua_api_details,
            )

        self.finish(reply_details)
