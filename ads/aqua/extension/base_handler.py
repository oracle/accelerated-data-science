#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import json
import traceback
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any

from notebook.base.handlers import APIHandler
from tornado.web import HTTPError, Application
from tornado import httputil
from ads.telemetry.client import TelemetryClient
from ads.config import AQUA_TELEMETRY_BUCKET, AQUA_TELEMETRY_BUCKET_NS
from ads.aqua import logger


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
        except:
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

        self.set_header("Content-Type", "application/json")
        reason = kwargs.get("reason")
        self.set_status(status_code, reason=reason)
        service_payload = kwargs.get("service_payload", {})
        message = self.get_default_error_messages(service_payload, str(status_code))
        reply = {
            "status": status_code,
            "message": message,
            "service_payload": service_payload,
            "reason": reason,
        }
        exc_info = kwargs.get("exc_info")
        if exc_info:
            logger.error("".join(traceback.format_exception(*exc_info)))
            e = exc_info[1]
            if isinstance(e, HTTPError):
                reply["message"] = e.log_message or message
                reply["reason"] = e.reason
                reply["request_id"] = str(uuid.uuid4())
            else:
                reply["request_id"] = str(uuid.uuid4())

        logger.warning(reply["message"])

        # telemetry may not be present if there is an error while initializing
        if hasattr(self, "telemetry"):
            self.telemetry.record_event_async(
                category="aqua/error",
                action=str(status_code),
                value=reason,
            )

        self.finish(json.dumps(reply))

    @staticmethod
    def get_default_error_messages(service_payload: dict, status_code: str):
        """Method that maps the error messages based on the operation performed or the status codes encountered."""

        messages = {
            "400": "Something went wrong with your request.",
            "403": "We're having trouble processing your request with the information provided.",
            "404": "Authorization Failed: The resource you're looking for isn't accessible.",
            "408": "Server is taking too long to response, please try again.",
            "500": "An error occurred while creating the resource.",
            "create": "Authorization Failed: Could not create resource.",
            "get": "Authorization Failed: The resource you're looking for isn't accessible.",
        }

        if service_payload and "operation_name" in service_payload:
            operation_name = service_payload["operation_name"]
            if operation_name:
                if operation_name.startswith("create"):
                    return messages["create"]
                elif operation_name.startswith("list") or operation_name.startswith(
                    "get"
                ):
                    return messages["get"] + f" Operation Name: {operation_name}."

        if status_code in messages:
            return messages[status_code]
        else:
            return "Unknown HTTP Error."


# todo: remove after error handler is implemented
class Errors(str):
    INVALID_INPUT_DATA_FORMAT = "Invalid format of input data."
    NO_INPUT_DATA = "No input data provided."
    MISSING_REQUIRED_PARAMETER = "Missing required parameter: '{}'"
