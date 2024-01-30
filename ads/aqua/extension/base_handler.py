#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import json
from dataclasses import asdict, is_dataclass
from typing import Any
from notebook.base.handlers import APIHandler
from tornado.web import HTTPError
from http.client import responses
import traceback


class AquaAPIhandler(APIHandler):
    """Base handler for Aqua REST APIs."""

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
            payload = {}
        # If the payload is a list, put into a dictionary with key=data
        if isinstance(payload, list):
            payload = {"data": payload}
        # Convert the payload to a JSON serializable object
        payload = json.loads(json.dumps(payload, default=self.serialize))
        return super().finish(payload)

    def write_error(self, status_code, **kwargs):
        """APIHandler errors are JSON, not human pages."""
        self.set_header("Content-Type", "application/json")
        message = responses.get(status_code, "Unknown HTTP Error")
        reply = {
            "message": message,
        }
        exc_info = kwargs.get("exc_info")
        if exc_info:
            # TODO: save the log
            self.log.error("".join(traceback.format_exception(*exc_info)))
            e = exc_info[1]
            import pdb

            pdb.set_trace()
            if isinstance(e, HTTPError):
                reply["message"] = e.log_message or message
                reply["reason"] = e.reason
                reply["service_payload"] = e.args[0] if e.args else None
                reply["traceback"] = "".join(traceback.format_exception(*exc_info))
            else:
                reply["message"] = "Unhandled error"
                reply["reason"] = None
                reply["service_payload"] = None
                reply["traceback"] = "".join(traceback.format_exception(*exc_info))
        self.log.warning(reply["message"])
        self.finish(json.dumps(reply))
