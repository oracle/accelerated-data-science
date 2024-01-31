#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import json
from dataclasses import asdict, is_dataclass
from typing import Any
from notebook.base.handlers import APIHandler


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


# todo: remove after error handler is implemented
class Errors(str):
    INVALID_INPUT_DATA_FORMAT = "Invalid format of input data."
    NO_INPUT_DATA = "No input data provided."
    MISSING_REQUIRED_PARAMETER = "Missing required parameter: '{}'"
