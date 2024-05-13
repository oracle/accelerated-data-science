#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import traceback
from abc import abstractmethod
from http.client import responses
from typing import List

from tornado.web import HTTPError

from ads.aqua import logger
from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.extension.models.ws_models import (
    AquaWsError,
    BaseRequest,
    BaseResponse,
    ErrorResponse,
    RequestResponseType,
)
from ads.config import AQUA_TELEMETRY_BUCKET, AQUA_TELEMETRY_BUCKET_NS
from ads.telemetry.client import TelemetryClient


class AquaWSMsgHandler:
    message: str

    def __init__(self, message: str):
        self.message = message
        try:
            self.telemetry = TelemetryClient(
                bucket=AQUA_TELEMETRY_BUCKET, namespace=AQUA_TELEMETRY_BUCKET_NS
            )
        except:
            pass

    @staticmethod
    @abstractmethod
    def get_message_types() -> List[RequestResponseType]:
        """This method should be implemented by the child class.
        This method should return a list of RequestResponseType that the child class can handle
        """
        pass

    @abstractmethod
    @handle_exceptions
    def process(self) -> BaseResponse:
        """This method should be implemented by the child class.
        This method will contain the core logic to be executed for handling the message
        """
        pass

    def write_error(self, status_code, **kwargs):
        """AquaWSMSGhandler errors are JSON, not human pages."""
        reason = kwargs.get("reason")
        service_payload = kwargs.get("service_payload", {})
        default_msg = responses.get(status_code, "Unknown HTTP Error")
        message = AquaAPIhandler.get_default_error_messages(
            service_payload, str(status_code), kwargs.get("message", default_msg)
        )
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
            else:
                logger.warning(reply["message"])
        # telemetry may not be present if there is an error while initializing
        if hasattr(self, "telemetry"):
            self.telemetry.record_event_async(
                category="aqua/error",
                action=str(status_code),
                value=reason,
            )
        response = AquaWsError(
            status=status_code,
            message=message,
            service_payload=service_payload,
            reason=reason,
        )
        base_message = BaseRequest.from_json(self.message, ignore_unknown=True)
        return ErrorResponse(
            message_id=base_message.message_id,
            kind=RequestResponseType.Error,
            data=response,
        )
