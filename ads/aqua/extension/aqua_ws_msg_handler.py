#!/usr/bin/env python

# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import abstractmethod
from typing import List

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.models.ws_models import (
    AquaWsError,
    BaseRequest,
    BaseResponse,
    ErrorResponse,
    RequestResponseType,
)
from ads.aqua.extension.utils import construct_error
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
        except Exception:
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

        service_payload = kwargs.get("service_payload", {})
        reply_details = construct_error(status_code, **kwargs)

        # telemetry may not be present if there is an error while initializing
        if hasattr(self, "telemetry"):
            aqua_api_details = kwargs.get("aqua_api_details", {})
            self.telemetry.record_event_async(
                category="aqua/error",
                action=str(status_code),
                value=reply_details.reason,
                **aqua_api_details,
            )
        response = AquaWsError(
            status=status_code,
            message=reply_details.message,
            service_payload=service_payload,
            reason=reply_details.reason,
        )
        base_message = BaseRequest.from_json(self.message, ignore_unknown=True)
        return ErrorResponse(
            message_id=base_message.message_id,
            kind=RequestResponseType.Error,
            data=response,
        )
