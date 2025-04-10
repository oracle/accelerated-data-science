#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import re
import traceback
import uuid
from dataclasses import fields
from http.client import responses
from typing import Dict, Optional

from tornado.web import HTTPError

from ads.aqua import logger
from ads.aqua.constants import (
    AQUA_TROUBLESHOOTING_LINK,
    OCI_OPERATION_FAILURES,
    STATUS_CODE_MESSAGES,
)
from ads.aqua.extension.errors import Errors, ReplyDetails


def validate_function_parameters(data_class, input_data: Dict):
    """Validates if the required parameters are provided in input data."""
    required_parameters = [
        field.name for field in fields(data_class) if field.type != Optional[field.type]
    ]

    for required_parameter in required_parameters:
        if not input_data.get(required_parameter):
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format(required_parameter)
            )


def get_default_error_messages(
    service_payload: dict,
    status_code: str,
    default_msg: str = "Unknown HTTP Error.",
) -> str:
    """Method that maps the error messages based on the operation performed or the status codes encountered."""

    if service_payload and "operation_name" in service_payload:
        operation_name = service_payload.get("operation_name")

        if operation_name and status_code in STATUS_CODE_MESSAGES:
            return f"{STATUS_CODE_MESSAGES[status_code]}\n{service_payload.get('message')}\nOperation Name: {operation_name}."

    return STATUS_CODE_MESSAGES.get(status_code, default_msg)


def get_documentation_link(key: str) -> str:
    """Generates appropriate GitHub link to AQUA Troubleshooting Documentation per the user's error."""
    github_header = re.sub(r"_", "-", key)
    return f"{AQUA_TROUBLESHOOTING_LINK}#{github_header}"


def get_troubleshooting_tips(service_payload: dict, status_code: str) -> str:
    """Maps authorization errors to potential solutions on Troubleshooting Page per Aqua Documentation on oci-data-science-ai-samples"""

    tip = f"For general tips on troubleshooting: {AQUA_TROUBLESHOOTING_LINK}"

    if status_code in (404, 400):
        failed_operation = service_payload.get("operation_name")

        if failed_operation in OCI_OPERATION_FAILURES:
            link = get_documentation_link(failed_operation)
            tip = OCI_OPERATION_FAILURES[failed_operation] + link

    return tip


def construct_error(status_code: int, **kwargs) -> ReplyDetails:
    """
    Formats an error response based on the provided status code and optional details.

    Args:
        status_code (int): The HTTP status code of the error.
        **kwargs: Additional optional parameters:
            - reason (str, optional): A brief reason for the error.
            - service_payload (dict, optional): Contextual error data from OCI SDK methods
            - message (str, optional): A custom error message, from error raised from failed AQUA methods calling OCI SDK methods
            - exc_info (tuple, optional): Exception information (e.g., from `sys.exc_info()`), used for logging.

    Returns:
        ReplyDetails: A Pydantic object containing details about the formatted error response.
        kwargs:
                - "status" (int): The HTTP status code.
                - "troubleshooting_tips" (str): a GitHub link to AQUA troubleshooting docs, may be linked to a specific header.
                - "message" (str): error message.
                - "service_payload" (Dict[str, Any], optional) : Additional context from OCI Python SDK call.
                - "reason" (str): The reason for the error.
                - "request_id" (str): A unique identifier for tracking the error.

    Logs:
        - Logs the error details with a unique request ID.
        - If `exc_info` is provided and contains an `HTTPError`, updates the response message and reason accordingly.

    """
    reason = kwargs.get("reason", "Unknown Error")
    service_payload = kwargs.get("service_payload", {})
    default_msg = responses.get(status_code, "Unknown HTTP Error")
    message = get_default_error_messages(
        service_payload, str(status_code), kwargs.get("message", default_msg)
    )

    tips = get_troubleshooting_tips(service_payload, status_code)

    reply = ReplyDetails(
        status=status_code,
        troubleshooting_tips=tips,
        message=message,
        service_payload=service_payload,
        reason=reason,
        request_id=str(uuid.uuid4()),
    )

    exc_info = kwargs.get("exc_info")
    if exc_info:
        logger.error(
            f"Error Request ID: {reply.request_id}\n"
            f"Error: {''.join(traceback.format_exception(*exc_info))}"
        )
        e = exc_info[1]
        if isinstance(e, HTTPError):
            reply.message = e.log_message or message
            reply.reason = e.reason if e.reason else reply.reason

    logger.error(
        f"Error Request ID: {reply.request_id}\n"
        f"Error: {reply.message} {reply.reason}"
    )
    return reply
