#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import traceback
import uuid
from dataclasses import fields
from datetime import datetime, timedelta
from http.client import responses
from typing import Dict, Optional

from cachetools import TTLCache, cached
from tornado.web import HTTPError

from ads.aqua import ODSC_MODEL_COMPARTMENT_OCID, logger
from ads.aqua.common.utils import fetch_service_compartment
from ads.aqua.extension.errors import Errors


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


@cached(cache=TTLCache(maxsize=1, ttl=timedelta(minutes=1), timer=datetime.now))
def ui_compatability_check():
    """This method caches the service compartment OCID details that is set by either the environment variable or if
    fetched from the configuration. The cached result is returned when multiple calls are made in quick succession
    from the UI to avoid multiple config file loads."""
    return ODSC_MODEL_COMPARTMENT_OCID or fetch_service_compartment()

@staticmethod
def get_default_error_messages(
    service_payload: dict,
    status_code: str,
    default_msg: str = "Unknown HTTP Error.",
):
    """Method that maps the error messages based on the operation performed or the status codes encountered."""

    messages = {
        "400": "Something went wrong with your request.",
        "403": "We're having trouble processing your request with the information provided.",
        "404": "Authorization Failed: The resource you're looking for isn't accessible.",
        "408": "Server is taking too long to response, please try again.",
        "create": "Authorization Failed: Could not create resource.",
        "get": "Authorization Failed: The resource you're looking for isn't accessible.",
    }

    if service_payload and "operation_name" in service_payload:
        operation_name = service_payload["operation_name"]
        if operation_name:
            if operation_name.startswith("create"):
                return messages["create"] + f" Operation Name: {operation_name}."
            elif operation_name.startswith("list") or operation_name.startswith(
                "get"
            ):
                return messages["get"] + f" Operation Name: {operation_name}."

    if status_code in messages:
        return messages[status_code]
    else:
        return default_msg

def get_documentation_link(key: str):
    return f"https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/policies/troubleshooting.md#{key}"


def get_troubleshooting_tips(service_payload: str,
                             status_code: str,
                             tip: str = "For general tips on troubleshooting: https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/policies/README.md#setting-up-policies-manually"):
    """Maps authorization errors to potential solutions on Troubleshooting Page per Aqua Documentation on oci-data-science-ai-samples"""

    operations = {
        "list_model_deployments": "Unable to list model deployments. See tips for troubleshooting:",
        "list_models": "Unable to list models. See tips for troubleshooting:",
        "get_namespace": "Unable to access specified Object Storage Bucket. See tips for troubleshooting missing policies:",
        "list_log_groups":"Unable to access logs. See tips for troubleshooting missing policies:" ,
    }
    if status_code == "404":
        failed_operation = service_payload.get('operation_name')
        link = get_documentation_link(failed_operation)
        tip = operations[failed_operation] + link

    return tip


def construct_error(status_code, **kwargs):
    reason = kwargs.get("reason")
    service_payload = kwargs.get("service_payload", {})
    default_msg = responses.get(status_code, "Unknown HTTP Error")
    message = get_default_error_messages(
        service_payload, str(status_code), kwargs.get("message", default_msg)
    )

    tips = get_troubleshooting_tips(service_payload, status_code)

    reply = {
        "status": status_code,
        "troubleshooting_tips": tips,
        "message": message,
        "service_payload": service_payload,
        "reason": reason,
        "request_id": str(uuid.uuid4()),
    }
    exc_info = kwargs.get("exc_info")
    if exc_info:
        logger.error(
            f"Error Request ID: {reply['request_id']}\n"
            f"Error: {''.join(traceback.format_exception(*exc_info))}"
        )
        e = exc_info[1]
        if isinstance(e, HTTPError):
            reply["message"] = e.log_message or message
            reply["reason"] = e.reason if e.reason else reply["reason"]

    logger.error(
        f"Error Request ID: {reply['request_id']}\n"
        f"Error: {reply['message']} {reply['reason']}"
    )
    return reply, message, reason
