#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Exception module."""

from oci.exceptions import ServiceError, ClientError
from tornado.web import HTTPError


class AquaError(Exception):
    """AquaError

    The base exception from which all exceptions raised by Aqua
    will inherit.
    """

    pass


class AquaServiceError(AquaError):
    """Exception raised for server side error."""

    def __init__(self, opc_request_id: str, status_code: int):
        super().__init__(
            f"Error occurred when invoking service. opc-request-id: {opc_request_id}. status code: {status_code}."
        )


class AquaClientError(AquaError):
    """Exception raised for client side error."""

    pass


def exception_handler(func):
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except AquaServiceError as service_error:
            raise HTTPError(500, str(service_error))
        except AquaClientError as client_error:
            raise HTTPError(400, str(client_error))
        except Exception as internal_error:
            raise HTTPError(500, str(internal_error))

    return inner_function


def oci_exception_handler(func):
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except ServiceError as se:
            raise AquaServiceError(opc_request_id=se.request_id, status_code=se.code)
        except ClientError as ce:
            raise AquaClientError(str(ce))

    return inner_function
