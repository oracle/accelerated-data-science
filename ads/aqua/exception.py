#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Exception module."""

from oci.exceptions import ServiceError
from tornado.web import HTTPError
from oci.exceptions import ServiceError


class AquaError(Exception):
    """AquaError

    The base exception from which all exceptions raised by Aqua
    will inherit.
    """

    pass


class AquaServiceError(AquaError):
    """Exception raised for server side error."""

    def __init__(self, opc_request_id: str, status_code: int, service_error: str):
        super().__init__(
            f"Error occurred when invoking service. opc-request-id: {opc_request_id}. status code: {status_code}."
            f"{service_error}"
        )


class AquaClientError(AquaError):
    """Exception raised for client side error."""

    pass


def exception_handler(func):
    """Handles AquaError."""

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
    """Handles OCI Service Error."""

    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except ServiceError as e:
            if e.status >= 500:
                raise AquaServiceError(
                    opc_request_id=e.request_id,
                    status_code=e.code,
                    service_error=str(e),
                )
            else:
                raise AquaClientError(str(e))
        except Exception as ex:
            raise AquaClientError(str(ex))

    return inner_function
