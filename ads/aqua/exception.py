#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Exception module."""

from oci.exceptions import ServiceError
from tornado.web import HTTPError
from oci.exceptions import ServiceError
from dataclasses import asdict, dataclass


@dataclass
class ErrorPayload:
    message: str
    reason: str
    service_payload: dict


class AquaError(Exception):
    """AquaError

    The base exception from which all exceptions raised by Aqua
    will inherit.
    """

    def to_payload(self) -> dict:
        """Builds error payload."""
        return asdict(
            ErrorPayload(
                message=self.message,
                reason=self.reason,
                service_payload=self.service_payload,
            )
        )


class AquaServiceError(AquaError):
    """Exception raised for server side error."""

    def __init__(
        self, message: str, service_payload: dict, status: int, reason: str = None
    ):
        """Initializes an AquaServiceError.

        Parameters
        ----------
        message: str
            User friendly error message.
        service_payload: dict
            `oci.exceptions.ServiceError`.
        status: int
            status code
        reason: (str, optional)
            Where the error raises. Defaults to None. For example, `ads.aqua.model.AquaModelApp.get`.
        """
        self.message = message
        self.service_payload = service_payload
        self.status = status
        self.reason = reason


class AquaClientError(AquaError):
    """Exception raised for client side error."""

    def __init__(
        self,
        message: str,
        service_payload: dict = None,
        status: int = None,
        reason: str = None,
    ):
        """Initializes an AquaServiceError.

        Parameters
        ----------
        message: str
            User friendly error message.
        service_payload: (dict, optional)
            `oci.exceptions.ServiceError`.
        status: (int, optional)
            status code
        reason: (str, optional)
            Where the error raises. Defaults to None. For example, `ads.aqua.model.AquaModelApp.get`.
        """
        self.message = message
        self.service_payload = service_payload
        self.status = status
        self.reason = reason


def exception_handler(func):
    """Handles AquaError."""

    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except AquaServiceError as service_error:
            raise HTTPError(
                service_error.status or 500,
                service_error.message,
                **service_error.to_payload(),
            )
        except AquaClientError as client_error:
            raise HTTPError(
                client_error.status or 400,
                client_error.message,
                **client_error.to_payload(),
            )
        except Exception as internal_error:
            raise HTTPError(500, str(internal_error))

    return inner_function


def oci_exception_handler(func):
    """Handles OCI Service Error."""

    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except ServiceError as e:
            error_details = e.args[0]
            if e.status >= 500:
                raise AquaServiceError(
                    message=e.message,
                    service_payload=error_details,
                    status=e.status,
                )
            else:
                raise AquaClientError(
                    message=e.message, service_payload=error_details, status=e.status
                )
        except Exception as ex:
            raise AquaClientError(message=str(ex))

    return inner_function
