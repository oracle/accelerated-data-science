#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Exception module."""

import sys
from functools import wraps

from oci.exceptions import ClientError, ServiceError

from ads.aqua.extension.base_handler import AquaAPIhandler


class AquaError(Exception):
    """AquaError

    The base exception from which all exceptions raised by Aqua
    will inherit.
    """

    def __init__(
        self,
        reason: str,
        service_payload: dict = None,
        status: int = None,
    ):
        """Initializes an AquaError.

        Parameters
        ----------
        reason: str
            User friendly error message.
        service_payload: dict
            Payload to contain more details related to the error.
        status: int
            Http status code that are going to raise.
        """
        self.service_payload = service_payload
        if status is None and self.service_payload and "status" in self.service_payload:
            self.status = self.service_payload["status"]
        else:
            self.status = status
        self.reason = reason


class AquaServiceError(AquaError):
    """Exception raised for server side error."""

    # TODO: remove? as we handle oci service error directly


class AquaClientError(AquaError):
    """Exception raised for client side error."""

    # TODO: split into more specific error,
    # like ValueError, FileNotFoundError, etc...


def exception_handler(func):
    """Writes errors raised during call to JSON."""

    @wraps(func)
    def inner_function(self: AquaAPIhandler, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except ServiceError as error:
            self.write_error(
                status_code=error.status or 500,
                reason=error.message,
                service_payload=error.args[0] if error.args else None,
                exc_info=sys.exc_info(),
            )
        except ClientError as error:
            self.write_error(
                status_code=400,
                reason=str(error),
                service_payload={"args": error.args},
                exc_info=sys.exc_info(),
            )
        # TODO: need to catch other OCI exceptions
        except AquaServiceError as error:
            self.write_error(
                status_code=error.status or 500,
                reason=error.reason,
                service_payload=error.service_payload,
                exc_info=sys.exc_info(),
            )
        except AquaClientError as error:
            self.write_error(
                status_code=error.status or 400,
                reason=error.reason,
                service_payload=error.service_payload,
                exc_info=sys.exc_info(),
            )
        except Exception as ex:
            self.write_error(
                status_code=500,
                reason=f"{type(ex).__name__}: {str(ex)}",
                exc_info=sys.exc_info(),
            )

    return inner_function
