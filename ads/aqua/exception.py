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
    # reply
    messages: str
    reasons: str
    oci_payload: dict


class AquaError(Exception):
    """AquaError

    The base exception from which all exceptions raised by Aqua
    will inherit.
    """

    # TODO:
    # convert error -> reply format
    def to_jupylab(self):
        """build payload"""
        return asdict(
            ErrorPayload(
                messages=self.messages,
                reasons=self.reasons,
                oci_payload=self.oci_payload,
            )
        )


class AquaServiceError(AquaError):
    """Exception raised for server side error."""

    def __init__(
        self,
        messages: str,
        oci_payload: dict,
        status: int,
        reasons: str = "aquamodelapp.get",
    ):
        # messazge: from oci
        self.messages = messages
        self.oci_payload = oci_payload
        self.status = status
        self.reasons = reasons


class AquaClientError(AquaError):
    """Exception raised for client side error."""

    def __init__(
        self,
        messages: str,
        oci_payload: dict = None,
        status: int = None,
        reasons: str = "aquamodelapp.get",
    ):
        # messazge: from oci
        self.messages = messages
        self.oci_payload = oci_payload
        self.status = status
        self.reasons = reasons


def exception_handler(func):
    """Handles AquaError."""

    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except AquaServiceError as service_error:
            raise HTTPError(service_error.status or 500, service_error.to_jupylab())
        except AquaClientError as client_error:
            raise HTTPError(client_error.status or 400, client_error.to_jupylab())
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
                    messages="this is error.",
                    oci_payload=e.__dict__,
                    status=e.status,
                    reasons="aquamodelapp.get",
                )
            else:
                raise AquaClientError(messages="THisis error")
        except Exception as ex:
            raise AquaClientError(str(ex))

    return inner_function
