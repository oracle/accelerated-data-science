#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Exception module."""


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
            f"Error occurred when invoking service. opc-request-id: {opc_request_id}. status code: {status_code}"
        )


class AquaClientError(AquaError):
    """Exception raised for client side error."""

    def __init__(self, invalid_input: str):
        super().__init__(f"Invalid input {invalid_input}.")
