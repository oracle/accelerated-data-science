#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Decorator module."""

import sys
from functools import wraps

from oci.exceptions import (
    ClientError,
    CompositeOperationError,
    ConnectTimeout,
    MissingEndpointForNonRegionalServiceClientError,
    MultipartUploadError,
    RequestException,
    ServiceError,
)

from ads.aqua.exception import AquaError
from ads.aqua.extension.base_handler import AquaAPIhandler


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
        except (
            ClientError,
            MissingEndpointForNonRegionalServiceClientError,
            RequestException,
        ) as error:
            self.write_error(
                status_code=400,
                reason=f"{type(ex).__name__}: {str(error)}",
                exc_info=sys.exc_info(),
            )
        except ConnectTimeout as error:
            self.write_error(
                status_code=408,
                reason=f"{type(ex).__name__}: {str(error)}",
                exc_info=sys.exc_info(),
            )
        except (MultipartUploadError, CompositeOperationError) as error:
            self.write_error(
                status_code=500,
                reason=f"{type(ex).__name__}: {str(error)}",
                exc_info=sys.exc_info(),
            )
        except AquaError as error:
            self.write_error(
                status_code=error.status,
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
