#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from dataclasses import fields
from datetime import datetime, timedelta
from typing import Dict, Optional

from cachetools import TTLCache, cached
from tornado.web import HTTPError

from ads.aqua import ODSC_MODEL_COMPARTMENT_OCID
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
