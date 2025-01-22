#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime, timedelta
from typing import Dict, Optional

import oci
from cachetools import TTLCache, cached
from tornado.web import HTTPError

from ads.aqua import ODSC_MODEL_COMPARTMENT_OCID
from ads.aqua.common.utils import fetch_service_compartment
from ads.aqua.constants import (
    AQUA_EXTENSION_LOAD_DEFAULT_TIMEOUT,
    AQUA_EXTENSION_LOAD_MAX_ATTEMPTS,
)
from ads.aqua.extension.errors import Errors
from ads.config import THREADED_DEFAULT_TIMEOUT


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


@contextmanager
def use_temporary_envs(overrides: dict):
    existing_vars: dict = {}
    for key, new_value in overrides.items():
        existing_vars[key] = os.getenv(key)
        os.environ[key] = new_value
    try:
        yield
    finally:
        for key, old_value in existing_vars.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@cached(cache=TTLCache(maxsize=1, ttl=timedelta(minutes=1), timer=datetime.now))
def ui_compatability_check():
    """This method caches the service compartment OCID details that is set by either the environment variable or if
    fetched from the configuration. The cached result is returned when multiple calls are made in quick succession
    from the UI to avoid multiple config file loads."""
    if ODSC_MODEL_COMPARTMENT_OCID:
        return ODSC_MODEL_COMPARTMENT_OCID

    # set threaded default to 2x the extension load timeout value
    env_overrides = {
        "THREADED_DEFAULT_TIMEOUT": max(
            THREADED_DEFAULT_TIMEOUT, AQUA_EXTENSION_LOAD_DEFAULT_TIMEOUT * 2
        )
    }
    with use_temporary_envs(env_overrides):
        retry_strategy = oci.retry.RetryStrategyBuilder(
            max_attempts=AQUA_EXTENSION_LOAD_MAX_ATTEMPTS
        )
        return fetch_service_compartment(
            config_kwargs={
                "timeout": AQUA_EXTENSION_LOAD_DEFAULT_TIMEOUT,
                "retry_strategy": retry_strategy.get_retry_strategy(),
            },
            raise_error=True,
        )
