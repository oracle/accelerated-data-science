#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib
import os
from logging import getLogger

ENV_VAR_LOG_LEVEL = "ADS_AQUA_LOG_LEVEL"

_LAZY_ATTRS = {
    "AsyncClient": ("ads.aqua.client.client", "AsyncClient"),
    "Client": ("ads.aqua.client.client", "Client"),
    "HttpxOCIAuth": ("ads.aqua.client.client", "HttpxOCIAuth"),
    "get_async_httpx_client": ("ads.aqua.client.client", "get_async_httpx_client"),
    "get_httpx_client": ("ads.aqua.client.client", "get_httpx_client"),
}

__all__ = [
    "AsyncClient",
    "Client",
    "ENV_VAR_LOG_LEVEL",
    "HttpxOCIAuth",
    "get_async_httpx_client",
    "get_httpx_client",
    "get_logger_level",
    "logger",
    "set_log_level",
]


def get_logger_level():
    """Retrieves logging level from environment variable `ADS_AQUA_LOG_LEVEL`."""
    level = os.environ.get(ENV_VAR_LOG_LEVEL, "INFO").upper()
    return level


logger = getLogger(__name__)
logger.setLevel(get_logger_level())


def set_log_level(log_level: str):
    """Global for setting logging level."""

    log_level = log_level.upper()
    logger.setLevel(log_level.upper())


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
