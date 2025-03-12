#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os
from logging import getLogger

from ads import logger, set_auth
from ads.aqua.client.client import (
    AsyncClient,
    Client,
    HttpxOCIAuth,
    get_async_httpx_client,
    get_httpx_client,
)
from ads.aqua.common.utils import fetch_service_compartment
from ads.config import OCI_RESOURCE_PRINCIPAL_VERSION

ENV_VAR_LOG_LEVEL = "ADS_AQUA_LOG_LEVEL"


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


if OCI_RESOURCE_PRINCIPAL_VERSION:
    set_auth("resource_principal")

ODSC_MODEL_COMPARTMENT_OCID = (
    os.environ.get("ODSC_MODEL_COMPARTMENT_OCID") or fetch_service_compartment()
)
