#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
import os
import sys

from ads import set_auth
from ads.aqua.utils import fetch_service_compartment
from ads.config import NB_SESSION_OCID, OCI_RESOURCE_PRINCIPAL_VERSION

ENV_VAR_LOG_LEVEL = "LOG_LEVEL"


def get_logger_level():
    """Retrieves logging level from environment variable `LOG_LEVEL`."""
    level = os.environ.get(ENV_VAR_LOG_LEVEL, "INFO").upper()
    return level


def configure_aqua_logger():
    """Configures the AQUA logger."""
    log_level = get_logger_level()
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s.%(module)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = configure_aqua_logger()


def set_log_level(log_level: str):
    """Global for setting logging level."""

    log_level = log_level.upper()
    logger.setLevel(log_level.upper())
    logger.handlers[0].setLevel(log_level)


if OCI_RESOURCE_PRINCIPAL_VERSION:
    set_auth("resource_principal")

ODSC_MODEL_COMPARTMENT_OCID = os.environ.get("ODSC_MODEL_COMPARTMENT_OCID")
if not ODSC_MODEL_COMPARTMENT_OCID:
    try:
        ODSC_MODEL_COMPARTMENT_OCID = fetch_service_compartment()
    except:
        pass

if not ODSC_MODEL_COMPARTMENT_OCID:
    logger.error(
        f"ODSC_MODEL_COMPARTMENT_OCID environment variable is not set for Aqua."
    )
    if NB_SESSION_OCID:
        logger.error(
            f"Aqua is not available for this notebook session {NB_SESSION_OCID}."
        )
    exit()
