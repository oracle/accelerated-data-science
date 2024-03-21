#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
import os
import sys


def get_logger_level():
    """Retrieves logging level from environment variable `LOG_LEVEL`."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
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
    logger.setLevel(log_level)
    logger.handlers[0].setLevel(log_level)
