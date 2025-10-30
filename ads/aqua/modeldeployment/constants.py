#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.modeldeployment.constants
~~~~~~~~~~~~~~

This module contains constants used in Aqua Model Deployment.
"""

from ads.common.extended_enum import ExtendedEnum

DEFAULT_WAIT_TIME = 12000
DEFAULT_POLL_INTERVAL = 10


class DeploymentType(ExtendedEnum):
    SINGLE = "SINGLE"
    STACKED = "STACKED"
    MULTI = "MULTI"
