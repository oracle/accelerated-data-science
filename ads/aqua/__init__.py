#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
import sys
import os
from ads.aqua.utils import fetch_service_compartment
from ads.config import OCI_RESOURCE_PRINCIPAL_VERSION
from ads import set_auth

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)

if OCI_RESOURCE_PRINCIPAL_VERSION:
    set_auth("resource_principal")

ODSC_MODEL_COMPARTMENT_OCID = os.environ.get("ODSC_MODEL_COMPARTMENT_OCID")
if not ODSC_MODEL_COMPARTMENT_OCID:
    try:
        ODSC_MODEL_COMPARTMENT_OCID = fetch_service_compartment()
    except Exception as e:
        logger.error(
            f"ODSC_MODEL_COMPARTMENT_OCID environment variable is not set for Aqua, due to {e}."
        )
