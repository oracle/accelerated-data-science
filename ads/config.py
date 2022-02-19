#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

OCI_ODSC_SERVICE_ENDPOINT = os.environ.get("OCI_ODSC_SERVICE_ENDPOINT")
OCI_IDENTITY_SERVICE_ENDPOINT = os.environ.get("OCI_IDENTITY_SERVICE_ENDPOINT")

NB_SESSION_COMPARTMENT_OCID = os.environ.get("NB_SESSION_COMPARTMENT_OCID") or None
PROJECT_OCID = os.environ.get("PROJECT_OCID") or None
NB_SESSION_OCID = os.environ.get("NB_SESSION_OCID") or None
USER_OCID = os.environ.get("USER_OCID") or None

# resource principal env vars
OCI_RESOURCE_PRINCIPAL_RPT_ENDPOINT = os.environ.get(
    "OCI_RESOURCE_PRINCIPAL_RPT_ENDPOINT"
)
OCI_RESOURCE_PRINCIPAL_VERSION = os.environ.get("OCI_RESOURCE_PRINCIPAL_VERSION")
OCI_RESOURCE_PRINCIPAL_RPT_PATH = os.environ.get("OCI_RESOURCE_PRINCIPAL_RPT_PATH")
OCI_RESOURCE_PRINCIPAL_RPT_ID = os.environ.get("OCI_RESOURCE_PRINCIPAL_RPT_ID")

TENANCY_OCID = os.environ.get("TENANCY_OCID")
OCI_REGION_METADATA = os.environ.get("OCI_REGION_METADATA")

JOB_RUN_OCID = os.environ.get("JOB_RUN_OCID") or None
JOB_RUN_COMPARTMENT_OCID = os.environ.get("JOB_RUN_COMPARTMENT_OCID")
