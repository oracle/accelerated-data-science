#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from ads import __version__
from enum import Enum, auto

import ads.config

EXTRA_USER_AGENT_INFO = "EXTRA_USER_AGENT_INFO"
USER_AGENT_KEY = "additional_user_agent"
ENV_MD_OCID = "MD_OCID"
UNKNOWN = "UNKNOWN"


class Surface(Enum):
    """
    An Enum class for labeling the surface where ADS is being used
    """

    WORKSTATION = auto()
    DATASCIENCE_JOB = auto()
    DATASCIENCE_NOTEBOOK = auto()
    DATASCIENCE_MODEL_DEPLOYMENT = auto()
    DATAFLOW = auto()
    OCI_SERVICE = auto()

    @classmethod
    def surface(cls):
        surface = cls.WORKSTATION
        if (
            ads.config.OCI_RESOURCE_PRINCIPAL_VERSION
            or ads.config.OCI_RESOURCE_PRINCIPAL_RPT_PATH
            or ads.config.OCI_RESOURCE_PRINCIPAL_RPT_ID
        ):
            surface = cls.OCI_SERVICE
            if ads.config.JOB_RUN_OCID:
                surface = cls.DATASCIENCE_JOB
            elif ads.config.NB_SESSION_OCID:
                surface = cls.DATASCIENCE_NOTEBOOK
            elif ads.config.MD_OCID:
                surface = cls.DATASCIENCE_MODEL_DEPLOYMENT
            elif ads.config.DATAFLOW_RUN_OCID:
                surface = cls.DATAFLOW
        return surface


def update_oci_client_config(config={}):
    if not config.get(USER_AGENT_KEY):
        config[
            USER_AGENT_KEY
        ] = f"Oracle-ads/version={__version__}/surface={Surface.surface().name}"  # To be enabled in future - /api={os.environ.get(EXTRA_USER_AGENT_INFO,UNKNOWN)}"
    return config
