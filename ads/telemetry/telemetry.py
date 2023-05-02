#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from enum import Enum, auto
from typing import Any, Dict, Optional

import ads.config
from ads import __version__
from ads.common import logger

LIBRARY = "Oracle-ads"
EXTRA_USER_AGENT_INFO = "EXTRA_USER_AGENT_INFO"
USER_AGENT_KEY = "additional_user_agent"
UNKNOWN = "UNKNOWN"


class Surface(Enum):
    """
    An Enum class for labeling the surface where ADS is being used.
    """

    WORKSTATION = auto()
    DATASCIENCE_JOB = auto()
    DATASCIENCE_NOTEBOOK = auto()
    DATASCIENCE_MODEL_DEPLOYMENT = auto()
    DATAFLOW = auto()
    OCI_SERVICE = auto()
    DATASCIENCE_PIPELINE = auto()

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
            elif ads.config.PIPELINE_RUN_OCID:
                surface = cls.DATASCIENCE_PIPELINE
        return surface


def update_oci_client_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Adds user agent information to the config if it is not setup yet.

    Returns
    -------
    Dict
        The updated configuration.
    """

    try:
        config = config or {}
        if not config.get(USER_AGENT_KEY):
            config.update(
                {
                    USER_AGENT_KEY: (
                        f"{LIBRARY}/version={__version__}#"
                        f"surface={Surface.surface().name}#"
                        f"api={os.environ.get(EXTRA_USER_AGENT_INFO,UNKNOWN) or UNKNOWN}"
                    )
                }
            )
    except Exception as ex:
        logger.debug(ex)

    return config
