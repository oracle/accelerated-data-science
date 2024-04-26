#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import os
import sys

from ads.aqua import (
    ENV_VAR_LOG_LEVEL,
    ODSC_MODEL_COMPARTMENT_OCID,
    logger,
    set_log_level,
)
from ads.aqua.deployment import AquaDeploymentApp
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.finetune import AquaFineTuningApp
from ads.aqua.model import AquaModelApp
from ads.common.utils import LOG_LEVELS
from ads.config import NB_SESSION_OCID


class AquaCommand:
    """Contains the command groups for project Aqua.

    Acts as an entry point for managing different components of the Aqua
    project including model management, fine-tuning, deployment, and
    evaluation.
    """

    model = AquaModelApp
    fine_tuning = AquaFineTuningApp
    deployment = AquaDeploymentApp
    evaluation = AquaEvaluationApp

    def __init__(
        self,
        log_level: str = os.environ.get(ENV_VAR_LOG_LEVEL, "ERROR").upper(),
    ):
        """
        Initialize the command line interface settings for the Aqua project.

        FLAGS
        -----
        log_level (str):
            Sets the logging level for the application.
            Default is retrieved from environment variable `ADS_AQUA_LOG_LEVEL`,
            or 'ERROR' if not set. Example values include 'DEBUG', 'INFO',
            'WARNING', 'ERROR', and 'CRITICAL'.
        """
        if log_level.upper() not in LOG_LEVELS:
            logger.error(
                f"Log level should be one of {LOG_LEVELS}. Setting default to ERROR."
            )
            log_level = "ERROR"
        set_log_level(log_level)
        # gracefully exit if env var is not set
        if not ODSC_MODEL_COMPARTMENT_OCID:
            logger.debug(
                "ODSC_MODEL_COMPARTMENT_OCID environment variable is not set for Aqua."
            )
            if NB_SESSION_OCID:
                logger.error(
                    f"Aqua is not available for the notebook session {NB_SESSION_OCID}. For more information, "
                    f"please refer to the documentation."
                )
            sys.exit(1)
