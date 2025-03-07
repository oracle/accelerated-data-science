#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import os

from ads.aqua import (
    ENV_VAR_LOG_LEVEL,
    ODSC_MODEL_COMPARTMENT_OCID,
    logger,
    set_log_level,
)
from ads.aqua.common.errors import AquaCLIError, AquaConfigError
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.finetuning import AquaFineTuningApp
from ads.aqua.model import AquaModelApp
from ads.aqua.modeldeployment import AquaDeploymentApp
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
        debug: bool = None,
        verbose: bool = None,
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
        debug (bool):
            Sets the logging level for the application to `DEBUG`.
        verbose (bool):
            Sets the logging level for the application to `INFO`.

        Raises
        ------
        AquaCLIError:
            When `--verbose` and `--debug` being used together.
            When missing required `ODSC_MODEL_COMPARTMENT_OCID` env var.
        """
        if verbose is not None and debug is not None:
            raise AquaCLIError(
                "Cannot use `--debug` and `--verbose` at the same time. "
                "Please select either `--debug` for `DEBUG` level logging or "
                "`--verbose` for `INFO` level logging."
            )
        elif verbose is not None:
            self._validate_value("--verbose", verbose)
            aqua_log_level = "INFO"
        elif debug is not None:
            self._validate_value("--debug", debug)
            aqua_log_level = "DEBUG"
        else:
            if log_level.upper() not in LOG_LEVELS:
                logger.warning(
                    f"Log level should be one of {LOG_LEVELS}. Setting default to ERROR."
                )
                log_level = "ERROR"
            aqua_log_level = log_level.upper()

        set_log_level(aqua_log_level)

        if not ODSC_MODEL_COMPARTMENT_OCID:
            if NB_SESSION_OCID:
                raise AquaConfigError(
                    f"Aqua is not available for the notebook session {NB_SESSION_OCID}. For more information, "
                    f"please refer to the documentation."
                )
            raise AquaConfigError(
                "ODSC_MODEL_COMPARTMENT_OCID environment variable is not set for Aqua."
            )

    @staticmethod
    def _validate_value(flag, value):
        """Check if the given value for bool flag is valid.

        Raises
        ------
        AquaCLIError:
            When the given value for bool flag is invalid.
        """
        if value not in [True, False]:
            raise AquaCLIError(
                f"Invalid input `{value}` for flag: {flag}, a boolean value is required. "
                "If you intend to chain a function call to the result, please separate the "
                "flag and the subsequent function call with separator `-`."
            )
