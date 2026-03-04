#!/usr/bin/env python

# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import os

from ads.aqua import (
    ENV_VAR_LOG_LEVEL,
    logger,
    set_log_level,
)
from ads.aqua.common.errors import AquaCLIError
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.finetuning import AquaFineTuningApp
from ads.aqua.model import AquaModelApp
from ads.aqua.modeldeployment import AquaDeploymentApp
from ads.aqua.verify_policies import AquaVerifyPoliciesApp
from ads.common.utils import LOG_LEVELS


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
    verify_policies = AquaVerifyPoliciesApp

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

    @staticmethod
    def install():
        """Install ADS Aqua Extension from wheel file. Set enviroment variable `AQUA_EXTENSTION_PATH` to change the wheel file path.

        Return
        ------
        int:
            Installatation status.
        """
        import subprocess

        wheel_file_path = os.environ.get(
            "AQUA_EXTENSTION_PATH", "/ads/extension/adsjupyterlab_aqua_extension*.whl"
        )
        status = subprocess.run(f"pip install {wheel_file_path} --no-deps", shell=True, check=False)
        return status.check_returncode
