#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import re
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, Optional

import ads.config
from ads import __version__
from ads.common import logger

TELEMETRY_ARGUMENT_NAME = "telemetry"


LIBRARY = "Oracle-ads"
EXTRA_USER_AGENT_INFO = "EXTRA_USER_AGENT_INFO"
USER_AGENT_KEY = "additional_user_agent"
UNKNOWN = "UNKNOWN"
DELIMITER = "&"


def update_oci_client_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Adds user agent information to the signer config if it is not setup yet.

    Parameters
    ----------
    config: Dict
        The signer configuration.

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


def telemetry(
    entry_point: str = "",
    name: str = "",
    environ_variable: str = EXTRA_USER_AGENT_INFO,
) -> Callable:
    """
    The telemetry decorator.
    Injects the Telemetry object into the `kwargs` arguments of the decorated function.
    This is essential for adding additional information to the telemetry from within the
    decorated function. Eventually this information will be merged into the `additional_user_agent`.

    Important Note: The telemetry decorator exclusively updates the specified environment
    variable and does not perform any additional actions.
    "

    Parameters
    ----------
    entry_point: str
        The entry point of the telemetry.
        Example: "plugin=project&action=run"
    name: str
        The name of the telemetry.
    environ_variable: (str, optional). Defaults to `EXTRA_USER_AGENT_INFO`.
        The name of the environment variable to capture the telemetry sequence.

    Examples
    --------
    >>> @telemetry(entry_point="plugin=project&action=run", name="ads")
    ... def test_function(**kwargs)
    ...     telemetry = kwargs.get("telemetry")
    ...     telemetry.add("param=hello_world")
    ...     print(telemetry)

    >>> test_function()
    ... "ads&plugin=project&action=run&param=hello_world"
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            telemetry = Telemetry(name=name, environ_variable=environ_variable).begin(
                entry_point
            )
            try:
                # todo: inject telemetry arg and later update all functions that use the @telemetry
                #   decorator to accept **kwargs. Comment the below line as some aqua apis don't support kwargs.
                # return func(*args, **{**kwargs, **{TELEMETRY_ARGUMENT_NAME: telemetry}})
                return func(*args, **kwargs)
            except:
                raise
            finally:
                telemetry.restore()

        return wrapper

    return decorator


class Surface(Enum):
    """
    An Enum class used to label the surface where ADS is being utilized.
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


@dataclass
class Telemetry:
    """
    This class is designed to capture a telemetry sequence and store it in the specified
    environment variable. By default the `EXTRA_USER_AGENT_INFO` environment variable is used.

    Attributes
    ----------
    name: (str, optional). Default to empty string.
        The name of the telemetry. The very beginning of the telemetry string.
    environ_variable: (str, optional). Defaults to `EXTRA_USER_AGENT_INFO`.
        The name of the environment variable to capture the telemetry sequence.
    """

    name: str = ""
    environ_variable: str = EXTRA_USER_AGENT_INFO

    def __post_init__(self):
        self.name = self._prepare(self.name)
        self._original_value = os.environ.get(self.environ_variable)
        os.environ[self.environ_variable] = ""

    def restore(self) -> "Telemetry":
        """Restores the original value of the environment variable.

        Returns
        -------
        self: Telemetry
            An instance of the Telemetry.
        """
        os.environ[self.environ_variable] = self._original_value or ""
        return self

    def clean(self) -> "Telemetry":
        """Cleans the associated environment variable.

        Returns
        -------
        self: Telemetry
            An instance of the Telemetry.
        """
        os.environ[self.environ_variable] = ""
        return self

    def _begin(self):
        self.clean()
        os.environ[self.environ_variable] = self.name

    def begin(self, value: str = "") -> "Telemetry":
        """
        This method should be invoked at the start of telemetry sequence capture.
        It resets the value of the associated environment variable.

        Parameters
        ----------
        value: (str, optional). Defaults to empty string.
            The value that need to be added to the telemetry.

        Returns
        -------
        self: Telemetry
            An instance of the Telemetry.
        """
        return self.clean().add(self.name).add(value)

    def add(self, value: str) -> "Telemetry":
        """Appends the new value to the telemetry data.

        Parameters
        ----------
        value: str
            The value that need to be added to the telemetry.

        Returns
        -------
        self: Telemetry
            An instance of the Telemetry.
        """
        if not os.environ.get(self.environ_variable):
            self._begin()

        if value:
            current_value = os.environ.get(self.environ_variable, "")
            new_value = self._prepare(value)

            if new_value not in current_value:
                os.environ[self.environ_variable] = (
                    f"{current_value}{DELIMITER}{new_value}"
                    if current_value
                    else new_value
                )
        return self

    def print(self) -> None:
        """Prints the telemetry sequence from environment variable."""
        print(f"{self.environ_variable} = {os.environ.get(self.environ_variable)}")

    def _prepare(self, value: str):
        """Replaces the special characters with the `_` in the input string."""
        return (
            re.sub("[^a-zA-Z0-9\.\-\_\&\=]", "_", re.sub(r"\s+", " ", value))
            if value
            else ""
        )
