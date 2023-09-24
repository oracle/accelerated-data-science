#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from functools import wraps
from typing import Callable, Dict, List

from ads.common.auth import AuthContext
from ads.opctl import logger
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger

RUN_ID_FIELD = "run_id"


class OpctlEnvironmentError(Exception):
    """The custom error to validate OPCTL environment."""

    NOT_SUPPORTED_ENVIRONMENTS = (
        "Notebook Sessions",
        "Data Science Jobs",
        "ML Pipelines",
        "Data Flow Applications",
    )

    def __init__(self):
        super().__init__(
            "This operation cannot be executed in the current environment. "
            f"It is not supported in: {', '.join(self.NOT_SUPPORTED_ENVIRONMENTS)}."
        )


def print_watch_command(func: callable) -> Callable:
    """The decorator to help build the `opctl watch` command."""

    @wraps(func)
    def wrapper(*args: List, **kwargs: Dict) -> Dict:
        result = func(*args, **kwargs)
        if result and isinstance(result, Dict) and RUN_ID_FIELD in result:
            msg_header = (
                f"{'*' * 40} To monitor the progress of the task, "
                "execute the following command {'*' * 40}"
            )
            print(msg_header)
            print(f"ads opctl watch {result[RUN_ID_FIELD]}")
            print("*" * len(msg_header))
        return result

    return wrapper


def validate_environment(func: callable) -> Callable:
    """Validates whether an opctl command can be executed in the current environment."""

    @wraps(func)
    def wrapper(*args: List, **kwargs: Dict) -> Dict:
        try:
            import docker

            docker.from_env().version()
        except Exception as ex:
            logger.debug(ex)
            raise OpctlEnvironmentError()

        return func(*args, **kwargs)

    return wrapper


def click_options(options):
    """The decorator to help group the click options."""

    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def with_auth(func: Callable) -> Callable:
    """The decorator to add AuthContext to the method."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Dict:
        p = ConfigProcessor().step(ConfigMerger, **kwargs)

        with AuthContext(
            **{
                key: value
                for key, value in {
                    "auth": p.config["execution"]["auth"],
                    "oci_config_location": p.config["execution"]["oci_config"],
                    "profile": p.config["execution"]["oci_profile"],
                }.items()
                if value
            }
        ):
            return func(*args, **kwargs)

    return wrapper
