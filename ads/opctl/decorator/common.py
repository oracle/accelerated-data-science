#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict, Callable
from functools import wraps

from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.common.auth import AuthContext, AuthType

RUN_ID_FIELD = "run_id"


def print_watch_command(func: callable) -> Callable:
    """The decorator to help build the `opctl watch` command."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Dict:
        result = func(*args, **kwargs)
        if result and isinstance(result, Dict) and RUN_ID_FIELD in result:
            msg_header = f"{'*' * 40} To monitor the progress of a task, execute the following command {'*' * 40}"
            print(msg_header)
            print(f"ads opctl watch {result[RUN_ID_FIELD]}")
            print("*" * len(msg_header))
        return result

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
