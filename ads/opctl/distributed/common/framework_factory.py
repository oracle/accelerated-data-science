#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.distributed.dask.dask_framework import DaskFramework
from typing import List


class FrameworkFactory:
    """
    Factory class for getting cluster set up details by framework
    """

    framework = {"dask": DaskFramework}

    @staticmethod
    def get_framework(key, *args, **kwargs):
        return FrameworkFactory.framework[key](*args, **kwargs)


def update_env_vars(config, env_vars: List):
    """
    env_vars: List, should be formatted as [{"name": "OCI__XXX", "value": YYY},]
    """
    # TODO move this to a class which checks the version, kind, type, etc.
    config["spec"]["Runtime"]["spec"]["environmentVariables"].extend(env_vars)
    return config
