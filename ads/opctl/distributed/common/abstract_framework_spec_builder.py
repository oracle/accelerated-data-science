#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import ABCMeta
from typing import List


class AbstractFrameworkSpecBuilder(metaclass=ABCMeta):
    """
    Provides contract for implementing Framework specific Cluster Spec Builder

    In the example of jobs, this class handles adding framework specific environment variables to the job definition.

    NOTE: This class is not invoked while the cluster is running. Only after a call to `ads opctl`.
    """

    def __init__(self, config):
        self.config = config

    def update(self):
        new_env_vars = []
        # Call DaskFramework().build_env_vars
        # FrameworkFactory.get("kind").build_env_vars("spec, framework")
        user_entrypoint = self.config["spec"]["Runtime"]["spec"]["entrypoint"]
        args = self.config["spec"]["Runtime"]["spec"]["args"]

        # TODO:
        # kwargs = self.config["spec"]["Runtime"]["spec"]["kwargs"]

        nanny_port = self.config["spec"]["Framework"]["spec"]["schedulerConfig"][
            "nannyPortRange"
        ]
        worker_port = self.config["spec"]["Framework"]["spec"]["schedulerConfig"][
            "workerPortRange"
        ]
        worker_timeout = self.config["spec"]["Framework"]["spec"]["workerConfig"][
            "timeout"
        ]

        framework = self.config["spec"]["Framework"]["kind"]
        work_dir = self.config["spec"]["Framework"]["spec"]["workDir"]
        ephemeral = self.config["spec"]["Infrastructure"]["spec"]["cluster"][
            "ephemeral"
        ]
        worker_count = self.config["spec"]["Infrastructure"]["spec"]["cluster"][
            "numWorkers"
        ]

        new_env_vars.extend(
            [
                {"name": "OCI__ENTRY_SCRIPT", "value": user_entrypoint},
                {"name": "OCI__ENTRY_SCRIPT_ARGS", "value": args},  # TODO check with QQ
                {
                    "name": "OCI__ENTRY_SCRIPT_KWARGS",
                    "value": None,
                },  # TODO check with QQ
                {"name": "OCI__NANNY_PORT", "value": nanny_port},
                {"name": "OCI__WORKER_PORT", "value": worker_port},
                {"name": "OCI__WORKER_TIMEOUT", "value": worker_timeout},
                {"name": "OCI__WORKER_COUNT", "value": worker_count},
                {"name": "OCI__CLUSTER_TYPE", "value": framework},
                {
                    "name": "OCI__EPHEMERAL",
                    "value": ephemeral,
                },  # TODO: remove from yaml or build out
                {"name": "OCI__LIFE_SPAN", "value": ""},  # TODO: for testing later
                {"name": "OCI__WORK_DIR", "value": work_dir},
            ]
        )

        new_env_vars = self._add_framework_envs(new_env_vars)
        return update_env_vars(self.config, new_env_vars)

    def _add_framework_envs(self, new_env_vars: List):
        # Overwrite this function in your subclass in order to add/subtract environment variables
        return new_env_vars


def update_env_vars(config, env_vars: List):
    """
    env_vars: List, should be formatted as [{"name": "OCI__XXX", "value": YYY},]
    """
    # TODO move this to a class which checks the version, kind, type, etc.
    config["spec"]["Runtime"]["spec"]["environmentVariables"].extend(env_vars)
    return config
