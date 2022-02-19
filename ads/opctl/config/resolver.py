#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import base64
import json
import os
from typing import Dict
from typing import Tuple

import inflection

from ads.common.auth import get_signer
from ads.opctl import logger
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.utils import NotSupportedError
from ads.opctl.constants import (
    ML_JOB_GPU_IMAGE,
    ML_JOB_IMAGE,
    OPS_IMAGE_BASE,
    OPS_IMAGE_GPU_BASE,
)
from ads.opctl.utils import (
    _list_ads_operators,
    _parse_conda_uri,
    _get_region_key,
    _get_namespace,
)


class ConfigResolver(ConfigProcessor):
    """Infer and fill in fields necessary for running an operator based on user inputs.
    For instance, if user pass in an ADS operator name, we will fill in image name, command,
    volumes to mount for local run, source folder path, etc.

    Please check functions with name `_resolve_<thing>`.

    List of things being resolved (and brief description):
    - operator name (could be passed in directly or in YAML)
    - conda (type is determined on whether a slug or uri is passed in. slug -> service, uri -> published. also do some additional checks.)
    - source folder (mainly for ADS operator)
    - command (docker command)
    - image name (mainly for ADS operator)
    - env vars
    - mount volumes (for local runs)

    There are additional comments in some of the functions below.
    """

    def __init__(self, config: Dict = None) -> None:
        super().__init__(config)
        self.ads_operators = _list_ads_operators()

    def process(self):
        # this function should be run after merging configs
        # conda pack scenarios --
        # - user runs their own scripts and commands -> source_folder + entry script + (command)
        # - user runs ADS operator -> name/YAML
        # - user runs custom operator -> source_folder + YAML
        # docker image scenarios --
        # - user runs their own docker image -> image (to run) + (entry script) + (command)
        # - user runs ADS operator -> name/YAML
        # - user runs custom operator -> (image if remote) + source folder + YAML

        if (
            self.config["execution"].get("job_id")
            and self.config["execution"].get("backend", None) == "job"
        ):
            return self

        self._resolve_operator_name()

        if not (
            self._is_ads_operator()
            or self.config["execution"].get("source_folder", None)
            or self.config["execution"].get("image", None)
        ):
            raise ValueError(
                "Either an ADS operator or a source folder or a docker image needs to be given."
            )

        self._resolve_conda()
        if not (
            self._is_operator()
            or self.config["execution"].get("conda_slug", None)
            or self.config["execution"].get("image", None)
        ):
            raise ValueError(
                "If not running an operator, conda pack info or image name needs to be given."
            )
        if self.config["execution"].get("conda_slug") and self.config["execution"].get(
            "image", None
        ):
            raise ValueError(
                "Both conda pack info and image name are given. It is ambiguous which one to use."
            )

        self._resolve_source_folder_path()
        self._resolve_command()
        self._resolve_image_name()
        self._resolve_env_vars()
        self._reslve_mounted_volumes()
        self._resolve_job_name()

        logger.debug(f"Config: {self.config}")

        return self

    def _is_operator(self) -> bool:
        return (
            self.config["execution"].get("operator_name", None)
            or self.config.get("kind", None) == "MLOperator"
        )

    def _is_ads_operator(self) -> bool:
        return self.config["execution"].get("operator_name", None) in self.ads_operators

    def _resolve_operator_name(self) -> str:
        exec_config = self.config["execution"]
        operator_name = exec_config.get("operator_name", None) or self.config.get(
            "name", None
        )
        self.config["execution"]["operator_name"] = operator_name

    def _resolve_source_folder_path(self) -> str:
        # this should be run after resolve_operator_name()
        # resolve ADS operator source folder path
        exec_config = self.config["execution"]
        if exec_config.get("source_folder", None):
            self.config["execution"]["source_folder"] = os.path.abspath(
                os.path.normpath(exec_config["source_folder"])
            )
            if not os.path.exists(self.config["execution"]["source_folder"]):
                raise FileNotFoundError(
                    f"{self.config['execution']['source_folder']} is not found."
                )
        else:
            if self._is_ads_operator():
                curr_dir = os.path.dirname(os.path.abspath(__file__))
                self.config["execution"]["source_folder"] = os.path.normpath(
                    os.path.join(
                        curr_dir,
                        "..",
                        "operators",
                        inflection.underscore(
                            self.config["execution"]["operator_name"]
                        ),
                    )
                )
            else:
                self.config["execution"]["source_folder"] = None

    def _resolve_command(self) -> str:
        # this should be run after resolve_operator_name and resolve_conda
        # resolve command for operators (ADS or custom)
        exec_config = self.config["execution"]
        if exec_config.get("entrypoint", None) and exec_config.get("conda_slug", None):
            # do not touch in case of running a user script with conda pack
            self.config["execution"]["command"] = exec_config.get("cmd_args", None)
        elif not self._is_operator() and exec_config.get("image", None):
            # do not touch case of running arbitrary image
            self.config["execution"]["command"] = exec_config.get("command", None)
        else:

            # the rest scenarios are running operators
            logger.info("Building command for operator runs...")
            operator_name = exec_config.get("operator_name", None) or os.path.basename(
                exec_config["source_folder"]
            )
            command = f"-n {operator_name} -c {exec_config['oci_config']} -p {exec_config['oci_profile']}"
            if "spec" in self.config:
                encoded_spec = base64.b64encode(
                    json.dumps(self.config["spec"]).encode()
                ).decode()
                command += f" -s {encoded_spec}"
            self.config["execution"]["command"] = command

    def _resolve_conda(self) -> Tuple[str, str]:
        # this should be run after resolve_operator_name
        exec_config = self.config["execution"]
        # two scenarios that indicate conda pack should be used:
        # - conda slug is found
        # - use-conda is turned on explicitly. in this case for ads operators, look up index.yaml to fill in conda info
        slug = None
        conda_type = None
        if exec_config.get("conda_uri", None):
            slug = _parse_conda_uri(exec_config["conda_uri"])[-1]
            conda_type = "published"
        elif exec_config.get("conda_slug", None):
            slug = exec_config["conda_slug"]
            conda_type = "service"
        elif exec_config.get("use_conda", False):
            operator_name = exec_config.get("operator_name", None)
            if self._is_ads_operator():
                if self.ads_operators[operator_name].get("conda_slug", None):
                    slug = self.ads_operators[operator_name]["conda_slug"]
                    conda_type = "service"
                else:
                    raise NotSupportedError(
                        f"Conda pack is not supported for ADS operator {operator_name}."
                    )
        else:
            logger.info("Conda pack slug info is not found.")
        self.config["execution"]["conda_slug"] = slug
        self.config["execution"]["conda_type"] = conda_type

    def _resolve_image_name(self) -> str:
        # this should be run after resolve_conda() and resolve_command()
        exec_config = self.config["execution"]
        if exec_config.get("conda_slug", None):
            self.config["execution"]["image"] = (
                ML_JOB_GPU_IMAGE if exec_config.get("gpu", False) else ML_JOB_IMAGE
            )
        elif self._is_ads_operator():
            image = self.ads_operators[exec_config["operator_name"]].get("image", None)
            oci_auth = get_signer(
                self.config["execution"].get("oci_config", None),
                self.config["execution"].get("oci_profile", None),
            )
            region_key = _get_region_key(oci_auth).lower()
            if exec_config.get("namespace", None):
                namespace = exec_config["namespace"]
            else:
                namespace = _get_namespace(oci_auth)
            if self.config.get("infrastructure", {}).get("docker_registry", None):
                self.config["execution"]["image"] = os.path.join(
                    self.config["infrastructure"]["docker_registry"], image
                )
            else:
                self.config["execution"][
                    "image"
                ] = f"{region_key}.ocir.io/{namespace}/{image}"
        elif (
            self._is_operator()
            and self.config["execution"].get("backend", None) == "local"
        ):
            self.config["execution"]["image"] = (
                OPS_IMAGE_GPU_BASE if exec_config.get("gpu", False) else OPS_IMAGE_BASE
            )

    def _resolve_env_vars(self) -> dict:
        env_vars = self.config["execution"].get("env_vars", {})
        if self.config["execution"].get("env_var", None):
            for ev in self.config["execution"]["env_var"]:
                if "=" in ev:
                    k, v = ev.split("=")
                    env_vars[k.strip()] = v.strip()
                elif ev in os.environ:
                    env_vars[ev] = os.environ[ev]
                else:
                    raise ValueError(
                        "Please provide environment variable as VAR=value or "
                        "give an environment variable that is set locally."
                    )
        self.config["execution"]["env_vars"] = env_vars

    def _reslve_mounted_volumes(self) -> dict:
        volumes = self.config["execution"].get("volumes", {})
        if self.config["execution"].get("volume", None):
            for v in self.config["execution"]["volume"]:
                host, remote = v.split(":")
                if not os.path.exists(host.strip()):
                    raise FileNotFoundError(f"{host} does not exist")
                volumes[os.path.abspath(host.strip())] = {"bind": remote.strip()}
        self.config["execution"]["volumes"] = volumes

    def _resolve_job_name(self) -> str:
        # this should be run after resolve_operator_name
        if not self.config["execution"].get("job_name", None):
            if self.config["execution"].get("operator_name", None):
                self.config["execution"]["job_name"] = self.config["execution"][
                    "operator_name"
                ]
