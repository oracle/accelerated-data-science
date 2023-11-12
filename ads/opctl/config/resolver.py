#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import base64
import json
import os
from typing import Dict
from typing import Tuple

import yaml
import glob

from ads.common.auth import create_signer
from ads.opctl import logger
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.utils import NotSupportedError, convert_notebook
from ads.opctl.constants import (
    ML_JOB_GPU_IMAGE,
    ML_JOB_IMAGE,
    BACKEND_NAME,
)
from ads.opctl.utils import (
    list_ads_operators,
    parse_conda_uri,
    get_region_key,
    get_namespace,
)
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
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
        self.ads_operators = list_ads_operators()

    def process(self):
        # this function should be run after merging configs
        # conda pack scenarios --
        # - user runs their own scripts and commands -> source_folder + entrypoint + (command)
        # - user runs ADS operator -> name/YAML

        # docker image scenarios --
        # - user runs their own docker image -> image (to run) + (entrypoint) + (command)
        # - user runs ADS operator -> name/YAML

        if self.config["execution"].get("job_id"):
            return self

        if self.config.get("kind") == "pipeline":
            # For pipelines, properties like the conda slug and source folder don't apply to the entire pipeline,
            # so we can skip all this validation. Each individual step will have its own config with the required
            # values set.
            return self

        self._resolve_operator_name()

        if not (
            self._is_ads_operator()
            or self.config["execution"].get("source_folder")
            or self.config["execution"].get("image")
        ):
            raise ValueError(
                "Either an ADS operator or a source folder or a docker image needs to be given."
            )

        self._resolve_conda()
        if not (
            self.config["execution"].get("conda_slug")
            or self.config["execution"].get("image")
            or self.config["execution"]["backend"] == BACKEND_NAME.DATAFLOW.value
        ):
            raise ValueError(
                "If not running an operator, conda pack info or image name needs to be given."
            )
        if self.config["execution"].get("conda_slug") and self.config["execution"].get(
            "image"
        ):
            raise ValueError(
                "Both conda pack info and image name are given. It is ambiguous which one to use."
            )

        self._resolve_source_folder_path()
        self._resolve_entry_script()
        self._resolve_command()
        self._resolve_image_name()
        self._resolve_env_vars()
        self._resolve_mounted_volumes()
        self._resolve_job_name()

        return self

    def _is_ads_operator(self) -> bool:
        return self.config["execution"].get("operator_name") in self.ads_operators

    def _resolve_operator_name(self) -> None:
        self.config["execution"]["operator_name"] = self.config["execution"].get(
            "operator_name"
        ) or self.config.get("name")

    @runtime_dependency(module="inflection", install_from=OptionalDependency.OPCTL)
    def _resolve_source_folder_path(self) -> None:
        # this should be run after resolve_operator_name()
        # resolve ADS operator source folder path
        exec_config = self.config["execution"]
        if exec_config.get("source_folder"):
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

    def _resolve_entry_script(self) -> None:
        # this should be run after _resolve_source_folder_path
        if not self._is_ads_operator():
            if (
                self.config["execution"].get("entrypoint")
                and os.path.splitext(self.config["execution"]["entrypoint"])[1]
                == ".ipynb"
            ):
                input_path = os.path.join(
                    self.config["execution"]["source_folder"],
                    self.config["execution"]["entrypoint"],
                )
                exclude_tags = self.config["execution"].get("exclude_tag")
                self.config["execution"]["entrypoint"] = os.path.basename(
                    convert_notebook(
                        input_path,
                        {},
                        exclude_tags,
                        overwrite=self.config["execution"].get("overwrite", False),
                    )
                )

    def _resolve_command(self) -> None:
        # this should be run after resolve_operator_name and resolve_conda
        # resolve command for operators
        exec_config = self.config["execution"]
        if self._is_ads_operator():
            logger.info("Building command for operator runs...")
            operator_name = exec_config.get("operator_name")
            command = f"-n {operator_name} -c {exec_config['oci_config']} -p {exec_config['oci_profile']}"
            if "spec" in self.config:
                encoded_spec = base64.b64encode(
                    json.dumps(self.config["spec"]).encode()
                ).decode()
                command += f" -s {encoded_spec}"
            self.config["execution"]["command"] = command
        else:
            # cmd_args is for running scripts, either locally, with jobs or dataflow; command is for docker command
            self.config["execution"]["command"] = exec_config.get(
                "cmd_args"
            ) or exec_config.get("command")

    def _resolve_conda(self) -> None:
        # this should be run after resolve_operator_name
        exec_config = self.config["execution"]
        # two scenarios that indicate conda pack should be used:
        # - conda slug is found
        # - use-conda is turned on explicitly. in this case for ads operators, look up index.yaml to fill in conda info
        slug = None
        conda_type = None
        if exec_config.get("conda_uri"):
            slug = parse_conda_uri(exec_config["conda_uri"])[-1]
            conda_type = "published"
        elif exec_config.get("conda_slug"):
            slug = exec_config["conda_slug"]
            conda_uri, conda_type = self._get_conda_uri_and_type(slug)
            self.config["execution"]["conda_uri"] = conda_uri
            if conda_type and conda_type == "data_science":
                conda_type = "service"
            conda_type = conda_type or "service"
        elif exec_config.get("use_conda", False):
            operator_name = exec_config.get("operator_name")
            if self._is_ads_operator():
                if self.ads_operators[operator_name].get("conda_slug"):
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

    def _get_conda_uri_and_type(self, slug: str) -> Tuple[str, str]:
        conda_pack_path = os.path.join(
            os.path.expanduser(self.config["execution"]["conda_pack_folder"]), slug
        )
        if os.path.exists(conda_pack_path):
            manifest_path = glob.glob(os.path.join(conda_pack_path, "*_manifest.yaml"))[
                0
            ]
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f.read())["manifest"]
            return manifest.get("pack_uri"), manifest.get("type")
        else:
            return None, None

    def _resolve_image_name(self) -> None:
        # this should be run after resolve_conda() and resolve_command()
        exec_config = self.config["execution"]
        if exec_config.get("conda_slug", None):
            self.config["execution"]["image"] = (
                ML_JOB_GPU_IMAGE if exec_config.get("gpu", False) else ML_JOB_IMAGE
            )
        elif self._is_ads_operator():
            image = self.ads_operators[exec_config["operator_name"]].get("image")
            oci_auth = create_signer(
                self.config["execution"].get("auth"),
                self.config["execution"].get("oci_config"),
                self.config["execution"].get("oci_profile"),
            )
            region_key = get_region_key(oci_auth).lower()
            if exec_config.get("namespace"):
                namespace = exec_config["namespace"]
            else:
                namespace = get_namespace(oci_auth)
            if self.config.get("infrastructure", {}).get("docker_registry"):
                self.config["execution"]["image"] = os.path.join(
                    self.config["infrastructure"]["docker_registry"], image
                )
            else:
                self.config["execution"][
                    "image"
                ] = f"{region_key}.ocir.io/{namespace}/{image}"

    def _resolve_env_vars(self) -> None:
        env_vars = self.config["execution"].get("env_vars", {})
        if self.config["execution"].get("env_var"):
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

    def _resolve_mounted_volumes(self) -> None:
        volumes = self.config["execution"].get("volumes", {})
        if self.config["execution"].get("volume"):
            for v in self.config["execution"]["volume"]:
                host, remote = v.split(":")
                if not os.path.exists(host.strip()):
                    raise FileNotFoundError(f"{host} does not exist")
                volumes[os.path.abspath(host.strip())] = {"bind": remote.strip()}
        self.config["execution"]["volumes"] = volumes

    def _resolve_job_name(self) -> None:
        # this should be run after resolve_operator_name
        if not self.config["execution"].get("job_name"):
            if self.config["execution"].get("operator_name"):
                self.config["execution"]["job_name"] = self.config["execution"][
                    "operator_name"
                ]
