#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import tempfile
from typing import List, Dict
import shlex

from docker import errors

from ads.opctl import logger
from ads.opctl.backend.base import Backend
from ads.opctl.conda.cmds import _install
from ads.opctl.config.resolver import ConfigResolver
from ads.opctl.constants import (
    ML_JOB_IMAGE,
    ML_JOB_GPU_IMAGE,
    OPS_IMAGE_BASE,
    OPS_IMAGE_GPU_BASE,
    DEFAULT_IMAGE_HOME_DIR,
    DEFAULT_IMAGE_SCRIPT_DIR,
)
from ads.opctl.utils import _get_docker_client
from ads.opctl.utils import build_image, _run_command, _run_container


class CondaPackNotFound(Exception):
    pass


class LocalBackend(Backend):
    def __init__(self, config: Dict) -> None:
        """
        Initialize a LocalBackend object with given config.

        Parameters
        ----------
        config: dict
            dictionary of configurations
        """
        self.config = config

    def run(self):
        bind_volumes = {
            os.path.expanduser(
                os.path.dirname(self.config["execution"]["oci_config"])
            ): {"bind": os.path.join(DEFAULT_IMAGE_HOME_DIR, ".oci")}
        }
        if self.config["execution"].get("conda_slug", None):
            self._run_with_conda_pack(bind_volumes)
        elif self.config["execution"].get("image", None):
            self._run_with_image(bind_volumes)
        else:
            raise ValueError("Either conda pack info or image should be specified.")
        return {}

    def init_vscode_container(self) -> None:
        """
        Create a .devcontainer.json file for development with VSCode.

        Returns
        -------
        None
        """
        source_folder = self.config["execution"].get("source_folder", None)
        if source_folder:
            source_folder = os.path.abspath(source_folder)
            if not os.path.exists(source_folder):
                raise FileNotFoundError(
                    f"source folder {source_folder} does not exist."
                )
            if self.config["execution"].get("gpu", False):
                image = self.config["execution"].get("image", ML_JOB_GPU_IMAGE)
            else:
                image = self.config["execution"].get("image", ML_JOB_IMAGE)
            oci_config_folder = os.path.expanduser(
                os.path.dirname(self.config["execution"]["oci_config"])
            )
            dev_container = {
                "image": image,
                "extensions": ["ms-python.python"],
                "mounts": [
                    f"source={oci_config_folder},target={os.path.join(DEFAULT_IMAGE_HOME_DIR, '.oci')},type=bind",
                ],
                "workspaceMount": f"source={source_folder},target={os.path.join(DEFAULT_IMAGE_SCRIPT_DIR, 'operators', os.path.basename(source_folder))},type=bind",
                "workspaceFolder": os.path.join(DEFAULT_IMAGE_SCRIPT_DIR, "operators"),
                "name": f"{image}-dev-env",
            }
            if image == ML_JOB_IMAGE or image == ML_JOB_GPU_IMAGE:
                conda_folder = os.path.expanduser(
                    self.config["execution"]["conda_pack_folder"]
                )
                dev_container["mounts"].append(
                    f"source={conda_folder},target=/opt/conda/envs/,type=bind"
                )
                dev_container[
                    "postCreateCommand"
                ] = "conda init bash && source ~/.bashrc"
            else:
                if not (image == OPS_IMAGE_BASE or image == OPS_IMAGE_GPU_BASE):
                    raise ValueError(
                        "`--source-folder` option works with image `ml-job`, `ml-job-gpu`, `ads-operators-base`, `ads-operators-gpu-base` only. Those can be build with `ads opctl build-image`. Please check `ads opctl build-image -h`."
                    )
        else:
            image = self.config["execution"].get("image", None)
            if not image:
                raise ValueError("Image must be specified.")
            else:
                dev_container = {
                    "image": image,
                    "mounts": [],
                    "extensions": ["ms-python.python"],
                }

        dev_container["containerEnv"] = self.config["execution"]["env_vars"]
        for k, v in self.config["execution"]["volumes"].items():
            dev_container["mounts"].append(
                f"source={os.path.abspath(k)},target={v['bind']},type=bind"
            )

        try:
            client = _get_docker_client()
            client.api.inspect_image(image)
        except errors.ImageNotFound:
            cmd = None
            if image == ML_JOB_IMAGE:
                cmd = "ads opctl build-image job-local"
            elif image == ML_JOB_GPU_IMAGE:
                cmd = "ads opctl build-image job-local -g"
            elif image == OPS_IMAGE_BASE:
                cmd = "ads opctl build-image ads-ops-base"
            elif image == OPS_IMAGE_GPU_BASE:
                cmd = "ads opctl build-image ads-ops-base -g"
            if cmd:
                raise RuntimeError(
                    f"Image {image} not found. Please run `{cmd}` to build the image."
                )
            else:
                raise RuntimeError(f"Image {image} not found.")
        if source_folder:
            with open(
                os.path.join(os.path.abspath(source_folder), ".devcontainer.json"), "w"
            ) as f:
                f.write(json.dumps(dev_container, indent=2))
            print(f"File {os.path.join(source_folder, '.devcontainer.json')} created.")
        else:
            with open(os.path.abspath(".devcontainer.json"), "w") as f:
                f.write(json.dumps(dev_container, indent=2))
            print(f"File {os.path.abspath('.devcontainer.json')} created.")

    def _run_with_conda_pack(self, bind_volumes: Dict) -> None:
        env_vars = self.config["execution"]["env_vars"]
        slug = self.config["execution"]["conda_slug"]
        image = self.config["execution"]["image"]
        # bind_volumes is modified in-place and does not need to be returned
        # it is returned just to be explicit that it is changed during this function call
        bind_volumes = self._check_conda_pack_and_install_if_applicable(
            slug, bind_volumes
        )
        bind_volumes = self._mount_source_folder_if_exists(bind_volumes)

        if ConfigResolver(self.config)._is_operator():
            # running operators
            command = (
                f"python {os.path.join(DEFAULT_IMAGE_SCRIPT_DIR, 'operators/run.py')} "
            )
        else:
            # running a user script
            entry_script = self.config["execution"].get("entrypoint", None)
            if not entry_script:
                raise ValueError(
                    "An entrypoint script must be specified when running with conda pack. "
                    "Use `--entrypoint`."
                )
            mount_path = os.path.join(
                DEFAULT_IMAGE_SCRIPT_DIR,
                "operators",
                os.path.basename(self.config["execution"]["source_folder"]),
            )
            if os.path.splitext(entry_script)[-1] == ".py":
                command = f"python {os.path.join(mount_path, entry_script)} "
            elif os.path.splitext(entry_script)[-1] == ".sh":
                command = f"cd {mount_path} && /bin/bash {entry_script} "
            else:
                logger.warn(
                    "ML Job only support .py and .sh files."
                    "If you intend to submit to ML Job later, please update file extension."
                )
                command = f"cd {mount_path} && {entry_script} "
        # append args
        if self.config["execution"].get("command", None):
            command += f"{self.config['execution']['command']}"
        self._activate_conda_env_and_run(image, slug, command, bind_volumes, env_vars)

    def _run_with_image(self, bind_volumes: Dict) -> None:
        env_vars = self.config["execution"]["env_vars"]
        image = self.config["execution"]["image"]
        if ConfigResolver(self.config)._is_operator():
            # running operators
            command = (
                f"python {os.path.join(DEFAULT_IMAGE_SCRIPT_DIR, 'operators/run.py')} "
            )
            entrypoint = None
            if self.config["execution"].get("command", None):
                command += f"{self.config['execution']['command']}"
        else:
            # in case of running a user image, entrypoint is not required
            entrypoint = self.config["execution"].get("entrypoint", None)
            command = self.config["execution"].get("command", None)
        if self.config["execution"].get("source_folder", None):
            bind_volumes.update(self._mount_source_folder_if_exists(bind_volumes))
        bind_volumes.update(self.config["execution"]["volumes"])
        _run_container(image, bind_volumes, env_vars, command, entrypoint)

    def _check_conda_pack_and_install_if_applicable(
        self, slug: str, bind_volumes: Dict
    ) -> Dict:
        conda_pack_path = os.path.join(
            os.path.expanduser(self.config["execution"]["conda_pack_folder"]), slug
        )
        if not os.path.exists(conda_pack_path):
            if self.config["execution"].get("conda_uri", None):
                _install(
                    self.config["execution"]["conda_uri"],
                    self.config["execution"]["conda_pack_folder"],
                    self.config["execution"]["oci_config"],
                    self.config["execution"]["oci_profile"],
                )
            else:
                raise CondaPackNotFound(
                    f"Conda pack {conda_pack_path} not found. Please run `ads opctl conda create` or `ads opctl conda install`."
                )
        bind_volumes[
            os.path.abspath(
                os.path.expanduser(self.config["execution"]["conda_pack_folder"])
            )
        ] = {"bind": os.path.join(DEFAULT_IMAGE_HOME_DIR, "conda")}
        return bind_volumes

    def _mount_source_folder_if_exists(self, bind_volumes: Dict) -> Dict:
        source_folder = os.path.abspath(self.config["execution"]["source_folder"])
        if not os.path.exists(source_folder):
            raise FileNotFoundError(f"source folder {source_folder} does not exist.")
        mount_path = os.path.join(
            DEFAULT_IMAGE_SCRIPT_DIR,
            "operators",
            os.path.basename(self.config["execution"]["source_folder"]),
        )
        bind_volumes[source_folder] = {"bind": mount_path}
        return bind_volumes

    @staticmethod
    def _activate_conda_env_and_run(
        image: str, slug: str, command: List[str], bind_volumes: Dict, env_vars: Dict
    ) -> None:
        try:
            client = _get_docker_client()
            client.api.inspect_image(image)
        except errors.ImageNotFound:
            logger.info(f"Image {image} not found. Attempt building it now....")
            if image == ML_JOB_IMAGE:
                build_image("job-local", gpu=False)
            else:
                build_image("job-local", gpu=True)

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "entryscript.sh"), "w") as f:
                f.write(
                    f"""
#!/bin/bash
source {os.path.join(DEFAULT_IMAGE_HOME_DIR, 'conda', slug, 'bin/activate')}
{command}
                        """
                )
            bind_volumes[os.path.join(td, "entryscript.sh")] = {
                "bind": os.path.join(DEFAULT_IMAGE_SCRIPT_DIR, "entryscript.sh")
            }
            env_vars["conda_slug"] = slug
            _run_container(
                image,
                bind_volumes,
                env_vars,
                command=f"bash {os.path.join(DEFAULT_IMAGE_SCRIPT_DIR, 'entryscript.sh')}",
            )
            # Bad Request ("OCI runtime create failed: container_linux.go:380:
            # starting container process caused: exec: "source": executable file not found in $PATH: unknown")
            # command=["source", f"/home/datascience/conda/{slug}/bin/activate", "&&", command]
