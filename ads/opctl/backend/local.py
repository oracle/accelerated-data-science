#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import json
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from time import sleep
from typing import List, Dict

from ads.opctl import logger
from ads.opctl.backend.base import Backend
from ads.opctl.config.resolver import ConfigResolver
from ads.opctl.distributed.cmds import local_run, load_ini
from ads.opctl.constants import (
    ML_JOB_IMAGE,
    ML_JOB_GPU_IMAGE,
    DEFAULT_IMAGE_HOME_DIR,
    DEFAULT_IMAGE_SCRIPT_DIR,
    DEFAULT_IMAGE_CONDA_DIR,
    DEFAULT_NOTEBOOK_SESSION_CONDA_DIR,
    DEFAULT_NOTEBOOK_SESSION_SPARK_CONF_DIR,
)
from ads.opctl.utils import get_docker_client, is_in_notebook_session
from ads.opctl.utils import build_image, run_container, run_command
from ads.opctl.spark.cmds import (
    generate_core_site_properties_str,
    generate_core_site_properties,
)
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.pipeline.ads_pipeline import Pipeline, PipelineStep
from oci.data_science.models import PipelineStepRun


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
        if self.config.get("version") == "v1.0":
            docker_image = self.config["spec"]["Infrastructure"]["spec"]["dockerImage"]
            # TODO: don't hard code api keys
            bind_volumes = {
                os.path.expanduser("~/.oci"): {
                    "bind": os.path.join(DEFAULT_IMAGE_HOME_DIR, ".oci"),
                    "mode": "ro",
                }
            }
            self._run_with_image_v1(bind_volumes)

        else:
            bind_volumes = {}
            if not is_in_notebook_session():
                bind_volumes = {
                    os.path.expanduser(
                        os.path.dirname(self.config["execution"]["oci_config"])
                    ): {"bind": os.path.join(DEFAULT_IMAGE_HOME_DIR, ".oci")}
                }
            if self.config["execution"].get("conda_slug", None):
                exit_code = self._run_with_conda_pack(bind_volumes)
            elif self.config["execution"].get("image"):
                exit_code = self._run_with_image(bind_volumes)
            else:
                raise ValueError("Either conda pack info or image should be specified.")

            if exit_code != 0:
                raise RuntimeError(
                    f"Job did not complete successfully. Exit code: {exit_code}. "
                    f"Run with the --debug argument to view container logs."
                )

    @runtime_dependency(module="docker", install_from=OptionalDependency.OPCTL)
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
                "workspaceMount": f"source={source_folder},target={os.path.join(DEFAULT_IMAGE_HOME_DIR, os.path.basename(source_folder))},type=bind",
                "workspaceFolder": DEFAULT_IMAGE_HOME_DIR,
                "name": f"{image}-dev-env",
            }
            if image == ML_JOB_IMAGE or image == ML_JOB_GPU_IMAGE:
                conda_folder = os.path.expanduser(
                    self.config["execution"]["conda_pack_folder"]
                )
                dev_container["mounts"].append(
                    f"source={conda_folder},target={DEFAULT_IMAGE_CONDA_DIR},type=bind"
                )
                dev_container[
                    "postCreateCommand"
                ] = "conda init bash && source ~/.bashrc"
            else:
                raise ValueError(
                    "`--source-folder` option works with image `ml-job`, `ml-job-gpu` only. Those can be build with `ads opctl build-image`. Please check `ads opctl build-image -h`."
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
            client = get_docker_client()
            client.api.inspect_image(image)
        except docker.errors.ImageNotFound:
            cmd = None
            if image == ML_JOB_IMAGE:
                cmd = "ads opctl build-image job-local"
            elif image == ML_JOB_GPU_IMAGE:
                cmd = "ads opctl build-image job-local -g"
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

    def _run_with_conda_pack(self, bind_volumes: Dict) -> int:
        env_vars = self.config["execution"]["env_vars"]
        slug = self.config["execution"]["conda_slug"]
        image = self.config["execution"]["image"]

        # bind_volumes is modified in-place and does not need to be returned
        # it is returned just to be explicit that it is changed during this function call
        bind_volumes, env_vars = self._check_conda_pack_and_install_if_applicable(
            slug, bind_volumes, env_vars
        )
        bind_volumes = self._mount_source_folder_if_exists(bind_volumes)
        command = self._build_command_for_conda_run()
        if is_in_notebook_session():
            run_command(command, shell=True)
        else:
            conda_pack_path = os.path.join(
                os.path.expanduser(self.config["execution"]["conda_pack_folder"]), slug
            )
            if os.path.exists(os.path.join(conda_pack_path, "spark-defaults.conf")):
                env_vars["SPARK_CONF_DIR"] = os.path.join(DEFAULT_IMAGE_CONDA_DIR, slug)
            return self._activate_conda_env_and_run(
                image, slug, command, bind_volumes, env_vars
            )

    def _build_command_for_conda_run(self) -> str:
        if ConfigResolver(self.config)._is_ads_operator():
            if is_in_notebook_session():
                curr_dir = os.path.dirname(os.path.abspath(__file__))
                script = os.path.abspath(
                    os.path.join(curr_dir, "..", "operators", "run.py")
                )
            else:
                script = os.path.join(DEFAULT_IMAGE_SCRIPT_DIR, "operators/run.py")
            command = f"python {script} "
            if self.config["execution"]["auth"] == "resource_principal":
                command += "-r "
        else:
            entry_script = self.config["execution"].get("entrypoint")
            if not entry_script:
                raise ValueError(
                    "An entrypoint script must be specified when running with conda pack. "
                    "Use `--entrypoint`."
                )
            if not os.path.exists(
                os.path.join(self.config["execution"]["source_folder"], entry_script)
            ):
                raise FileNotFoundError(
                    f"{os.path.join(self.config['execution']['source_folder'], entry_script)} is not found."
                )
            if is_in_notebook_session():
                source_folder = os.path.join(self.config["execution"]["source_folder"])
            else:
                source_folder = os.path.join(
                    DEFAULT_IMAGE_SCRIPT_DIR,
                    "operators",
                    os.path.basename(self.config["execution"]["source_folder"]),
                )
            if os.path.splitext(entry_script)[-1] == ".py":
                command = f"python {os.path.join(source_folder, entry_script)} "
                if is_in_notebook_session():
                    command = (
                        f"source activate {os.path.join(DEFAULT_NOTEBOOK_SESSION_CONDA_DIR, self.config['execution']['conda_slug'])} && "
                        + command
                    )
            elif os.path.splitext(entry_script)[-1] == ".sh":
                command = f"cd {source_folder} && /bin/bash {entry_script} "
            else:
                logger.warn(
                    "ML Job only support .py and .sh files."
                    "If you intend to submit to ML Job later, please update file extension."
                )
                command = f"cd {source_folder} && {entry_script} "
        if self.config["execution"].get("command"):
            command += f"{self.config['execution']['command']}"
        return command

    def _run_with_image(self, bind_volumes: Dict) -> int:
        env_vars = self.config["execution"]["env_vars"]
        image = self.config["execution"]["image"]
        if ConfigResolver(self.config)._is_ads_operator():
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
        return run_container(image, bind_volumes, env_vars, command, entrypoint)

    def _run_with_image_v1(self, bind_volumes: Dict) -> int:
        env_vars = [
            str(d["name"]) + "=" + str(d["value"])
            for d in self.config["spec"]["Runtime"]["spec"]["environmentVariables"]
        ]
        image = self.config["spec"]["Infrastructure"]["spec"]["dockerImage"]
        command = self.config["spec"]["Runtime"]["spec"]["entrypoint"]
        entrypoint = "python /etc/datascience/operator/cluster_helper.py"

        print("looking to bind volume")
        bind_volumes.update(self.config["spec"]["Framework"]["spec"]["bindVolumes"])
        return run_container(
            image=image,
            bind_volumes=bind_volumes,
            env_vars=env_vars,
            command=command,
            entrypoint=entrypoint,
        )

    def _check_conda_pack_and_install_if_applicable(
        self, slug: str, bind_volumes: Dict, env_vars: Dict
    ) -> Dict:
        conda_pack_path = os.path.join(
            os.path.expanduser(self.config["execution"]["conda_pack_folder"]), slug
        )
        if not os.path.exists(conda_pack_path):
            raise CondaPackNotFound(
                f"Conda pack {conda_pack_path} not found. Please run `ads opctl conda create` or `ads opctl conda install`."
            )
        if os.path.exists(os.path.join(conda_pack_path, "spark-defaults.conf")):
            if not is_in_notebook_session():
                env_vars["SPARK_CONF_DIR"] = os.path.join(DEFAULT_IMAGE_CONDA_DIR, slug)
            # write core_site.xml
            if self.config["execution"]["auth"] == "api_key":
                properties = generate_core_site_properties(
                    "api_key",
                    self.config["execution"]["oci_config"],
                    self.config["execution"]["oci_profile"],
                )
                # key path cannot have "~/"
                oci_config_folder = os.path.expanduser(
                    os.path.dirname(self.config["execution"]["oci_config"])
                )
                properties[-1] = (
                    properties[-1][0],
                    os.path.join(
                        DEFAULT_IMAGE_HOME_DIR,
                        ".oci",
                        os.path.relpath(
                            os.path.expanduser(properties[-1][1]), oci_config_folder
                        ),
                    ),
                )
            else:
                properties = generate_core_site_properties("resource_principal")

            core_site_str = generate_core_site_properties_str(properties)
            if is_in_notebook_session():
                with open(
                    os.path.join(
                        DEFAULT_NOTEBOOK_SESSION_SPARK_CONF_DIR, "core-site.xml"
                    ),
                    "w",
                ) as f:
                    f.write(core_site_str)
            else:
                with open(os.path.join(conda_pack_path, "core-site.xml"), "w") as f:
                    f.write(core_site_str)
        bind_volumes[
            os.path.abspath(
                os.path.expanduser(
                    os.path.join(self.config["execution"]["conda_pack_folder"], slug)
                )
            )
        ] = {"bind": os.path.join(DEFAULT_IMAGE_CONDA_DIR, slug)}
        return bind_volumes, env_vars

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
    @runtime_dependency(module="docker", install_from=OptionalDependency.OPCTL)
    def _activate_conda_env_and_run(
        image: str, slug: str, command: List[str], bind_volumes: Dict, env_vars: Dict
    ) -> int:
        try:
            client = get_docker_client()
            client.api.inspect_image(image)
        except docker.errors.ImageNotFound:
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
source {os.path.join(DEFAULT_IMAGE_CONDA_DIR, slug, 'bin/activate')}
{command}
                        """
                )
            bind_volumes[os.path.join(td, "entryscript.sh")] = {
                "bind": os.path.join(DEFAULT_IMAGE_SCRIPT_DIR, "entryscript.sh")
            }
            env_vars["conda_slug"] = slug
            return run_container(
                image,
                bind_volumes,
                env_vars,
                command=f"bash {os.path.join(DEFAULT_IMAGE_SCRIPT_DIR, 'entryscript.sh')}",
            )


class LocalBackendDistributed(LocalBackend):
    def __init__(self, config: Dict) -> None:
        """
        Initialize a LocalBackendDistributed object with given config. This serves local single node(docker) testing
        for Distributed Tranining


        Parameters
        ----------
        config: dict
            dictionary of configurations
        """
        self.config = config

    def run(self):
        local_run(self.config, load_ini())


class LocalPipelineBackend(Backend):
    LOG_PREFIX = "Local Pipeline:"
    DEFAULT_PARALLEL_CONTAINER_MAXIMUM = 4
    DEFAULT_STATUS_POLL_INTERVAL_SECONDS = 5

    def __init__(self, config: Dict) -> None:
        """
        Initialize a LocalPipelineBackend object with given config.

        Parameters
        ----------
        config: dict
            dictionary of configurations
        """
        self.config = config

    def run(self) -> None:
        pipeline = Pipeline.from_dict(self.config)

        self._log_orchestration_message(f"Starting pipeline {pipeline.name} locally.")
        if pipeline.dag:
            self._log_orchestration_message(f"Pipeline DAG:")
            for d in pipeline.dag:
                self._log_orchestration_message(f"  {d}")

        completed_status = {}
        waiting_steps = {}
        for s in pipeline.step_details:
            waiting_steps[s.name] = s
        futures = {}
        pipeline_failure = False
        done = False

        if "max_parallel_containers" in self.config["infrastructure"]:
            max_parallel_containers = int(
                self.config["infrastructure"]["max_parallel_containers"]
            )
        else:
            max_parallel_containers = min(
                self.DEFAULT_PARALLEL_CONTAINER_MAXIMUM, os.cpu_count()
            )
            logger.warn(
                f"max_parallel_containers not specified in the config. Defaulting to {max_parallel_containers}."
                " Run `ads opctl configure` to define your local backend config."
            )

        poll_interval_seconds = int(
            self.config["infrastructure"].get(
                "pipeline_status_poll_interval_seconds",
                self.DEFAULT_STATUS_POLL_INTERVAL_SECONDS,
            )
        )

        with ThreadPoolExecutor(max_workers=max_parallel_containers) as executor:
            while not done:
                # Check if any running steps have completed
                for s in list(futures):
                    if futures[s].done():
                        if futures[s].exception() is None:
                            self._log_orchestration_message(
                                f"Step {s} completed successfully."
                            )
                            completed_status[
                                s
                            ] = PipelineStepRun.LIFECYCLE_STATE_SUCCEEDED
                        else:
                            pipeline_failure = True
                            self._log_orchestration_message(f"Step {s} failed:")
                            logger.error(futures[s].exception())
                            completed_status[s] = PipelineStepRun.LIFECYCLE_STATE_FAILED
                        del futures[s]

                for s in list(waiting_steps):
                    # Cancel all waiting steps if a failure is encountered
                    if pipeline_failure:
                        self._log_orchestration_message(
                            f"Skipping step {s} - pipeline failure encountered."
                        )
                        completed_status[s] = PipelineStepRun.LIFECYCLE_STATE_SKIPPED
                        del waiting_steps[s]
                        continue

                    # Start a waiting step if all of its dependencies have completed successfully
                    completed_deps = [
                        dep
                        for dep in waiting_steps[s].depends_on
                        if dep in completed_status
                    ]
                    if len(waiting_steps[s].depends_on) == len(completed_deps):
                        self._log_orchestration_message(f"Starting step {s}")
                        futures[s] = self._start_pipeline_step(
                            waiting_steps[s], executor
                        )
                        del waiting_steps[s]

                if len(completed_status) == len(pipeline.step_details):
                    done = True
                else:
                    sleep(poll_interval_seconds)

        self._log_orchestration_message("Pipeline run complete!")
        self._log_orchestration_message("Summary:")
        for step in pipeline.step_details:
            self._log_orchestration_message(
                f"  {step.name} - {completed_status[step.name]}"
            )

    def _start_pipeline_step(
        self, step: PipelineStep, executor: ThreadPoolExecutor
    ) -> Future:
        """
        Starts a single pipeline step.

        Parameters
        ----------
        step: PipelineStep
            The pipeline step to start
        executor: ThreadPoolExecutor
            The executor used to run the pipeline step.

        Returns
        -------
        future: Future
            The Future that can be used to query the status of the pipeline step.
        """
        step_config = self._create_step_config(step)

        # Have a local job backend execute the step using the updated step config
        local_job = LocalBackend(step_config)
        return executor.submit(local_job.run)

    def _create_step_config(self, pipeline_step: PipelineStep) -> Dict:
        """
        Creates the config for local execution of an individual pipeline step.

        Parameters
        ----------
        pipeline_step: PipelineStep
            The pipeline step whose config should be generated

        Returns
        -------
        step_config: Dict
            The config for the individual pipeline step.
        """
        if pipeline_step.kind.upper() != "CUSTOM_SCRIPT":
            raise ValueError(
                f"Step {pipeline_step.name} has unsupported kind. "
                f"Local pipeline execution only supports pipeline steps with kind customScript."
            )

        step_config = copy.deepcopy(self.config)
        step_config["kind"] = pipeline_step.kind
        step_config["type"] = pipeline_step.type
        del step_config["spec"]
        step_execution_config = step_config["execution"]
        step_execution_config["conda_slug"] = pipeline_step.runtime.conda["slug"]
        step_execution_config["env_vars"] = pipeline_step.runtime.envs

        if pipeline_step.runtime.type == "script":
            step_execution_config["entrypoint"] = pipeline_step.runtime.script_uri
        elif pipeline_step.runtime.type == "python":
            step_execution_config["entrypoint"] = pipeline_step.runtime.script_uri
            step_execution_config["source_folder"] = pipeline_step.runtime.working_dir
        elif pipeline_step.runtime.type == "notebook":
            step_execution_config["entrypoint"] = pipeline_step.runtime.notebook_uri
        else:
            raise ValueError(
                f"Step {pipeline_step.name} has unsupported runtime. "
                f"Supported values are: script, python, notebook"
            )

        if not step_execution_config.get("source_folder"):
            logger.warn(
                "No source_folder provided; defaulting to the current working directory. To specify a source"
                "folder for all pipeline steps, use the --source-folder parameter. To specify a source folder"
                "for individual steps, use a runtime with type python and specify the workingDir property."
            )
            step_execution_config["source_folder"] = os.getcwd()

        ConfigResolver(step_config).process()
        return step_config

    def _log_orchestration_message(self, str: str) -> None:
        """
        Logs a message related to pipeline run orchestration

        Parameters
        ----------
        str: str
            The message to log
        """
        logger.info(f"{self.LOG_PREFIX}: {str}")
