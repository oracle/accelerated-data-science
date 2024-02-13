#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.jobs.builders.runtimes.artifact import PythonArtifact, GitPythonArtifact
from ads.jobs.builders.runtimes.base import MultiNodeRuntime
from ads.jobs.builders.runtimes.python_runtime import (
    PythonRuntime,
    GitPythonRuntime,
)


class PyTorchDistributedRuntime(PythonRuntime, MultiNodeRuntime):
    """Represents runtime supporting PyTorch Distributed training."""
    CONST_GIT = "git"
    CONST_INPUT = "inputs"
    CONST_DEP = "dependencies"
    CONST_PIP_REQ = "pipRequirements"
    CONST_PIP_PKG = "pipPackages"
    CONST_COMMAND = "command"
    CONST_DEEPSPEED = "deepspeed"

    def with_git(
        self, url: str, branch: str = None, commit: str = None, secret_ocid: str = None
    ):
        """Specifies the Git repository and branch/commit for the job source code.

        Parameters
        ----------
        url : str
            URL of the Git repository.
        branch : str, optional
            Git branch name, by default None, the default branch will be used.
        commit : str, optional
            Git commit ID (SHA1 hash), by default None, the most recent commit will be used.
        secret_ocid : str
            The secret OCID storing the SSH key content for checking out the Git repository.

        Returns
        -------
        self
            The runtime instance.
        """
        git_spec = {GitPythonRuntime.CONST_GIT_URL: url}
        if branch:
            git_spec[GitPythonRuntime.CONST_BRANCH] = branch
        if commit:
            git_spec[GitPythonRuntime.CONST_COMMIT] = commit
        if secret_ocid:
            git_spec[GitPythonRuntime.CONST_GIT_SSH_SECRET_ID] = secret_ocid
        return self.set_spec(self.CONST_GIT, git_spec)

    @property
    def git(self) -> str:
        """The specification for source code from Git repository."""
        return self.get_spec(self.CONST_GIT)

    def with_inputs(self, mappings: dict):
        """Specifies the input files to be copied into the job run.

        Parameters
        ----------
        mappings : dict
            Each key is the source path (uri). It can be http/ftp link or OCI object storage URI.
            The corresponding value is the destination path in the job run, relative to the working directory.

        Returns
        -------
        self
            The runtime instance.

        Examples
        --------
        >>> pt_runtime.with_inputs({"oci://bucket@namespace/path/to/file.txt": "data/input.txt"})

        """
        return self.set_spec(self.CONST_INPUT, mappings)

    @property
    def inputs(self) -> dict:
        """The input files to be copied into the job run."""
        return self.get_spec(self.CONST_INPUT)

    def with_dependency(self, pip_req=None, pip_pkg=None):
        """Specifies additional dependencies to be installed using pip.

        Parameters
        ----------
        pip_req : str, optional
            Path of the requirements.txt file, relative to the working directory, by default None
        pip_pkg : str, optional
            Command line args for `pip install`, by default None.
            Packages with version specification needs to be quoted.

        Returns
        -------
        self
            The runtime instance.

        Examples
        --------
        >>> pt_runtime.with_dependency('"package>1.0"')
        """
        dep = {}
        if pip_req:
            dep[self.CONST_PIP_REQ] = pip_req
        if pip_pkg:
            dep[self.CONST_PIP_PKG] = pip_pkg
        if dep:
            self.set_spec(self.CONST_DEP, dep)
        return self

    @property
    def dependencies(self) -> dict:
        """Additional pip dependencies."""
        return self.get_spec(self.CONST_DEP)

    def with_command(self, command: str, use_deepspeed=False):
        """Specifies the command for launching the workload.

        Parameters
        ----------
        command : str
            The command for launching the workload.
            The command should start with `torchrun`, `deepspeed` or `accelerate launch`.

            For `torchrun`,
            ADS will set `--nnode`, `--nproc_per_node`, `--rdzv_backend` and `--rdzv_endpoint` automatically.
            The default `rdzv_backend` will be `c10d`.
            The default port for `rdzv_endpoint` is 29400

            For `deepspeed`,
            ADS will generate the hostfile automatically and setup the SSH configurations.

            For `accelerate launch`
            You can add your config YAML to the source code and specify it using `--config_file` argument.
            In your config, please use `LOCAL_MACHINE` as the compute environment.
            The same config file will be used by all nodes in multi-node workload.
            ADS will set `--num_processes`, `--num_machines`, `--machine_rank`, `--main_process_ip`
            and `--main_process_port` automatically. These values will override the ones from your config YAML.
            The default `main_process_port` is 29400

            If you don't want to use the options set by ADS automatically,
            you can specify them explicitly in the command.

        use_deepspeed : bool, optional
            Indicate whether to configure deepspeed for multi-node workload, by default False.
            If your command starts with "deepspeed" or contains the argument "--use_deepspeed",
            your job runs will be configured for deepspeed regardless of this setting.
            Make sure to set use_deepspeed to `True` here
            if you are using `accelerate launch` with deepspeed setting in config YAML.

        Returns
        -------
        self
            The runtime instance.

        Examples
        --------
        >>> pt_runtime.with_command("torchrun train.py")
        """
        if use_deepspeed:
            self.set_spec(self.CONST_DEEPSPEED, True)
        return self.set_spec(self.CONST_COMMAND, command)

    @property
    def command(self):
        """The command for launching the workload."""
        return self.get_spec(self.CONST_COMMAND)
    
    @property
    def use_deepspeed(self):
        """Indicate whether whether to configure deepspeed for multi-node workload"""
        if self.get_spec(self.CONST_DEEPSPEED):
            return True
        return False


class PyTorchDistributedArtifact(PythonArtifact):
    CONST_DRIVER_SCRIPT = "driver_pytorch.py"
    CONST_LIB_HOSTNAME = "hostname_from_env.c"
    CONST_OCI_METRICS = "oci_metrics.py"

    def __init__(self, source, runtime=None) -> None:
        if not source:
            source = ""
        super().__init__(source, runtime)

    def build(self):
        """Prepares job artifact."""
        self._copy_artifacts(
            drivers=[
                self.CONST_DRIVER_UTILS,
                self.CONST_DRIVER_SCRIPT,
                self.CONST_LIB_HOSTNAME,
                self.CONST_OCI_METRICS,
                GitPythonArtifact.CONST_DRIVER_SCRIPT,
            ]
        )

        # Zip the job artifact
        self.path = self._zip_artifacts()
