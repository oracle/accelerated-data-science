#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from ads.jobs.builders.runtimes.base import Runtime


class ContainerRuntime(Runtime):
    """Represents a container job runtime

    To define container runtime:
    >>> ContainerRuntime()
    >>> .with_image(
    >>>     "iad.ocir.io/ociodscdev/qq-repo/ubuntu",
    >>>     entrypoint=["/bin/sh", -c],
    >>>     cmd="sleep 5 && echo Hello World",
    >>> )
    >>> .with_environment_variable(MY_ENV="MY_VALUE")
    or
    >>> ContainerRuntime()
    >>> .with_image("iad.ocir.io/ociodscdev/qq-repo/ubuntu")
    >>> .with_cmd("sleep 5 && echo Hello World")
    >>> .with_entrypoint(["/bin/sh", "-c"])
    >>> .with_environment_variable(MY_ENV="MY_VALUE")

    Docker ENTRYPOINT and CMD can be either "exec form" or "shell form" (See references).
    The exec form is used when a list is passed in.
    The shell form is used when a space separated string is passed in.

    When using the ContainerRuntime with OCI Data Science Job, the exec form is recommended.
    For most images, when the entrypoint is set to ["/bin/sh", "-c"],
        cmd can be a string as if you are running shell command.

    Reference:
    https://docs.docker.com/engine/reference/builder/#entrypoint
    https://docs.docker.com/engine/reference/builder/#cmd

    """

    CONST_IMAGE = "image"
    CONST_ENTRYPOINT = "entrypoint"
    CONST_CMD = "cmd"

    @property
    def image(self) -> str:
        """The container image"""
        return self.get_spec(self.CONST_IMAGE)

    def with_image(self, image: str, entrypoint: str = None, cmd: str = None):
        """Specify the image for the container job.

        Parameters
        ----------
        image : str
            The container image, e.g. iad.ocir.io/your_tenancy/your_repo/your_image:your_tag
        entrypoint : str, optional
            Entrypoint for the job, by default None (the entrypoint defined in the image will be used).
        cmd : str, optional
            Command for the job, by default None.

        Returns
        -------
        self
            The runtime instance.
        """
        self.with_entrypoint(entrypoint)
        self.set_spec(self.CONST_CMD, cmd)
        return self.set_spec(self.CONST_IMAGE, image)

    @property
    def entrypoint(self) -> str:
        """Entrypoint of the container job"""
        return self.get_spec(self.CONST_ENTRYPOINT)

    def with_entrypoint(self, entrypoint: str):
        """Specifies the entrypoint for the container job.

        Parameters
        ----------
        entrypoint : str
            Entrypoint for the container job

        Returns
        -------
        self
            The runtime instance.
        """
        self._spec[self.CONST_ENTRYPOINT] = entrypoint
        return self

    @property
    def cmd(self) -> str:
        """Command of the container job"""
        return self.get_spec(self.CONST_CMD)

    def with_cmd(self, cmd: str):
        """Specifies the command for the container job.

        Parameters
        ----------
        cmd : str
            Command for the container job

        Returns
        -------
        self
            The runtime instance.
        """
        self._spec[self.CONST_CMD] = cmd
        return self
