#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import logging
from typing import Union
from ads.jobs.builders.runtimes.base import MultiNodeRuntime

logger = logging.getLogger(__name__)


class ContainerRuntime(MultiNodeRuntime):
    """Represents a container job runtime

    To define container runtime:

    >>> ContainerRuntime()
    >>> .with_image("iad.ocir.io/<your_tenancy>/<your_image>:<tag>")
    >>> .with_cmd("sleep 5 && echo Hello World")
    >>> .with_entrypoint(["/bin/sh", "-c"])
    >>> .with_image_digest("<image_digest>")
    >>> .with_image_signature_id("<image_signature_id>")
    >>> .with_environment_variable(MY_ENV="MY_VALUE")

    Alternatively, you can define the ``entrypoint``, ``cmd``,
    ``image_digest``and ``image_signature_id`` along with the image.

    >>> ContainerRuntime()
    >>> .with_image(
    >>>     "iad.ocir.io/<your_tenancy>/<your_image>:<tag>",
    >>>     entrypoint=["/bin/sh", "-c"],
    >>>     cmd="sleep 5 && echo Hello World",
    >>>     image_digest="<image_digest>",
    >>>     image_signature_id="<image_signature_id>",
    >>> )
    >>> .with_environment_variable(MY_ENV="MY_VALUE")

    The entrypoint and cmd can be either "exec form" or "shell form" (See references).
    The exec form is used when a list is passed in.
    The shell form is used when a space separated string is passed in.

    When using the ContainerRuntime with OCI Data Science Job, the exec form is recommended.
    For most images, when the entrypoint is set to ``["/bin/sh", "-c"]``,
    ``cmd`` can be a string as if you are running shell command.

    References
    ----------
    https://docs.docker.com/engine/reference/builder/#entrypoint
    https://docs.docker.com/engine/reference/builder/#cmd

    """

    CONST_IMAGE = "image"
    CONST_ENTRYPOINT = "entrypoint"
    CONST_CMD = "cmd"
    CONST_IMAGE_DIGEST = "imageDigest"
    CONST_IMAGE_SIGNATURE_ID = "imageSignatureId"
    CONST_SCRIPT_PATH = "scriptPathURI"
    attribute_map = {
        CONST_IMAGE: CONST_IMAGE,
        CONST_ENTRYPOINT: CONST_ENTRYPOINT,
        CONST_CMD: CONST_CMD,
        CONST_IMAGE_DIGEST: "image_digest",
        CONST_IMAGE_SIGNATURE_ID: "image_signature_id",
    }
    attribute_map.update(MultiNodeRuntime.attribute_map)

    @property
    def job_env_type(self) -> str:
        """The container type"""
        return "OCIR_CONTAINER"

    @property
    def image(self) -> str:
        """The container image"""
        return self.get_spec(self.CONST_IMAGE)

    def with_image(
        self,
        image: str,
        entrypoint: Union[str, list, None] = None, 
        cmd: str = None,
        image_digest: str = None,
        image_signature_id: str = None,
    ) -> "ContainerRuntime":
        """Specify the image for the container job.

        Parameters
        ----------
        image : str
            The container image, e.g. iad.ocir.io/<your_tenancy>/<your_image>:<your_tag>
        entrypoint : str or list, optional
            Entrypoint for the job, by default None (the entrypoint defined in the image will be used).
        cmd : str, optional
            Command for the job, by default None.
        image_digest: str, optional
            The image digest, by default None.
        image_signature_id: str, optional
            The image signature id, by default None.

        Returns
        -------
        ContainerRuntime
            The runtime instance.
        """
        if not isinstance(image, str):
            raise ValueError(
                "Custom image must be provided as a string."
            )
        if image.find(":") < 0:
            logger.warning(
                "Tag is required for custom image. Accepted format: iad.ocir.io/<tenancy>/<image>:<tag>."
            )
        self.with_entrypoint(entrypoint)
        self.set_spec(self.CONST_CMD, cmd)
        self.with_image_digest(image_digest)
        self.with_image_signature_id(image_signature_id)
        return self.set_spec(self.CONST_IMAGE, image)

    @property
    def image_digest(self) -> str:
        """The container image digest."""
        return self.get_spec(self.CONST_IMAGE_DIGEST)

    def with_image_digest(self, image_digest: str) -> "ContainerRuntime":
        """Sets the digest of custom image.

        Parameters
        ----------
        image_digest: str
            The image digest.

        Returns
        -------
        ContainerRuntime
            The runtime instance.
        """
        return self.set_spec(self.CONST_IMAGE_DIGEST, image_digest)

    @property
    def image_signature_id(self) -> str:
        """The container image signature id."""
        return self.get_spec(self.CONST_IMAGE_SIGNATURE_ID)

    def with_image_signature_id(self, image_signature_id: str) -> "ContainerRuntime":
        """Sets the signature id of custom image.

        Parameters
        ----------
        image_signature_id: str
            The image signature id.

        Returns
        -------
        ContainerRuntime
            The runtime instance.
        """
        return self.set_spec(
            self.CONST_IMAGE_SIGNATURE_ID,
            image_signature_id
        )

    @property
    def entrypoint(self) -> str:
        """Entrypoint of the container job"""
        return self.get_spec(self.CONST_ENTRYPOINT)

    def with_entrypoint(self, entrypoint: Union[str, list]) -> "ContainerRuntime":
        """Specifies the entrypoint for the container job.

        Parameters
        ----------
        entrypoint : str or list
            Entrypoint for the container job

        Returns
        -------
        ContainerRuntime
            The runtime instance.
        """
        self._spec[self.CONST_ENTRYPOINT] = entrypoint
        return self

    @property
    def cmd(self) -> str:
        """Command of the container job"""
        return self.get_spec(self.CONST_CMD)

    def with_cmd(self, cmd: str) -> "ContainerRuntime":
        """Specifies the command for the container job.

        Parameters
        ----------
        cmd : str
            Command for the container job

        Returns
        -------
        ContainerRuntime
            The runtime instance.
        """
        self._spec[self.CONST_CMD] = cmd
        return self

    def init(self, **kwargs) -> "ContainerRuntime":
        """Initializes a starter specification for the runtime.

        Returns
        -------
        ContainerRuntime
            The runtime instance.
        """
        super().init(**kwargs)

        return self.with_image(
            image=kwargs.get("image", "iad.ocir.io/namespace/image:tag"),
            entrypoint=["bash", "--login", "-c"],
            cmd="{Container CMD. For MLflow and Operator will be auto generated}",
        )

    @property
    def artifact_uri(self) -> str:
        """The URI of the source code"""
        return self.get_spec(self.CONST_SCRIPT_PATH)

    def with_artifact(self, uri: str):
        """Specifies the artifact to be added to the container.

        Parameters
        ----------
        uri : str
            URI to the source code script, which can be any URI supported by fsspec,
            including http://, https:// and OCI object storage.
            For example: oci://your_bucket@your_namespace/path/to/script.py

        Returns
        -------
        self
            The runtime instance.
        """
        return self.set_spec(self.CONST_SCRIPT_PATH, uri)