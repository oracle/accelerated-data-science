#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/from typing import Dict


from typing import Dict, List
from ads.jobs.builders.base import Builder

MODEL_DEPLOYMENT_RUNTIME_KIND = "runtime"


class ModelDeploymentRuntimeType:
    CONDA = "conda"
    CONTAINER = "container"


class OCIModelDeploymentRuntimeType:
    CONDA = "DEFAULT"
    CONTAINER = "OCIR_CONTAINER"


class ModelDeploymentRuntime(Builder):
    """A class used to represent a Model Deployment Runtime.

    Attributes
    ----------
    env: Dict
        The environment variables of model deployment runtime.
    deployment_mode: str
        The deployment mode of model deployment.
    input_stream_ids: List
        The input stream ids of model deployment.
    output_stream_ids: List
        The output stream ids of model deployment.
    model_uri: str
        The model uri of model deployment.

    Methods
    -------
    with_env(env)
        Sets the environment variables of model deployment
    with_deployment_mode(deployment_mode)
        Sets the deployment mode of model deployment
    with_input_stream_ids(input_stream_ids)
        Sets the input stream ids of model deployment
    with_output_stream_ids(output_stream_ids)
        Sets the output stream ids of model deployment
    with_model_uri(model_uri)
        Sets the model uri of model deployment
    """

    CONST_MODEL_ID = "modelId"
    CONST_MODEL_URI = "modelUri"
    CONST_ENV = "env"
    CONST_ENVIRONMENT_VARIABLES = "environmentVariables"
    CONST_ENVIRONMENT_CONFIG_TYPE = "environmentConfigurationType"
    CONST_DEPLOYMENT_MODE = "deploymentMode"
    CONST_INPUT_STREAM_IDS = "inputStreamIds"
    CONST_OUTPUT_STREAM_IDS = "outputStreamIds"
    CONST_ENVIRONMENT_CONFIG_DETAILS = "environmentConfigurationDetails"

    attribute_map = {
        CONST_ENV: "env",
        CONST_ENVIRONMENT_VARIABLES: "environment_variables",
        CONST_INPUT_STREAM_IDS: "input_stream_ids",
        CONST_OUTPUT_STREAM_IDS: "output_stream_ids",
        CONST_DEPLOYMENT_MODE: "deployment_mode",
        CONST_MODEL_URI: "model_uri",
    }

    ENVIRONMENT_CONFIG_DETAILS_PATH = (
        "model_deployment_configuration_details.environment_configuration_details"
    )
    STREAM_CONFIG_DETAILS_PATH = (
        "model_deployment_configuration_details.stream_configuration_details"
    )
    MODEL_CONFIG_DETAILS_PATH = (
        "model_deployment_configuration_details.model_configuration_details"
    )

    payload_attribute_map = {
        CONST_ENV: f"{ENVIRONMENT_CONFIG_DETAILS_PATH}.environment_variables",
        CONST_INPUT_STREAM_IDS: f"{STREAM_CONFIG_DETAILS_PATH}.input_stream_ids",
        CONST_OUTPUT_STREAM_IDS: f"{STREAM_CONFIG_DETAILS_PATH}.output_stream_ids",
        CONST_DEPLOYMENT_MODE: "deployment_mode",
        CONST_MODEL_URI: f"{MODEL_CONFIG_DETAILS_PATH}.model_id",
    }

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        super().__init__(spec, **kwargs)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in YAML.

        Returns
        -------
        str
            runtime
        """
        return MODEL_DEPLOYMENT_RUNTIME_KIND

    @property
    def env(self) -> Dict:
        """The environment variables of model deployment.

        Returns
        -------
        Dict
            The environment variables of model deployment.
        """
        return self.get_spec(self.CONST_ENV, {})

    def with_env(self, env: Dict) -> "ModelDeploymentRuntime":
        """Sets the environment variables of model deployment.

        Parameters
        ----------
        env: Dict
            The environment variables of model deployment.

        Returns
        -------
        ModelDeploymentRuntime
            The ModelDeploymentRuntime instance (self).
        """
        return self.set_spec(self.CONST_ENV, env)

    @property
    def deployment_mode(self) -> str:
        """The deployment mode of model deployment.

        Returns
        -------
        str
            The deployment mode of model deployment.
        """
        return self.get_spec(self.CONST_DEPLOYMENT_MODE, None)

    def with_deployment_mode(self, deployment_mode: str) -> "ModelDeploymentRuntime":
        """Sets the deployment mode of model deployment.
        Can be HTTPS_ONLY or STREAM_ONLY.

        Parameters
        ----------
        deployment_mode: str
            The deployment mode of model deployment.

        Returns
        -------
        ModelDeploymentRuntime
            The ModelDeploymentRuntime instance (self).
        """
        return self.set_spec(self.CONST_DEPLOYMENT_MODE, deployment_mode)

    @property
    def input_stream_ids(self) -> List[str]:
        """The input stream ids of model deployment.

        Returns
        -------
        List
            The input stream ids of model deployment.
        """
        return self.get_spec(self.CONST_INPUT_STREAM_IDS, [])

    def with_input_stream_ids(
        self, input_stream_ids: List[str]
    ) -> "ModelDeploymentRuntime":
        """Sets the input stream ids of model deployment.

        Parameters
        ----------
        input_stream_ids: List
            The input stream ids of model deployment.

        Returns
        -------
        ModelDeploymentRuntime
            The ModelDeploymentRuntime instance (self).
        """
        return self.set_spec(self.CONST_INPUT_STREAM_IDS, input_stream_ids)

    @property
    def output_stream_ids(self) -> List[str]:
        """The output stream ids of model deployment.

        Returns
        -------
        List
            The output stream ids of model deployment.
        """
        return self.get_spec(self.CONST_OUTPUT_STREAM_IDS, [])

    def with_output_stream_ids(
        self, output_stream_ids: List[str]
    ) -> "ModelDeploymentRuntime":
        """Sets the output stream ids of model deployment.

        Parameters
        ----------
        output_stream_ids: List
            The output stream ids of model deployment.

        Returns
        -------
        ModelDeploymentRuntime
            The ModelDeploymentRuntime instance (self).
        """
        return self.set_spec(self.CONST_OUTPUT_STREAM_IDS, output_stream_ids)

    @property
    def model_uri(self) -> str:
        """The model uri of model deployment.

        Returns
        -------
        List
            The model uri of model deployment.
        """
        return self.get_spec(self.CONST_MODEL_URI, None)

    def with_model_uri(self, model_uri: str) -> "ModelDeploymentRuntime":
        """Sets the model uri of model deployment.

        Parameters
        ----------
        model_uri: str
            The model uri of model deployment.

        Returns
        -------
        ModelDeploymentRuntime
            The ModelDeploymentRuntime instance (self).
        """
        return self.set_spec(self.CONST_MODEL_URI, model_uri)


class ModelDeploymentCondaRuntime(ModelDeploymentRuntime):
    """A class used to represent a Model Deployment Conda Runtime.

    Examples
    --------
    >>> conda_runtime = ModelDeploymentCondaRuntime()
    ...        .with_env({"key":"value"})
    ...        .with_deployment_mode("HTTPS_ONLY")
    ...        .with_model_uri(<model_uri>)
    >>> conda_runtime.to_dict()
    """

    @property
    def type(self) -> str:
        """The type of the object as showing in YAML.

        Returns
        -------
        str
            conda
        """
        return ModelDeploymentRuntimeType.CONDA

    @property
    def environment_config_type(self) -> str:
        """The environment config type of model deployment.

        Returns
        -------
        str
            DEFAULT
        """
        return OCIModelDeploymentRuntimeType.CONDA


class ModelDeploymentContainerRuntime(ModelDeploymentRuntime):
    """A class used to represent a Model Deployment Container Runtime.

    Attributes
    ----------
    image: str
        The image of model deployment container runtime.
    image_digest: str
        The image digest of model deployment container runtime.
    cmd: List
        The cmd of model deployment container runtime.
    entrypoint: List
        The entrypoint of model deployment container runtime.
    server_port: int
        The server port of model deployment container runtime.
    health_check_port: int
        The health check port of model deployment container runtime.

    Methods
    -------
    with_image(image)
        Sets the image of model deployment container runtime
    with_image_digest(image_digest)
        Sets the image digest of model deployment container runtime
    with_cmd(cmd)
        Sets the cmd of model deployment container runtime
    with_entrypoint(entrypoint)
        Sets the entrypoint of model deployment container runtime
    with_server_port(server_port)
        Sets the server port of model deployment container runtime
    with_health_check_port(health_check_port)
        Sets the health check port of model deployment container runtime

    Examples
    --------
    Build runtime from builder apis:
    >>> container_runtime = ModelDeploymentContainerRuntime()
    ...        .with_image(<image>)
    ...        .with_image_digest(<image_digest>)
    ...        .with_entrypoint(<entrypoint>)
    ...        .with_server_port(<server_port>)
    ...        .with_health_check_port(<health_check_port>)
    ...        .with_env({"key":"value"})
    ...        .with_deployment_mode("HTTPS_ONLY")
    ...        .with_model_uri(<model_uri>)
    >>> container_runtime.to_dict()

    Build runtime from yaml:
    >>> container_runtime = ModelDeploymentContainerRuntime.from_yaml(uri=<path_to_yaml>)
    """

    CONST_IMAGE = "image"
    CONST_IMAGE_DIGEST = "imageDigest"
    CONST_CMD = "cmd"
    CONST_ENTRYPOINT = "entrypoint"
    CONST_SERVER_PORT = "serverPort"
    CONST_HEALTH_CHECK_PORT = "healthCheckPort"

    attribute_map = {
        **ModelDeploymentRuntime.attribute_map,
        CONST_IMAGE: "image",
        CONST_IMAGE_DIGEST: "image_digest",
        CONST_CMD: "cmd",
        CONST_ENTRYPOINT: "entrypoint",
        CONST_SERVER_PORT: "server_port",
        CONST_HEALTH_CHECK_PORT: "health_check_port",
    }

    payload_attribute_map = {
        **ModelDeploymentRuntime.payload_attribute_map,
        CONST_IMAGE: f"{ModelDeploymentRuntime.ENVIRONMENT_CONFIG_DETAILS_PATH}.image",
        CONST_IMAGE_DIGEST: f"{ModelDeploymentRuntime.ENVIRONMENT_CONFIG_DETAILS_PATH}.image_digest",
        CONST_CMD: f"{ModelDeploymentRuntime.ENVIRONMENT_CONFIG_DETAILS_PATH}.cmd",
        CONST_ENTRYPOINT: f"{ModelDeploymentRuntime.ENVIRONMENT_CONFIG_DETAILS_PATH}.entrypoint",
        CONST_SERVER_PORT: f"{ModelDeploymentRuntime.ENVIRONMENT_CONFIG_DETAILS_PATH}.server_port",
        CONST_HEALTH_CHECK_PORT: f"{ModelDeploymentRuntime.ENVIRONMENT_CONFIG_DETAILS_PATH}.health_check_port",
    }

    @property
    def type(self) -> str:
        """The type of the object as showing in YAML.

        Returns
        -------
        str
            container
        """
        return ModelDeploymentRuntimeType.CONTAINER

    @property
    def environment_config_type(self) -> str:
        """The environment config type of model deployment.

        Returns
        -------
        str
            OCIR_CONTAINER
        """
        return OCIModelDeploymentRuntimeType.CONTAINER

    @property
    def image(self) -> str:
        """The image of model deployment container runtime.

        Returns
        -------
        str
            The image of model deployment container runtime.
        """
        return self.get_spec(self.CONST_IMAGE, None)

    def with_image(self, image: str) -> "ModelDeploymentContainerRuntime":
        """Sets the image of model deployment container runtime.

        Parameters
        ----------
        image: str
            The image of model deployment container runtime.

        Returns
        -------
        ModelDeploymentContainerRuntime
            The ModelDeploymentContainerRuntime instance (self).
        """
        return self.set_spec(self.CONST_IMAGE, image)

    @property
    def image_digest(self) -> str:
        """The image digest of model deployment container runtime.

        Returns
        -------
        str
            The image digest of model deployment container runtime.
        """
        return self.get_spec(self.CONST_IMAGE_DIGEST, None)

    def with_image_digest(self, image_digest: str) -> "ModelDeploymentContainerRuntime":
        """Sets the image digest of model deployment container runtime.

        Parameters
        ----------
        image_digest: str
            The image digest of model deployment container runtime.

        Returns
        -------
        ModelDeploymentContainerRuntime
            The ModelDeploymentContainerRuntime instance (self).
        """
        return self.set_spec(self.CONST_IMAGE_DIGEST, image_digest)

    @property
    def cmd(self) -> List[str]:
        """The command lines of model deployment container runtime.

        Returns
        -------
        List
            The command lines of model deployment container runtime.
        """
        return self.get_spec(self.CONST_CMD, [])

    def with_cmd(self, cmd: List[str]) -> "ModelDeploymentContainerRuntime":
        """Sets the cmd of model deployment container runtime.

        Parameters
        ----------
        cmd: List
            The cmd of model deployment container runtime.

        Returns
        -------
        ModelDeploymentContainerRuntime
            The ModelDeploymentContainerRuntime instance (self).
        """
        return self.set_spec(self.CONST_CMD, cmd)

    @property
    def entrypoint(self) -> List[str]:
        """The entry point of model deployment container runtime.

        Returns
        -------
        List
            The entry point of model deployment container runtime.
        """
        return self.get_spec(self.CONST_ENTRYPOINT, [])

    def with_entrypoint(
        self, entrypoint: List[str]
    ) -> "ModelDeploymentContainerRuntime":
        """Sets the entrypoint of model deployment container runtime.

        Parameters
        ----------
        entrypoint: List
            The entrypoint of model deployment container runtime.

        Returns
        -------
        ModelDeploymentContainerRuntime
            The ModelDeploymentContainerRuntime instance (self).
        """
        return self.set_spec(self.CONST_ENTRYPOINT, entrypoint)

    @property
    def server_port(self) -> int:
        """The server port of model deployment container runtime.

        Returns
        -------
        int
            The server port of model deployment container runtime.
        """
        return self.get_spec(self.CONST_SERVER_PORT, None)

    def with_server_port(self, server_port: int) -> "ModelDeploymentContainerRuntime":
        """Sets the server port of model deployment container runtime.

        Parameters
        ----------
        server_port: List
            The server port of model deployment container runtime.

        Returns
        -------
        ModelDeploymentContainerRuntime
            The ModelDeploymentContainerRuntime instance (self).
        """
        return self.set_spec(self.CONST_SERVER_PORT, server_port)

    @property
    def health_check_port(self) -> int:
        """The health check port of model deployment container runtime.

        Returns
        -------
        int
            The health check port of model deployment container runtime.
        """
        return self.get_spec(self.CONST_HEALTH_CHECK_PORT, None)

    def with_health_check_port(
        self, health_check_port: int
    ) -> "ModelDeploymentContainerRuntime":
        """Sets the health check port of model deployment container runtime.

        Parameters
        ----------
        health_check_port: List
            The health check port of model deployment container runtime.

        Returns
        -------
        ModelDeploymentContainerRuntime
            The ModelDeploymentContainerRuntime instance (self).
        """
        return self.set_spec(self.CONST_HEALTH_CHECK_PORT, health_check_port)
