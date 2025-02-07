#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Optional

from ads.aqua.config.utils.serializer import Serializable


class ContainerSpec:
    """
    Class to hold to hold keys within the container spec.
    """

    CONTAINER_SPEC = "containerSpec"
    CLI_PARM = "cliParam"
    SERVER_PORT = "serverPort"
    HEALTH_CHECK_PORT = "healthCheckPort"
    ENV_VARS = "envVars"
    RESTRICTED_PARAMS = "restrictedParams"
    EVALUATION_CONFIGURATION = "evaluationConfiguration"


class ShapeInfo(Serializable):
    instance_shape: Optional[str] = None
    instance_count: Optional[int] = None
    ocpus: Optional[float] = None
    memory_in_gbs: Optional[float] = None

    class Config:
        extra = "ignore"


class ModelInfo(Serializable):
    """Class for maintaining details of model to be deployed, usually for multi-model deployment."""

    model_id: str
    gpu_count: Optional[int] = None
    env_var: Optional[dict] = None

    class Config:
        extra = "ignore"
        protected_namespaces = ()
