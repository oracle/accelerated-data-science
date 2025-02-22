#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Optional

from pydantic import Field

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


class AquaMultiModelRef(Serializable):
    """
    Lightweight model descriptor used for multi-model deployment.

    This class only contains essential details
    required to fetch complete model metadata and deploy models.

    Attributes
    ----------
    model_id : str
        The unique identifier of the model.
    model_name : Optional[str]
        The name of the model.
    gpu_count : Optional[int]
        Number of GPUs required for deployment.
    env_var : Optional[Dict[str, Any]]
        Optional environment variables to override during deployment.
    """

    model_id: str = Field(..., description="The model OCID to deploy.")
    model_name: Optional[str] = Field(None, description="The name of model.")
    gpu_count: Optional[int] = Field(
        None, description="The gpu count allocation for the model."
    )
    env_var: Optional[dict] = Field(
        default_factory=dict, description="The environment variables of the model."
    )

    class Config:
        extra = "ignore"
        protected_namespaces = ()
