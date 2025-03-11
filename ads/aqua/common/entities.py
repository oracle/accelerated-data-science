#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict

from oci.data_science.models import Model
from pydantic import BaseModel, Field


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


class ModelConfigResult(BaseModel):
    """
    Represents the result of getting the AQUA model configuration.

    Attributes:
        model_details (Dict[str, Any]): A dictionary containing model details extracted from OCI.
        config (Dict[str, Any]): A dictionary of the loaded configuration.
    """

    config: Dict[str, Any] = Field(..., description="Loaded configuration dictionary.")
    model_details: Model = Field(..., description="Details of the model from OCI.")
