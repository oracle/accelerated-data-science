#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import Optional

from ads.aqua.common.entities import ContainerSpec
from ads.aqua.common.utils import get_container_config
from ads.aqua.config.evaluation.evaluation_service_config import EvaluationServiceConfig

DEFAULT_EVALUATION_CONTAINER = "odsc-llm-evaluate"


def get_evaluation_service_config(
    container: Optional[str] = DEFAULT_EVALUATION_CONTAINER,
) -> EvaluationServiceConfig:
    """
    Retrieves the common evaluation configuration.

    Returns
    -------
    EvaluationServiceConfig: The evaluation common config.
    """

    container = container or DEFAULT_EVALUATION_CONTAINER
    return EvaluationServiceConfig(
        **get_container_config()
        .get(ContainerSpec.CONTAINER_SPEC, {})
        .get(container, {})
    )
