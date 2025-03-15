#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import json
from typing import Optional

from ads.aqua.app import AquaApp
from ads.aqua.config.evaluation.evaluation_service_config import (
    EvaluationServiceConfig,
    ModelParamsConfig,
    UIConfig,
)

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
    container_item = next(
        (
            c
            for c in AquaApp().get_container_config()
            if c.is_latest and c.family_name == container
        ),
        None,
    )
    shapes = json.loads(
        container_item.workload_configuration_details_list[0]
        .use_case_configuration.get("additionalConfigurations")
        .get("shapes")
    )
    metrics = json.loads(
        container_item.workload_configuration_details_list[0]
        .use_case_configuration.get("additionalConfigurations")
        .get("metrics")
    )
    model_params = ModelParamsConfig(
        default={
            "model": "odsc-llm",
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop": [],
        }
    )
    return EvaluationServiceConfig(
        ui_config=UIConfig(model_params=model_params, shapes=shapes, metrics=metrics)
    )
