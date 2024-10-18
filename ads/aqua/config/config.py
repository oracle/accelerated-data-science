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


# TODO: move this to global config.json in object storage
def get_finetuning_config_defaults():
    """Generate and return the fine-tuning default configuration dictionary."""
    return {
        "shape": {
            "VM.GPU.A10.1": {"batch_size": 1, "replica": "1-10"},
            "VM.GPU.A10.2": {"batch_size": 1, "replica": "1-10"},
            "BM.GPU.A10.4": {"batch_size": 1, "replica": 1},
            "BM.GPU4.8": {"batch_size": 4, "replica": 1},
            "BM.GPU.L40S-NC.4": {"batch_size": 4, "replica": 1},
            "BM.GPU.A100-v2.8": {"batch_size": 6, "replica": 1},
            "BM.GPU.H100.8": {"batch_size": 6, "replica": 1},
        }
    }
