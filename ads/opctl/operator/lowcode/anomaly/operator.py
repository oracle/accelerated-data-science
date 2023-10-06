#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

from .model.factory import AnomalyOperatorModelFactory
from .operator_config import AnomalyOperatorConfig


def operate(operator_config: AnomalyOperatorConfig) -> None:
    """Runs the anomaly detection operator."""
    AnomalyOperatorModelFactory.get_model(operator_config).generate_report()


def verify(spec: Dict, **kwargs: Dict) -> bool:
    """Verifies the anomaly detection operator config."""
    operator = AnomalyOperatorConfig.from_dict(spec)
    msg_header = (
        f"{'*' * 50} The operator config has been successfully verified {'*' * 50}"
    )
    print(msg_header)
    print(operator.to_yaml())
    print("*" * len(msg_header))
