#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

from .model.factory import ForecastOperatorModelFactory
from .operator_config import ForecastOperatorConfig

from ads.opctl import logger

def operate(operator_config: ForecastOperatorConfig) -> None:
    """Runs the forecasting operator."""
    ForecastOperatorModelFactory.get_model(operator_config).generate_report()


def verify(spec: Dict) -> bool:
    """Verifies the forecasting operator config."""
    operator = ForecastOperatorConfig.from_dict(spec)
    msg_header = (
        f"{'*' * 50} The operator config has been successfully verified {'*' * 50}"
    )
    logger.info(operator.to_yaml())
    logger.info("*" * len(msg_header))
