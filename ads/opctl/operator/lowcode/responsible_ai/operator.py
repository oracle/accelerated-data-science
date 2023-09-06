#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

from .guardrail.guardrail import OfflineGuardRail, OnlineGuardRail



def operate(operator_config: dict) -> None:
    """Runs the forecasting operator."""
    if operator_config.get("spec", {}).get("type") == "deployment":
        return OnlineGuardRail(operator_config).predict()
    elif operator_config.get("spec", {}).get("type") == "job":
        return OfflineGuardRail(operator_config).generate_report()
    else:
        raise NotImplemented("`type` can be only `deployment` or `job`.")


def verify(spec: Dict) -> bool:
    """Verifies the forecasting operator config."""
    pass
