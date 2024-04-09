#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

from pydantic import BaseModel, PositiveInt, ValidationError


class SupportMetricsFormat(BaseModel):
    """Format for supported evaluation metrics."""

    use_case: list
    key: str
    name: str
    description: str
    args: dict


class EvaluationConfigFormat(BaseModel):
    """Evaluation config format."""

    model_params: dict
    shape: Dict[str, dict]
    default: Dict[str, PositiveInt]


def check(conf_schema, conf):
    """Check if the format of the output dictionary is correct."""
    try:
        conf_schema(**conf)
        return True
    except ValidationError:
        return False
