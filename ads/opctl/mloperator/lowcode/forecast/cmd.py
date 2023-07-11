#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

import click

from ads.opctl.mloperator.common.utils import YamlGenerator, _load_yaml_from_uri

SUPPORTED_MODELS = ["arima", "automlx", "neuralprophet", "prophet"]


def init(**kwargs: Dict) -> str:
    """
    Generates a starter specification template YAML for the operator.

    Returns
    -------
    str
        The YAML specification generated based on the schema.
    """
    print("==== Forecasting related options ====")
    model_type = click.prompt(
        "Provide a model type:", type=click.Choice(SUPPORTED_MODELS), default="prophet"
    )
    schema = _load_yaml_from_uri(__file__.replace("cmd.py", "schema.yaml"))
    return YamlGenerator(schema=schema).generate_example()
