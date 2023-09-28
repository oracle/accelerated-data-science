#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

import click

from ads.opctl import logger
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.constants import DEFAULT_ADS_CONFIG_FOLDER
from ads.opctl.operator.common.operator_yaml_generator import YamlGenerator
from ads.opctl.operator.common.utils import _load_yaml_from_uri


def init(**kwargs: Dict) -> str:
    """
    Generates operator config by the schema.

    Properties
    ----------
    kwargs: (Dict, optional).
        Additional key value arguments.

        - type: str
            The type of the operator.

    Returns
    -------
    str
        The YAML specification generated based on the schema.
    """
    logger.info("==== Feature Store related options ====")

    # Extract information from the configs and env variables
    p = ConfigProcessor().step(
        ConfigMerger,
        ads_config=kwargs.get("ads_config") or DEFAULT_ADS_CONFIG_FOLDER,
        **kwargs,
    )

    rm_stack_id = click.prompt(
        "Provide a resource manager stack id",
        type=str,
    )
    compartment_id = click.prompt(
        "Provide a resource manager stack id",
        type=str,
        default=p.config["infrastructure"].get("compartment_id"),
    )

    return YamlGenerator(
        schema=_load_yaml_from_uri(__file__.replace("cmd.py", "schema.yaml"))
    ).generate_example(
        values={
            "stack_id": rm_stack_id,
            "compartment_id": compartment_id,
        }
    )
