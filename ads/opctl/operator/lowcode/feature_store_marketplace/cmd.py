#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict, List

import click

from ads.opctl import logger
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.common.operator_yaml_generator import YamlGenerator

from ads.opctl.operator.lowcode.feature_store_marketplace.const import DBType


def _get_required_keys_for_db_type_(db_type: str) -> List[str]:
    if db_type == DBType.MySQL:
        # TODO: Revert when helidon server is available
        # return ["mysql", "mysql.vault", "mysql.basic"]
        return ["mysqlDBConfig"]
    elif db_type == DBType.ATP:
        return ["atpDBConfig"]
    else:
        return []


def init(**kwargs: Dict) -> dict:
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
    logger.info("==== Feature store marketplace related options ====")
    required_keys = []
    db_type = click.prompt(
        "Provide a database type:",
        type=click.Choice(DBType.values()),
        default=DBType.MySQL,
    )
    required_keys.append(_get_required_keys_for_db_type_(db_type))

    return YamlGenerator(
        schema=_load_yaml_from_uri(__file__.replace("cmd.py", "schema.yaml"))
    ).generate_example_dict(
        values={"configuredDB": db_type},
        required_keys=_get_required_keys_for_db_type_(db_type),
    )
