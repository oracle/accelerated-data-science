#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict, Any

import click

from ads.opctl import logger
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.common.operator_yaml_generator import YamlGenerator

from .const import SupportedDatabases

VAULT_OCID_PLACEHOLDER = '<VAULT_OCID>'
SECRET_NAME_PLACEHOLDER = '<SECRET_NAME>'
JDBC_CONNECTION_URL_PLACEHOLDER = '<JDBC_CONNECTION_URL>'
DB_URL_PLACEHOLDER = '<DB_URL>'


def __get_db_config(db_type: str) -> Dict[str, Any]:
    if db_type == SupportedDatabases.MySQL:
        return {
            "mysqlDBConfig": {
                "vaultOCID": VAULT_OCID_PLACEHOLDER,
                "secretName": SECRET_NAME_PLACEHOLDER,
                "jdbcConnectionUrl": JDBC_CONNECTION_URL_PLACEHOLDER,
            }
        }
    elif db_type == SupportedDatabases.ATP:
        return {
            "atpDBConfig": {
                "vaultOCID": JDBC_CONNECTION_URL_PLACEHOLDER,
                "dbURL": DB_URL_PLACEHOLDER
            }
        }
    else:
        return {}


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

    db_type = click.prompt(
        "Provide a database type:",
        type=click.Choice(SupportedDatabases.values()),
        default=SupportedDatabases.MySQL,
    )
    selected_db_config = __get_db_config(db_type)
    return YamlGenerator(
        schema=_load_yaml_from_uri(__file__.replace("cmd.py", "schema.yaml"))
    ).generate_example_dict(values={"configuredDB": db_type, **selected_db_config})
