#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

from ads.opctl.operator.common.operator_yaml_generator import YamlGenerator
from ads.opctl.operator.common.utils import _load_yaml_from_uri



def init(**kwargs: Dict) -> str:
    """Generates operator config by the schema."""
    return YamlGenerator(
        schema=_load_yaml_from_uri(__file__.replace("cmd.py", "schema.yaml"))
    ).generate_example_dict(values={"model": "random_forest", "type": kwargs.get("type")})
