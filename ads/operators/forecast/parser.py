#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import yaml
import fsspec
from string import Template
from typing import Dict, List, Tuple, Union
import os


def _load_yaml_from_string(doc, **kwargs) -> Dict:
    template_dict = {**os.environ, **kwargs}
    return yaml.safe_load(
        Template(doc).safe_substitute(
            **template_dict,
        )
    )

def _load_yaml_from_uri(uri, **kwargs) -> str:
    with fsspec.open(uri) as f:
        return _load_yaml_from_string(str(f.read(), "UTF-8"), **kwargs)


class ForecastYamlParser:
    def __init__(self):
        pass

    def parse(self, config):
        pass

    def validate(self, config):
        module_schema = _load_yaml_from_uri(__file__.replace("parser.py", "schema.yaml"))
        config.validate(module_schema)
        print("schema valid.")
        return config
