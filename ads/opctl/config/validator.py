#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.config.base import ConfigProcessor
from ads.opctl import logger
from cerberus import Validator
import yaml
import json


def _load_operator_schema(operator_type):
    try:
        import importlib.resources as pkg_resources
    except ImportError:
        # Try backported to PY<37 `importlib_resources`.
        import importlib_resources as pkg_resources
    forecast_module = pkg_resources.import_module(f"ads.operators.{operator_type}")
    try:
        inp_file = pkg_resources.files(forecast_module) / "schema.yaml"
        with inp_file.open("rb") as f:
            schema = f.read()
    except AttributeError:
        # Python < PY3.9, fall back to method deprecated in PY3.11.
        # schema = pkg_resources.read_text(forecast_module, "schema.yaml")
        # or for a file-like stream:
        schema = pkg_resources.open_text(forecast_module, "schema.yaml")
    return schema


class ConfigValidator(ConfigProcessor):
    def process(self):
        if self.config.get("kind") != "operator":
            # TODO: add validation using pydantic + datamodel-code-generator in future PR

            # called to update command, part of which is encoded spec
            # and spec might have been updated during the call above
            # self.["execution"]["command"] = _resolve_command(self.config)
            return self
        try:
            schema = _load_operator_schema(self.config.get("type"))
        except:
            logger.warn(
                "Yaml Validation file not found. Continuing without formal validation."
            )
            return self.config

        v = Validator(yaml.safe_load(schema))
        valid = v.validate(self.config)
        if not valid:
            raise ValueError(json.dumps(v.errors, indent=2))
        self.config = v.normalized(self.config)
        return self
