#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from logging import getLogger
from collections import namedtuple
import fsspec
import yaml
import importlib

logger = getLogger("ads.yaml")


class YamlSpecParser:
    def translate_config(self, config: list):
        """
        Translates config to three element tuple of dictionary- (file, envVars, concatenated args)

        All elements that are of type `env` goes to envVars
        All elements that are of type `args` are concatenated as `--key1=value1 --key2=value2 ..` string
        All elements of type `file` are currently ignored
        """
        # TODO parse file type

        parsed_config = namedtuple(
            "ClusterConfig", field_names=["files", "envVars", "cmd_args"]
        )
        envVars = {}
        files = {}
        cmd_args = ""
        if config:
            if config.get("env"):
                for item in config["env"]:
                    envVars[item["name"]] = item["value"]
            cmd_args = " ".join(config.get("startOptions", [""]))
        return parsed_config(files=files, envVars=envVars, cmd_args=cmd_args)

    def parse(self):
        pass

    @classmethod
    def parse_content(cls, file):
        yaml_spec = {}
        if isinstance(file, dict):
            yaml_spec = file
        elif isinstance(file, str):
            with fsspec.open(file) as yf:
                yaml_spec = yaml.load(yf.read(), yaml.SafeLoader)
        kind = yaml_spec.get("kind")
        parsed_output = None
        if kind:
            className = f"{kind[0].upper()}{kind[1:]}SpecParser"
            m = importlib.import_module("ads.opctl.config.yaml_parsers")
            parser = getattr(m, className)
            parsed_output = parser(yaml_spec).parse()
        return parsed_output
