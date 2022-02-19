#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import fsspec
import json
import yaml
from cerberus import Validator
from os import path

JOB_SCHEMA_PATH = "job_schema.json"
RUNTIME_SCHEMA_PATH = "runtime_schema.json"
INFRASTRUCTURE_SCHEMA_PATH = "infrastructure_schema.json"


def load_schema(schema_path):
    if path.exists(path.join(path.dirname(__file__), schema_path)):
        with fsspec.open(
            path.join(path.dirname(__file__), schema_path), mode="r", encoding="utf8"
        ) as f:
            schema = json.load(f)
        return schema
    else:
        raise FileNotFoundError(f"{schema_path} does not exist")


class ValidateRuntime:
    """Class used to validate a Runtime YAML"""

    def __init__(self, file):
        self.file = file
        self.schema = load_schema(RUNTIME_SCHEMA_PATH)

    def validate(self):
        """Validates the Runtime YAML

        Raises:
            ValueError: if invalid

        Returns:
            [dict]: Returns normalized dictionary if input matches a known schema, else raises error
        """
        v = Validator(self.schema)
        valid = v.validate(self.file)
        if not valid:
            raise ValueError(json.dumps(v.errors, indent=2))
        return v.normalized(self.file)


class ValidateInfrastructure:
    """Class used to validate an Engine YAML"""

    def __init__(self, file):
        self.file = file
        self.schema = load_schema(INFRASTRUCTURE_SCHEMA_PATH)

    def validate(self):
        """Validates the Engine YAML

        Raises:
            ValueError: if invalid

        Returns:
            [dict]: Returns normalized dictionary if input matches a known schema, else raises error
        """
        v = Validator(self.schema)
        valid = v.validate(self.file)
        if not valid:
            raise ValueError(json.dumps(v.errors, indent=2))
        return v.normalized(self.file)


class ValidateJob:
    """Class used to validate a Job YAML"""

    def __init__(self, file):
        self.file = file
        self.schema = load_schema(JOB_SCHEMA_PATH)

    def validate(self):
        """Validates the Job YAML

        Raises:
            ValueError: if invalid

        Returns:
            [dict]: Returns normalized dictionary if input matches a known schema, else raises error
        """
        v = Validator(self.schema)
        job_valid = v.validate(self.file)

        infrastructure_valid, runtime_valid = True, True

        if "infrastructure" in self.file["spec"]:
            infrastructure_valid = ValidateInfrastructure(
                self.file["spec"]["infrastructure"]
            ).validate()

        if "runtime" in self.file["spec"]:
            runtime_valid = ValidateRuntime(self.file["spec"]["runtime"]).validate()

        if not job_valid or not infrastructure_valid or not runtime_valid:
            raise ValueError(json.dumps(v.errors, indent=2))
        return v.normalized(self.file)


class ValidatorFactory:
    """ValidatorFactory is a factory class that calls appropriate
        Validator class based on the 'kind'

    Usage:
        spec = {}
        validated_dict = ValidatorFactory(spec).validate():
    """

    def __init__(self, file):
        if type(file) != dict:
            raise ValueError("Input is not a dictionary")
        else:
            self.file = file

    def validate(self):
        """Calls correct validator based on 'kind'

        Raises:
            TypeError: raised when 'kind' is not known

        Returns:
            [boolean]: Returns True if input matches a known schema, else False
        """
        if "kind" not in self.file:
            raise ValueError("No 'kind' found in input spec")
        if self.file["kind"] == "job":
            return ValidateJob(self.file).validate()
        elif self.file["kind"] == "infrastructure":
            return ValidateInfrastructure(self.file).validate()
        elif self.file["kind"] == "runtime":
            return ValidateRuntime(self.file).validate()
        else:
            raise TypeError(f"Unknown Kind: {self.file['kind']}")
