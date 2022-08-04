#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import re
from typing import Dict
from ads.jobs.builders.base import Builder
from ads.jobs import env_var_parser


class Runtime(Builder):
    """Base class for job runtime"""

    # Constant strings
    CONST_ENV_VAR = "env"
    CONST_ARGS = "args"
    CONST_MAXIMUM_RUNTIME_IN_MINUTES = "maximumRuntimeInMinutes"
    CONST_TAG = "freeformTags"

    attribute_map = {
        CONST_TAG: "freeform_tags",
        CONST_ENV_VAR: CONST_ENV_VAR,
    }

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        env = {}
        if spec and "env" in spec and isinstance(spec["env"], dict):
            env = spec.pop("env", {})
        if "env" in kwargs:
            env = kwargs.pop("env")
        super().__init__(spec, **kwargs)
        if isinstance(env, dict):
            self.with_environment_variable(**env)

    @property
    def kind(self):
        """Kind of the object to be stored in YAML. All runtimes will have "runtime" as kind.
        Subclass will have different types.
        """
        return "runtime"

    @property
    def type(self) -> str:
        """The type of the object as showing in YAML"""
        class_name = self.__class__.__name__
        if class_name.endswith("Runtime") and class_name != "Runtime":
            class_name = re.sub(r"Runtime$", "", class_name)
        return class_name[0].lower() + (class_name[1:] if len(class_name) > 1 else "")

    def with_argument(self, *args, **kwargs):
        """Adds command line arguments to the runtime.
        Existing arguments will be preserved.
        This method can be called (chained) multiple times to add various arguments.
        For example, runtime.with_argument(key="val").with_argument("path/to/file") will result in:
        "--key val path/to/file"

        Parameters
        ----------
        args:
            Positional arguments.
            In a single method call, positional arguments are always added before keyword arguments.
            You can call with_argument() to add positional arguments after keyword arguments.

        kwargs:
            Keyword arguments.
            To add a keyword argument without value, set the value to None.

        Returns
        -------
        Runtime
            This method returns self to support chaining methods.

        Raises
        ------
        ValueError
            Keyword arguments with space in a key.
        """
        arg_values = self.get_spec(self.CONST_ARGS, [])
        args = [str(arg) for arg in args]
        arg_values.extend(args)
        for k, v in kwargs.items():
            if " " in k:
                raise ValueError("Argument key cannot contain space.")
            arg_values.append(f"--{str(k)}")
            # Ignore None value
            if v is None:
                continue
            arg_values.append(str(v))
        self.set_spec(self.CONST_ARGS, arg_values)
        return self

    def with_environment_variable(self, **kwargs):
        """Sets environment variables

        Returns
        -------
        Runtime
            This method returns self to support chaining methods.
        """
        if not kwargs:
            return self
        envs = [{"name": k, "value": v} for k, v in kwargs.items()]
        return self.set_spec(self.CONST_ENV_VAR, envs)

    def with_freeform_tag(self, **kwargs):
        """Sets freeform tag

        Returns
        -------
        Runtime
            This method returns self to support chaining methods.
        """
        return self.set_spec(self.CONST_TAG, kwargs)

    def with_maximum_runtime_in_minutes(self, maximum_runtime_in_minutes: int):
        """Sets maximum runtime in minutes

        Returns
        -------
        Runtime
            This method returns self to support chaining methods.
        """
        return self.set_spec(
            self.CONST_MAXIMUM_RUNTIME_IN_MINUTES, maximum_runtime_in_minutes
        )

    @property
    def environment_variables(self) -> dict:
        """Environment variables

        Returns
        -------
        dict
            The runtime environment variables.
            The returned dictionary is a copy.
        """
        env_var_list = self.get_spec(self.CONST_ENV_VAR)
        if env_var_list:
            return env_var_parser.parse(env_var_list)
        return {}

    @property
    def envs(self) -> dict:
        """Environment variables"""
        return self.environment_variables

    @property
    def freeform_tags(self) -> dict:
        """freeform_tags"""
        return self.get_spec(self.CONST_TAG, {})

    @property
    def args(self) -> list:
        """Command line arguments"""
        return self.get_spec(self.CONST_ARGS, [])

    @property
    def maximum_runtime_in_minutes(self) -> int:
        """Maximum runtime in minutes"""
        return self.get_spec(self.CONST_MAXIMUM_RUNTIME_IN_MINUTES)
