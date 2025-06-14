#!/usr/bin/env python

# Copyright (c) 2022, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from __future__ import annotations

import re
import time
import traceback
from typing import Dict, TypeVar

from ads.jobs import env_var_parser
from ads.jobs.builders.base import Builder

Self = TypeVar("Self", bound="Runtime")


class Runtime(Builder):
    """Base class for job runtime"""

    # Constant strings
    CONST_ENV_VAR = "env"
    CONST_ARGS = "args"
    CONST_MAXIMUM_RUNTIME_IN_MINUTES = "maximumRuntimeInMinutes"
    CONST_FREEFORM_TAGS = "freeformTags"
    CONST_DEFINED_TAGS = "definedTags"

    attribute_map = {
        CONST_FREEFORM_TAGS: "freeform_tags",
        CONST_DEFINED_TAGS: "defined_tags",
        CONST_ENV_VAR: CONST_ENV_VAR,
        CONST_ARGS: CONST_ARGS,
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
    def kind(self) -> str:
        """Kind of the object to be stored in YAML.
        All runtime implementations will have "runtime" as kind.
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

    def with_argument(self: Self, *args, **kwargs) -> Self:
        """Adds command line arguments to the runtime.

        This method can be called (chained) multiple times to add various arguments.

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
        Self
            This method returns self to support chaining methods.

        Raises
        ------
        ValueError
            Keyword arguments with space in a key.

        Examples
        --------
        >>> runtime = Runtime().with_argument(key1="val1", key2="val2").with_argument("pos1")
        >>> print(runtime.args)
        ["--key1", "val1", "--key2", "val2", "pos1"]

        >>> runtime = Runtime()
        >>> runtime.with_argument("pos1")
        >>> runtime.with_argument(key1="val1", key2="val2.1 val2.2")
        >>> runtime.with_argument("pos2")
        >>> print(runtime.args)
        ['pos1', '--key1', 'val1', '--key2', 'val2.1 val2.2', 'pos2']

        >>> runtime = Runtime()
        >>> runtime.with_argument("pos1")
        >>> runtime.with_argument(key1=None, key2="val2")
        >>> runtime.with_argument("pos2")
        >>> print(runtime.args)
        ["pos1", "--key1", "--key2", "val2", "pos2"]

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

    def with_environment_variable(self: Self, **kwargs) -> Self:
        """Sets environment variables

        Environment variables enclosed by ``${...}`` will be substituted.

        * You can use ``$$`` to escape the substitution.
        * Undefined variable enclosed by ``${}`` will be ignored.
        * Double dollar signs ``$$`` will be substituted by a single one ``$``.

        Returns
        -------
        Self
            This method returns self to support chaining methods.

        Examples
        --------
        >>> runtime = (
        ...     PythonRuntime()
        ...     .with_environment_variable(
        ...         HOST="10.0.0.1",
        ...         PORT="443",
        ...         URL="http://${HOST}:${PORT}/path/",
        ...         ESCAPED_URL="http://$${HOST}:$${PORT}/path/",
        ...         MISSING_VAR="This is ${UNDEFINED}",
        ...         VAR_WITH_DOLLAR="$10",
        ...         DOUBLE_DOLLAR="$$10"
        ...     )
        ... )
        >>> for k, v in runtime.environment_variables.items():
        ...     print(f"{k}: {v}")
        HOST: 10.0.0.1
        PORT: 443
        URL: http://10.0.0.1:443/path/
        ESCAPED_URL: http://${HOST}:${PORT}/path/
        MISSING_VAR: This is ${UNDEFINED}
        VAR_WITH_DOLLAR: $10
        DOUBLE_DOLLAR: $10


        """
        if not kwargs:
            return self
        envs = [{"name": k, "value": v} for k, v in kwargs.items()]
        return self.set_spec(self.CONST_ENV_VAR, envs)

    def with_freeform_tag(self: Self, **kwargs) -> Self:
        """Sets freeform tags

        Returns
        -------
        Self
            This method returns self to support chaining methods.
        """
        return self.set_spec(self.CONST_FREEFORM_TAGS, kwargs)

    def with_defined_tag(self: Self, **kwargs) -> Self:
        """Sets defined tags

        Returns
        -------
        Self
            This method returns self to support chaining methods.
        """
        return self.set_spec(self.CONST_DEFINED_TAGS, kwargs)

    def with_maximum_runtime_in_minutes(
        self: Self, maximum_runtime_in_minutes: int
    ) -> Self:
        """Sets maximum runtime in minutes

        Returns
        -------
        Self
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
        """Freeform tags"""
        return self.get_spec(self.CONST_FREEFORM_TAGS, {})

    @property
    def defined_tags(self) -> dict:
        """Defined tags"""
        return self.get_spec(self.CONST_DEFINED_TAGS, {})

    @property
    def args(self) -> list:
        """Command line arguments"""
        return self.get_spec(self.CONST_ARGS, [])

    @property
    def maximum_runtime_in_minutes(self) -> int:
        """Maximum runtime in minutes"""
        return self.get_spec(self.CONST_MAXIMUM_RUNTIME_IN_MINUTES)

    def init(self, **kwargs) -> Self:
        """Initializes a starter specification for the runtime.

        Returns
        -------
        Self
            This method returns self to support chaining methods.
        """
        return (
            self.with_environment_variable(
                **kwargs.get(self.attribute_map[self.CONST_ENV_VAR], {})
            )
            .with_freeform_tag(
                **kwargs.get(self.attribute_map[self.CONST_FREEFORM_TAGS], {})
            )
            .with_argument(**kwargs.get(self.attribute_map[self.CONST_ARGS], {}))
        )


class MultiNodeRuntime(Runtime):
    """Represents runtime supporting multi-node jobs."""

    CONST_REPLICA = "replicas"

    def with_replica(self, count: int):
        """Specifies the number of nodes (job runs) for the job.

        Parameters
        ----------
        count : int
            Number of nodes (job runs)

        Returns
        -------
        self
            The runtime instance.
        """
        return self.set_spec(self.CONST_REPLICA, count)

    @property
    def replica(self) -> int:
        """The number of nodes (job runs)."""
        return self.get_spec(self.CONST_REPLICA)

    def run(self, dsc_job, **kwargs):
        """Starts the job runs"""
        # For multi-node job, there is no need to create multiple job run.
        if getattr(dsc_job, "job_node_configuration_details", None):
            return dsc_job.run(**kwargs)
        replicas = self.replica if self.replica else 1
        main_run = None
        job_runs = []
        try:
            for i in range(replicas):
                replica_kwargs = kwargs.copy()

                # Only update display name and env vars if replica is specified (not None).
                if self.replica is not None:
                    envs = replica_kwargs.get("environment_variables")
                    if not envs:
                        envs = {}
                    # HuggingFace accelerate requires machine rank
                    # Here we use NODE_RANK to store the machine rank
                    envs["NODE_RANK"] = str(i)
                    envs["NODE_COUNT"] = str(replicas)
                    if main_run:
                        envs["MAIN_JOB_RUN_OCID"] = main_run.id
                    name = replica_kwargs.get("display_name")
                    if not name:
                        name = dsc_job.display_name

                    replica_kwargs["display_name"] = f"{name}-{str(i)}"
                    replica_kwargs["environment_variables"] = envs
                run = dsc_job.run(**replica_kwargs)
                job_runs.append(run)
                if i == 0:
                    main_run = run
        except Exception as ex:
            traceback.print_exc()
            # Wait a few second to avoid the job run being in a transient state.
            time.sleep(2)
            # If there is any error when creating the job runs
            # cancel all the job runs.
            for run in job_runs:
                run.cancel()
            raise ex
        return main_run
