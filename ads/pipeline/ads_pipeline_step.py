#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import copy
from typing import List

from ads.jobs import Job
from ads.jobs.builders.infrastructure.dsc_job import DataScienceJob
from ads.jobs.builders.runtimes.base import Runtime

PIPELINE_STEP_KIND_TO_OCI_MAP = {
    "dataScienceJob": "ML_JOB",
    "customScript": "CUSTOM_SCRIPT",
}

PIPELINE_STEP_KIND_FROM_OCI_MAP = {
    "ML_JOB": "dataScienceJob",
    "CUSTOM_SCRIPT": "customScript",
}

PIPELINE_STEP_KIND = {"dataScienceJob", "customScript"}

PIPELINE_STEP_RESTRICTED_CHAR_SET = {",", ">", "(", ")"}


class PipelineStep(Job):
    """Represents the Data Science Machine Learning Pipeline Step."""

    CONST_NAME = "name"
    CONST_JOB_ID = "jobId"
    CONST_DESCRIPTION = "description"
    CONST_DEPENDS_ON = "dependsOn"
    CONST_KIND = "stepType"
    CONST_MAXIMUM_RUNTIME_IN_MINUTES = "maximumRuntimeInMinutes"
    CONST_ENVIRONMENT_VARIABLES = "environmentVariables"
    CONST_COMMAND_LINE_ARGUMENTS = "commandLineArguments"
    CONST_STEP_INFRA_CONFIG_DETAILS = "stepInfrastructureConfigurationDetails"
    CONST_STEP_CONFIG_DETAILS = "stepConfigurationDetails"
    CONST_INFRASTRUCTURE = "infrastructure"
    CONST_RUNTIME = "runtime"

    def __init__(
        self,
        name: str,
        job_id: str = None,
        infrastructure=None,
        runtime=None,
        description=None,
        maximum_runtime_in_minutes=None,
        environment_variable=None,
        command_line_argument=None,
        kind=None,
    ) -> None:
        """Initialize a pipeline step.

        Parameters
        ----------
        name : str, required
            The name of the pipeline step.
        job_id : str, optional
            The job id of the pipeline step, by default None.
        infrastructure : Infrastructure, optional
            Pipeline step infrastructure, by default None.
        runtime : Runtime, optional
            Pipeline step runtime, by default None.
        description : str, optional
            The description for pipeline step, by default None.
        maximum_runtime_in_minutes : int, optional
            The maximum runtime in minutes for pipeline step, by default None.
        environment_variable : dict, optional
            The environment variable for pipeline step, by default None.
        command_line_argument : str, optional
            The command line argument for pipeline step, by default None.
        kind: str, optional
            The kind of pipeline step.

        Attributes
        ----------
        kind: str
            The kind of the object as showing in YAML.
        name: str
            The name of pipeline step.
        job_id: str
            The job id of pipeline step.
        infrastructure: DataScienceJob
            The infrastructure of pipeline step.
        runtime: Runtime
            The runtime of pipeline step.
        description: str
            The description of pipeline step.
        maximum_runtime_in_minutes: int
            The maximum runtime in minutes of pipeline step.
        environment_variable: dict
            The environment variables of pipeline step.
        argument: str
            The argument of pipeline step.
        depends_on: list
            The depends on of pipeline step.

        Methods
        -------
        with_job_id(self, job_id: str) -> PipelineStep
            Sets the job id for pipeline step.
        with_infrastructure(self, infrastructure) -> PipelineStep
            Sets the infrastructure for pipeline step.
        with_runtime(self, runtime) -> PipelineStep
            Sets the runtime for pipeline step.
        with_description(self, description: str) -> PipelineStep
            Sets the description for pipeline step.
        with_maximum_runtime_in_minutes(self, maximum_runtime_in_minutes: int) -> PipelineStep
            Sets the maximum runtime in minutes for pipeline step.
        with_environment_variable(self, **kwargs) -> PipelineStep
            Sets the environment variables for pipeline step.
        with_argument(self, *args, **kwargs) -> PipelineStep
            Sets the command line arguments for pipeline step.
        with_kind(self, kind: str) -> PipelineStep
            Sets the kind for pipeline step.
        to_dict(self) -> dict
            Serializes the pipeline step specification dictionary.
        from_dict(cls, config: dict) -> PipelineStep
            Initializes a PipelineStep from a dictionary containing the configurations.
        to_yaml(self, uri=None, **kwargs)
            Returns PipelineStep serialized as a YAML string
        from_yaml(cls, yaml_string=None, uri=None, **kwargs)
            Creates an PipelineStep from YAML string provided or from URI location containing YAML string

        Example
        -------
        Here is an example for defining a pipeline step using builder:

        .. code-block:: python

            from ads.pipeline import PipelineStep, CustomScriptStep, ScriptRuntime
            # Define an OCI Data Science pipeline step to run a python script
            pipeline_step = (
                PipelineStep(name="<pipeline_step_name>")
                .with_infrastructure(
                    CustomScriptStep()
                    .with_shape_name("VM.Standard2.1")
                    .with_block_storage_size(50)
                )
                .with_runtime(
                    ScriptRuntime()
                    .with_source("oci://bucket_name@namespace/path/to/script.py")
                    .with_service_conda("tensorflow26_p37_cpu_v2")
                    .with_environment_variable(ENV="value")
                    .with_argument("argument", key="value")
                    .with_maximum_runtime_in_minutes(200)
                )
            )

            # Another way to define an OCI Data Science pipeline step from existing job
            pipeline_step = (
                PipelineStep(name="<pipeline_step_name>")
                .with_job_id("<job_id>")
                .with_description("<description>")
            )

        See Also
        --------
        https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/pipeline/index.html
        """
        self.attribute_set = {
            "jobId",
            "stepType",
            "stepName",
            "description",
            "dependsOn",
            "stepInfrastructureConfigurationDetails",
            "stepConfigurationDetails",
        }

        super().__init__()
        if not name:
            raise ValueError("PipelineStep name must be specified.")
        elif any(char in PIPELINE_STEP_RESTRICTED_CHAR_SET for char in name):
            raise ValueError(
                "PipelineStep name can not include any of the "
                f"restricted characters in "
                f"{''.join(PIPELINE_STEP_RESTRICTED_CHAR_SET)}."
            )
        self.set_spec("name", name)

        if job_id:
            self.with_job_id(job_id)
        elif infrastructure and runtime:
            self.with_infrastructure(infrastructure)
            self.with_runtime(runtime)

        if maximum_runtime_in_minutes:
            self.with_maximum_runtime_in_minutes(maximum_runtime_in_minutes)
        if environment_variable:
            self.with_environment_variable(**environment_variable)
        if command_line_argument:
            self.with_argument(command_line_argument)
        if description:
            self.with_description(description)
        if kind:
            self.with_kind(kind)

    @property
    def name(self) -> str:
        """The name of pipeline step.

        Returns
        -------
        str
            The name of the pipeline step.
        """
        return self.get_spec(self.CONST_NAME)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in YAML.

        Returns
        -------
        str
            The kind of the object as showing in YAML.
        """
        return self.get_spec(self.CONST_KIND)

    @property
    def job_id(self) -> str:
        """The job id of the pipeline step.

        Returns
        -------
        str
            The job id of the pipeline step.
        """
        return self.get_spec(self.CONST_JOB_ID)

    def with_job_id(self, job_id: str) -> "PipelineStep":
        """Sets the job id for pipeline step.

        Parameters
        ----------
        job_id : str
            The job id of pipeline step.

        Returns
        -------
            Pipeline step instance (self).
        """
        if not self.kind:
            self.set_spec(self.CONST_KIND, "ML_JOB")
        return self.set_spec(self.CONST_JOB_ID, job_id)

    @property
    def infrastructure(self) -> "DataScienceJob":
        """The infrastructure of the pipeline step.

        Returns
        -------
        DataScienceJob :
            Data science pipeline step instance.
        """
        return self.get_spec(self.CONST_INFRASTRUCTURE)

    def with_infrastructure(self, infrastructure) -> "PipelineStep":
        """Sets the infrastructure for pipeline step.

        Parameters
        ----------
        infrastructure :
            The infrastructure of pipeline step.

        Returns
        -------
            Pipeline step instance (self).
        """
        if not self.kind:
            self.set_spec(self.CONST_KIND, "CUSTOM_SCRIPT")
        return self.set_spec(self.CONST_INFRASTRUCTURE, infrastructure)

    @property
    def runtime(self) -> "Runtime":
        """The runtime of the pipeline step.

        Returns
        -------
        Runtime :
            Runtime instance.
        """
        return self.get_spec(self.CONST_RUNTIME)

    def with_runtime(self, runtime) -> "PipelineStep":
        """Sets the runtime for pipeline step.

        Parameters
        ----------
        runtime :
            The runtime of pipeline step.

        Returns
        -------
            Pipeline step instance (self).
        """
        if not self.kind:
            self.set_spec(self.CONST_KIND, "CUSTOM_SCRIPT")
        return self.set_spec(self.CONST_RUNTIME, runtime)

    @property
    def description(self) -> str:
        """The description of the pipeline step.

        Returns
        -------
        str
            The description of the pipeline step.
        """
        return self.get_spec(self.CONST_DESCRIPTION)

    def with_description(self, description: str) -> "PipelineStep":
        """Sets the description for pipeline step.

        Parameters
        ----------
        description : str
            The description of pipeline step.

        Returns
        -------
            Pipeline step instance (self).
        """
        return self.set_spec(self.CONST_DESCRIPTION, description)

    @property
    def maximum_runtime_in_minutes(self) -> int:
        """The maximum runtime in minutes of pipeline step.

        Returns
        -------
        int
            The maximum runtime in minutes of the pipeline step.
        """
        return self.get_spec(self.CONST_MAXIMUM_RUNTIME_IN_MINUTES)

    def with_maximum_runtime_in_minutes(
        self, maximum_runtime_in_minutes: int
    ) -> "PipelineStep":
        """Sets the maximum runtime in minutes of pipeline step.

        Parameters
        ----------
        maximum_runtime_in_minutes : int
            The maximum runtime in minutes of pipeline step.

        Returns
        -------
            Pipeline step instance (self).
        """
        return self.set_spec(
            self.CONST_MAXIMUM_RUNTIME_IN_MINUTES, maximum_runtime_in_minutes
        )

    @property
    def environment_variable(self) -> dict:
        """The environment variables of the pipeline step.

        Returns
        -------
        dict:
            The environment variables of the pipeline step.
        """
        return self.get_spec(self.CONST_ENVIRONMENT_VARIABLES)

    def with_environment_variable(self, **kwargs) -> "PipelineStep":
        """Sets environment variables of the pipeline step.

        Parameters
        ----------
        kwargs:
            Keyword arguments.
            To add a keyword argument without value, set the value to None.

        Returns
        -------
        Pipeline
            The Pipeline step instance (self).
        """
        if kwargs:
            environment_variable_dict = {}
            for k, v in kwargs.items():
                environment_variable_dict[k] = v
            self.set_spec(self.CONST_ENVIRONMENT_VARIABLES, environment_variable_dict)
        return self

    @property
    def argument(self) -> str:
        """The command line arguments of the pipeline step.

        Returns
        -------
        str:
            The command line arguments of the pipeline step.
        """
        return self.get_spec(self.CONST_COMMAND_LINE_ARGUMENTS)

    def with_argument(self, *args, **kwargs) -> "PipelineStep":
        """Adds command line arguments to the pipeline step.
        Existing arguments will be preserved.
        This method can be called (chained) multiple times to add various arguments.
        For example, pipeline.with_argument(key="val").with_argument("path/to/file") will result in:
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
        Pipeline
            The Pipeline step instance (self).

        Raises
        ------
        ValueError
            Keyword arguments with space in a key.
        """
        arg_values = self.get_spec(self.CONST_COMMAND_LINE_ARGUMENTS, [])
        args = [str(arg) for arg in args]
        arg_values.extend(args)
        for k, v in kwargs.items():
            if " " in k:
                raise ValueError("Argument key %s cannot contain space.", str(k))
            arg_values.append(f"--{str(k)}")
            # Ignore None value
            if v is None:
                continue
            arg_values.append(str(v))
        arg_string = " ".join(arg_values)
        self.set_spec(self.CONST_COMMAND_LINE_ARGUMENTS, arg_string)
        return self

    @property
    def depends_on(self) -> list:
        """The list of upstream pipeline steps for (self).

        Returns
        -------
        list
            The list of upstream pipeline steps for (self).
        """
        return self.get_spec(self.CONST_DEPENDS_ON)

    def _with_depends_on(self, depends_on: List["PipelineStep"]) -> "PipelineStep":
        """Sets the list of upstream pipeline steps for (self).

        Parameters
        ----------
        depends_on : list of PipelineStep objects
            The list of pipeline steps that (self) depends on.

        Returns
        -------
            Pipeline step instance (self).
        """
        if not depends_on:
            return self.set_spec(self.CONST_DEPENDS_ON, [])

        step_list = []
        for step in depends_on:
            step_list.append(step.name)
        return self.set_spec(self.CONST_DEPENDS_ON, step_list)

    def with_kind(self, kind: str) -> "PipelineStep":
        """Sets the kind of pipeline step.

        Parameters
        ----------
        kind : str
            The kind of pipeline step.

        Returns
        -------
            Pipeline step instance (self).
        """
        if kind in PIPELINE_STEP_KIND:
            self.set_spec(self.CONST_KIND, PIPELINE_STEP_KIND_TO_OCI_MAP[kind])
        else:
            raise ValueError(
                "Invalid PipelineStep kind. The allowed "
                f"values are {', '.join(PIPELINE_STEP_KIND)}."
            )
        return self.set_spec(self.CONST_KIND, kind)

    def to_dict(self) -> dict:
        """Serializes the pipeline step specification dictionary.

        Returns
        -------
        dict
            A dictionary containing pipeline step specification.
        """
        dict_details = copy.deepcopy(super().to_dict())
        if self.kind in PIPELINE_STEP_KIND_FROM_OCI_MAP:
            dict_details["kind"] = PIPELINE_STEP_KIND_FROM_OCI_MAP[self.kind]

        # remove information not going to show in to_dict()
        if self.CONST_INFRASTRUCTURE in dict_details["spec"]:
            dict_details["spec"][self.CONST_INFRASTRUCTURE].pop("type", None)
            dict_details["spec"][self.CONST_INFRASTRUCTURE]["spec"].pop(
                "jobInfrastructureType", None
            )
            dict_details["spec"][self.CONST_INFRASTRUCTURE]["spec"].pop("jobType", None)

        if self.job_id:
            dict_details["spec"][self.CONST_JOB_ID] = self.job_id
        if self.description:
            dict_details["spec"][self.CONST_DESCRIPTION] = self.description

        dict_details["spec"].pop(self.CONST_DEPENDS_ON, None)

        return dict_details

    @classmethod
    def from_dict(cls, config: dict) -> "PipelineStep":
        """Initializes a PipelineStep from a dictionary containing the configurations.

        Parameters
        ----------
        config : dict
            A dictionary containing the infrastructure and runtime specifications.

        Returns
        -------
        PipelineStep
            A PipelineStep instance

        Raises
        ------
        NotImplementedError
            If the type of the intrastructure or runtime is not supported.
        """
        if not isinstance(config, dict):
            raise ValueError("The config data for initializing the job is invalid.")
        spec = config.get("spec")

        mappings = {
            "infrastructure": cls._INFRASTRUCTURE_MAPPING,
            "runtime": cls._RUNTIME_MAPPING,
        }
        if spec["name"]:
            pipeline_step = cls(name=spec["name"])
        else:
            raise ValueError("PipelineStep name must be specified.")

        if config.get("kind", None):
            step_kind = config.get("kind")

            if step_kind in PIPELINE_STEP_KIND:
                pipeline_step.set_spec(
                    cls.CONST_KIND, PIPELINE_STEP_KIND_TO_OCI_MAP[step_kind]
                )
            else:
                raise ValueError(
                    "Invalid PipelineStep kind. The allowed "
                    f"values are {', '.join(PIPELINE_STEP_KIND)}."
                )
        else:
            pipeline_step.set_spec(cls.CONST_KIND, "ML_JOB")

        for key, value in spec.items():
            if key in mappings:
                mapping = mappings[key]
                child_config = copy.deepcopy(value)
                if key == "infrastructure":
                    child_config["type"] = "dataScienceJob"
                if child_config.get("type") not in mapping:
                    raise NotImplementedError(
                        f"{key.title()} type: {child_config.get('type')} is not supported."
                    )
                pipeline_step.set_spec(
                    key, mapping[child_config.get("type")].from_dict(child_config)
                )
            elif key == cls.CONST_STEP_CONFIG_DETAILS:
                for attribute in value:
                    pipeline_step.set_spec(attribute, value[attribute])
            else:
                pipeline_step.set_spec(key, value)

        return pipeline_step
