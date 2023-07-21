#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Contains classes for conversion between ADS runtime and OCI Data Science Job implementation.
This module is for ADS developers only.
In this module, a payload is defined as a dictionary for initializing a DSCJob object.
The DSCJob can be initialized with the same arguments for initializing oci.data_science.models.Job,
    plus an "artifact" argument for job artifact.
The payload also contain infrastructure information.
The conversion from a runtime to a payload is called translate in this module.
The conversion from a DSCJob to a runtime is called extract in this module.
"""
from __future__ import annotations

import json
import os
import shlex
from typing import Optional
from urllib import parse
from ads.common.utils import extract_region
from ads.jobs.builders.runtimes.base import Runtime
from ads.jobs.builders.runtimes.python_runtime import (
    CondaRuntime,
    ScriptRuntime,
    PythonRuntime,
    NotebookRuntime,
    GitPythonRuntime,
)
from ads.jobs.builders.runtimes.container_runtime import ContainerRuntime
from ads.jobs.builders.runtimes.pytorch_runtime import (
    PyTorchDistributedRuntime,
    PyTorchDistributedArtifact,
)
from ads.jobs.builders.runtimes.artifact import (
    ScriptArtifact,
    NotebookArtifact,
    PythonArtifact,
    GitPythonArtifact,
)
from ads.opctl.distributed.common import cluster_config_helper
from ads.jobs.builders.infrastructure.utils import get_value
from ads.jobs.templates import driver_utils


class IncompatibleRuntime(Exception):  # pragma: no cover
    """Represents an exception when runtime is not compatible with the OCI data science job configuration.
    This exception is designed to be raised during the extraction of a runtime from OCI data science job.
    The data science job does not explicitly contain information of the type of the ADS runtime.
    Each runtime handler should determine if the configuration of the job can be converted to the runtime.
    This exception should be raised during the extract() call if the configuration cannot be converted.
    The RuntimeManager uses this exception to determine if the conversion is successful.
    """


class RuntimeHandler:
    """Base class for Runtime Handler.

    Each runtime handler should define the RUNTIME_CLASS to be the runtime it can handle.

    Each runtime handler is initialized with a DataScienceJob instance.
    This instance is a reference and the modification will be exposed to the users.

    Each runtime handler expose two methods: translate() and extract().
    In this class, translate or extract signals the direction of conversion.
    All method starts with "translate" handles the conversion from ADS runtime to OCI API payload.
    All method starts with "extract" handles the conversion from OCI data science Job to ADS runtime.
    This base class defines the default handling for translate() and extract().
    Each sub-class can override the two methods to provide additional handling.
    Alternatively, a sub-class can also override a sub-method, which is called by the translate() or extract() method.
    For example, _translate_env() handles the conversion of environment variables from ADS runtime to OCI API payload.

    See the individual methods for more details.
    """

    # Defines the class of the runtime to be handled.
    RUNTIME_CLASS = Runtime

    def __init__(self, data_science_job) -> None:
        """Initialize the runtime handler.

        Parameters
        ----------
        data_science_job : DataScienceJob
            An instance of the DataScienceJob to be created or extracted from.
        """
        self.data_science_job = data_science_job

    def translate(self, runtime: Runtime) -> dict:
        """Translates the runtime into a JSON payload for OCI API.
        This method calls the following sub-methods:
        * _translate_artifact()
        * _translate_config()
          * _translate_env()
        A sub-class can modify one of more of these methods.

        Parameters
        ----------
        runtime : Runtime
            An instance of the runtime to be converted to a JSON payload.

        Returns
        -------
        dict
            JSON payload for defining a Data Science Job with OCI API
        """
        payload = {}
        payload["artifact"] = self._translate_artifact(runtime)
        payload["job_configuration_details"] = self._translate_config(runtime)
        if runtime.freeform_tags:
            payload["freeform_tags"] = runtime.freeform_tags
        if runtime.defined_tags:
            payload["defined_tags"] = runtime.defined_tags
        self.data_science_job.runtime = runtime
        return payload

    def _translate_artifact(self, runtime: Runtime):
        """Translate the runtime artifact.
        OCI data science requires an artifact file to be uploaded before the job is created.
        For Python runtime, the artifact is the script for running the job.
        For container runtime, the artifact is not actually used.
        For notebook runtime, ADS needs to convert the artifact to Python script before uploading.

        Parameters
        ----------
        runtime : Runtime
            An instance of the runtime

        Returns
        -------
        str or Artifact
            The artifact that is ready to be used by DSCJob.
            This can either be a string storing the path the artifact file,
            or an instance of Artifact class, which contains logic for additional processing.
        """
        return None

    def _translate_env(self, runtime: Runtime) -> dict:
        """Translate the environment variable.

        OCI Data Science job uses environment variables for various settings.
        These settings are properties in ADS runtime.
        This method is designed to handle the conversion of the ADS runtime properties to environment variables.
        By default, no conversion is made in this method.
        Sub-class should override this method to add conversion logic.

        Parameters
        ----------
        runtime : Runtime
            An instance of the runtime

        Returns
        -------
        dict
            A dictionary storing the environment variable for OCI data science job.
        """
        return runtime.envs

    def _translate_config(self, runtime: Runtime) -> dict:
        """Prepares the job configuration from runtime specifications.

        Parameters
        ----------
        runtime : Runtime
            An instance of the runtime

        Returns
        -------
        dict
            A dictionary for OCI data science job configuration.
            The dictionary may have the following keys:
                "jobType"
                "commandLineArguments"
                "environmentVariables"
                "maximumRuntimeInMinutes"
            The configurations will be used to initialize the DSCJob instance.
            The DSCJob class can handle keys in either camel or snake format.
        """
        job_configuration_details = {
            "jobType": self.data_science_job.job_type,
        }
        if runtime.maximum_runtime_in_minutes:
            job_configuration_details[
                "maximum_runtime_in_minutes"
            ] = runtime.maximum_runtime_in_minutes
        job_configuration_details["environment_variables"] = self._translate_env(
            runtime
        )
        if runtime.args:
            # shlex.join() is not available until python 3.8
            job_configuration_details["command_line_arguments"] = " ".join(
                shlex.quote(str(arg)) for arg in runtime.get_spec(runtime.CONST_ARGS)
            )
        return job_configuration_details

    @staticmethod
    def _translate_specs(
        runtime: Runtime, spec_mappings: dict, delimiter: Optional[str] = None
    ) -> dict:
        """Converts runtime properties to OCI data science job environment variables based on a mapping.

        Parameters
        ----------
        runtime : Runtime
            The runtime containing the properties to be converted.
        spec_mappings : dict
            Mapping from runtime properties to environment variables.
            Each key is a specification key (property name) of a runtime
            Each value is the corresponding name of the environment variable in OCI data science job.
        delimiter : str, Optional
            Environment variables must be strings.
            For list or tuple, specify the delimiter for joining the values into a string.

        Returns
        -------
        dict
            A dictionary containing environment variables for OCI data science job.
        """
        envs = {}
        for spec_key, dsc_key in spec_mappings.items():
            val = runtime.get_spec(spec_key)
            if val:
                if delimiter and isinstance(val, list) or isinstance(val, tuple):
                    val = delimiter.join(val)
                envs[dsc_key] = val
        return envs

    @staticmethod
    def _extract_specs(envs: dict, spec_mappings: dict) -> dict:
        """Converts the environment variables in OCI data science job to runtime properties.

        Parameters
        ----------
        envs : dict
            A dictionary containing environment variables from OCI data science job.
        spec_mappings : dict
            Mapping from runtime properties to environment variables.
            This mapping is the same as the one in _translate_spec().

        This method does not convert strings into list or tuple as there is no way to identify them.

        Returns
        -------
        dict
            A dictionary for specifying the runtime.
        """
        spec = {}
        for spec_key, dsc_key in spec_mappings.items():
            val = envs.pop(dsc_key, None)
            if val:
                spec[spec_key] = val
        return spec

    @staticmethod
    def _format_env_var(runtime_spec: dict) -> dict:
        """Formats the environment variables in runtime specification (as dict) from a dictionary to list.
        The list of environment variables uses the same format as environment variables in Kubernetes.

        Parameters
        ----------
        runtime_spec : dict
            Runtime specification in a dictionary.
            This is the dictionary that can be used to initialize a runtime instance.
            Except that environment variables are stored in a dict instead of list.

        Returns
        -------
        dict
            Runtime specification with environment variables stored in a list.
        """
        env_var = runtime_spec.pop(Runtime.CONST_ENV_VAR, None)
        if env_var and isinstance(env_var, dict):
            runtime_spec[Runtime.CONST_ENV_VAR] = [
                {"name": k, "value": v} for k, v in env_var.items()
            ]
        return runtime_spec

    def extract(self, dsc_job):
        """Extract the runtime from an OCI data science job object.
        This method calls the following sub-methods:
        * _extract_tags()
        * _extract_args()
        * _extract_envs()
        * _extract_artifact()
        * _extract_runtime_minutes()
        Each of these method returns a dict for specifying the runtime.
        The dictionaries are combined before initalizing the runtime.
        A sub-class can modify one of more of these methods.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        Runtime:
            The runtime extracted from the data science job.
        """
        runtime_spec = {}
        extractions = [
            self._extract_tags,
            self._extract_args,
            self._extract_envs,
            self._extract_artifact,
            self._extract_runtime_minutes,
        ]
        for extraction in extractions:
            runtime_spec.update(extraction(dsc_job))
        return self.RUNTIME_CLASS(self._format_env_var(runtime_spec))

    def _extract_args(self, dsc_job) -> dict:
        """Extracts the command line arguments from data science job.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        args_string = get_value(
            dsc_job, "job_configuration_details.command_line_arguments"
        )
        if args_string:
            return {Runtime.CONST_ARGS: shlex.split(args_string)}
        return {}

    def _extract_envs(self, dsc_job):
        """Extract the environment variables from data science job.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        envs = get_value(dsc_job, "job_configuration_details.environment_variables")
        if envs:
            return {Runtime.CONST_ENV_VAR: envs}
        return {}

    def _extract_tags(self, dsc_job):
        """Extract the freeform tags from data science job.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        tags = {}
        value = get_value(dsc_job, "freeform_tags")
        if value:
            tags[Runtime.CONST_FREEFORM_TAGS] = value
        value = get_value(dsc_job, "defined_tags")
        if value:
            tags[Runtime.CONST_DEFINED_TAGS] = value
        return tags

    def _extract_artifact(self, dsc_job):
        """Extract the job artifact from data science job.

        This is the base method which does not extract the job artifact.
        Sub-class should implement the extraction if needed.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        return {}

    def _extract_runtime_minutes(self, dsc_job):
        """Extract the maximum runtime in minutes from data science job.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        maximum_runtime_in_minutes = get_value(
            dsc_job, "job_configuration_details.maximum_runtime_in_minutes"
        )
        if maximum_runtime_in_minutes:
            return {
                Runtime.CONST_MAXIMUM_RUNTIME_IN_MINUTES: maximum_runtime_in_minutes
            }
        return {}


class CondaRuntimeHandler(RuntimeHandler):
    """Runtime Handler for CondaRuntime"""

    RUNTIME_CLASS = CondaRuntime
    CONST_CONDA_TYPE = "CONDA_ENV_TYPE"
    CONST_CONDA_SLUG = "CONDA_ENV_SLUG"
    CONST_CONDA_OBJ_NAME = "CONDA_ENV_OBJECT_NAME"
    CONST_CONDA_REGION = "CONDA_ENV_REGION"
    CONST_CONDA_NAMESPACE = "CONDA_ENV_NAMESPACE"
    CONST_CONDA_BUCKET = "CONDA_ENV_BUCKET"

    def __get_auth_region(self) -> str:
        return extract_region(self.data_science_job.dsc_job.auth)

    def _translate_env(self, runtime: CondaRuntime) -> dict:
        """Translate the environment variable.

        Parameters
        ----------
        runtime : CondaRuntime
            An instance of CondaRuntime

        Returns
        -------
        dict
            A dictionary containing environment variables for OCI data science job.
        """
        envs = super()._translate_env(runtime)
        if runtime.conda:
            envs[self.CONST_CONDA_TYPE] = runtime.conda.get(
                CondaRuntime.CONST_CONDA_TYPE
            )
            if (
                runtime.conda.get(CondaRuntime.CONST_CONDA_TYPE)
                == CondaRuntime.CONST_CONDA_TYPE_SERVICE
            ):
                envs.update(
                    {
                        self.CONST_CONDA_SLUG: runtime.conda.get(
                            CondaRuntime.CONST_CONDA_SLUG
                        ),
                    }
                )
            elif (
                runtime.conda.get(CondaRuntime.CONST_CONDA_TYPE)
                == CondaRuntime.CONST_CONDA_TYPE_CUSTOM
            ):
                uri = runtime.conda.get(CondaRuntime.CONST_CONDA_URI)
                p = parse.urlparse(uri)
                if not (p.username and p.hostname and p.path):
                    raise ValueError(
                        f"Invalid URI for custom conda pack: {uri}. "
                        "A valid URI should have the format: oci://your_bucket@namespace/object_name"
                    )
                region = runtime.conda.get(CondaRuntime.CONST_CONDA_REGION)
                if not region:
                    region = self.__get_auth_region()
                if not region:
                    raise AttributeError(
                        "Unable to determine the region for the custom conda pack. "
                        "Specify the region using with_custom_conda(uri, region)."
                    )
                envs.update(
                    {
                        self.CONST_CONDA_NAMESPACE: p.hostname,
                        self.CONST_CONDA_BUCKET: p.username,
                        self.CONST_CONDA_OBJ_NAME: p.path.lstrip("/"),
                        self.CONST_CONDA_REGION: region,
                    }
                )
        return envs

    def _extract_envs(self, dsc_job) -> dict:
        """Extract the environment variables from data science job.
        CondaRuntime contains environment variables for specifying conda environment.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        spec = super()._extract_envs(dsc_job)
        envs = spec.pop(CondaRuntime.CONST_ENV_VAR, {})
        conda_spec = self.__extract_conda_env(envs)
        if conda_spec:
            spec[CondaRuntime.CONST_CONDA] = conda_spec
        if envs:
            spec[CondaRuntime.CONST_ENV_VAR] = envs
        return spec

    @staticmethod
    def __extract_conda_env(envs: dict) -> Optional[dict]:
        """Extracts conda pack specification from environment variables

        Parameters
        ----------
        envs : dict
            Environment variables.

        Returns
        -------
        Optional[dict]
            Conda pack runtime specification.
        """
        if not envs:
            return None
        if (
            CondaRuntimeHandler.CONST_CONDA_TYPE in envs
            and CondaRuntimeHandler.CONST_CONDA_SLUG in envs
        ):
            return {
                CondaRuntime.CONST_CONDA_TYPE: envs.pop(
                    CondaRuntimeHandler.CONST_CONDA_TYPE
                ),
                CondaRuntime.CONST_CONDA_SLUG: envs.pop(
                    CondaRuntimeHandler.CONST_CONDA_SLUG
                ),
            }
        if (
            envs.get(CondaRuntimeHandler.CONST_CONDA_TYPE)
            == CondaRuntime.CONST_CONDA_TYPE_CUSTOM
            and CondaRuntimeHandler.CONST_CONDA_BUCKET in envs
            and CondaRuntimeHandler.CONST_CONDA_BUCKET in envs
            and CondaRuntimeHandler.CONST_CONDA_OBJ_NAME in envs
        ):
            bucket = envs.pop(CondaRuntimeHandler.CONST_CONDA_BUCKET)
            namespace = envs.pop(CondaRuntimeHandler.CONST_CONDA_NAMESPACE)
            name = envs.pop(CondaRuntimeHandler.CONST_CONDA_OBJ_NAME)
            conda_spec = {
                CondaRuntime.CONST_CONDA_TYPE: envs.pop(
                    CondaRuntimeHandler.CONST_CONDA_TYPE
                ),
                CondaRuntime.CONST_CONDA_URI: f"oci://{bucket}@{namespace}/{name}",
            }
            if CondaRuntimeHandler.CONST_CONDA_REGION in envs:
                conda_spec[CondaRuntime.CONST_CONDA_REGION] = envs.pop(
                    CondaRuntimeHandler.CONST_CONDA_REGION
                )
            return conda_spec
        return None


class ScriptRuntimeHandler(CondaRuntimeHandler):
    """Runtime Handler for ScriptRuntime"""

    RUNTIME_CLASS = ScriptRuntime
    CONST_ENTRYPOINT = "JOB_RUN_ENTRYPOINT"

    def _translate_env(self, runtime: ScriptRuntime) -> dict:
        """Translate the environment variable.

        Parameters
        ----------
        runtime : ScriptRuntime
            An instance of ScriptRuntime

        Returns
        -------
        dict
            A dictionary contianing environment variables for OCI data science job.
        """
        envs = super()._translate_env(runtime)
        if runtime.entrypoint:
            envs[self.CONST_ENTRYPOINT] = runtime.entrypoint
        return envs

    def _translate_artifact(self, runtime: ScriptRuntime):
        return ScriptArtifact(runtime.source_uri, runtime)

    def _extract_envs(self, dsc_job) -> dict:
        """Extract the environment variables from data science job.
        ScriptRuntime may contain entrypoint as environment variable in addition to those for conda environment.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        spec = super()._extract_envs(dsc_job)
        envs = spec.pop(ScriptRuntime.CONST_ENV_VAR, {})
        entrypoint = envs.pop(ScriptRuntimeHandler.CONST_ENTRYPOINT, None)
        if entrypoint:
            spec[ScriptRuntime.CONST_ENTRYPOINT] = entrypoint
        if envs:
            spec[ScriptRuntime.CONST_ENV_VAR] = envs
        return spec

    def _extract_artifact(self, dsc_job):
        """Extract the job artifact from data science job.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        spec = super()._extract_artifact(dsc_job)
        spec.update({ScriptRuntime.CONST_SCRIPT_PATH: str(dsc_job.artifact)})
        return spec


class PythonRuntimeHandler(CondaRuntimeHandler):
    """Runtime Handler for PythonRuntime"""

    RUNTIME_CLASS = PythonRuntime
    PATH_DELIMITER = ":"
    CONST_JOB_ENTRYPOINT = "JOB_RUN_ENTRYPOINT"
    CONST_CODE_ENTRYPOINT = "CODE_ENTRYPOINT"
    CONST_ENTRY_FUNCTION = "ENTRY_FUNCTION"
    CONST_PYTHON_PATH = "PYTHON_PATH"
    CONST_OUTPUT_DIR = "OUTPUT_DIR"
    CONST_OUTPUT_URI = "OUTPUT_URI"
    CONST_WORKING_DIR = "WORKING_DIR"

    SPEC_MAPPINGS = {
        PythonRuntime.CONST_ENTRYPOINT: CONST_CODE_ENTRYPOINT,
        PythonRuntime.CONST_ENTRY_FUNCTION: CONST_ENTRY_FUNCTION,
        PythonRuntime.CONST_PYTHON_PATH: CONST_PYTHON_PATH,
        PythonRuntime.CONST_OUTPUT_DIR: CONST_OUTPUT_DIR,
        PythonRuntime.CONST_OUTPUT_URI: CONST_OUTPUT_URI,
        PythonRuntime.CONST_WORKING_DIR: CONST_WORKING_DIR,
    }

    def _translate_artifact(self, runtime: PythonRuntime):
        return PythonArtifact(runtime.script_uri, runtime)

    def _translate_env(self, runtime: PythonRuntime) -> dict:
        envs = super()._translate_env(runtime)
        envs.update(
            self._translate_specs(runtime, self.SPEC_MAPPINGS, self.PATH_DELIMITER)
        )

        if runtime.entrypoint:
            envs[self.CONST_CODE_ENTRYPOINT] = runtime.entrypoint
        elif runtime.script_uri:
            envs[self.CONST_CODE_ENTRYPOINT] = os.path.basename(runtime.script_uri)

        envs[self.CONST_JOB_ENTRYPOINT] = PythonArtifact.CONST_DRIVER_SCRIPT
        return envs

    def _extract_envs(self, dsc_job) -> dict:
        """Extract the runtime specification from environment variables.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        spec = super()._extract_envs(dsc_job)
        envs = spec.pop(PythonRuntime.CONST_ENV_VAR, {})
        if (
            self.__class__ == PythonRuntimeHandler
            and self.CONST_CODE_ENTRYPOINT not in envs
        ):
            raise IncompatibleRuntime()
        # PyTorchDistributedRuntime does not require entrypoint.
        envs.pop(PythonRuntimeHandler.CONST_JOB_ENTRYPOINT, None)
        spec.update(self._extract_specs(envs, self.SPEC_MAPPINGS))
        if PythonRuntime.CONST_PYTHON_PATH in spec:
            spec[PythonRuntime.CONST_PYTHON_PATH] = spec[
                PythonRuntime.CONST_PYTHON_PATH
            ].split(self.PATH_DELIMITER)
        if envs:
            spec[PythonRuntime.CONST_ENV_VAR] = envs
        return spec

    def _extract_artifact(self, dsc_job):
        """Extract the job artifact from data science job.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        spec = super()._extract_artifact(dsc_job)
        # It is not possible to get the actual script path
        # since the information is not stored in the job.
        # Here we only extract the name of the artifact.
        spec.update(
            {
                PythonRuntime.CONST_SCRIPT_PATH: os.path.splitext(
                    str(dsc_job.artifact)
                )[0]
            }
        )
        return spec


class NotebookRuntimeHandler(CondaRuntimeHandler):
    """Runtime Handler for NotebookRuntime"""

    RUNTIME_CLASS = NotebookRuntime
    CONST_NOTEBOOK_NAME = "JOB_RUN_NOTEBOOK"
    CONST_ENTRYPOINT = "JOB_RUN_ENTRYPOINT"
    CONST_OUTPUT_URI = "OUTPUT_URI"
    CONST_EXCLUDE_TAGS = "NOTEBOOK_EXCLUDE_TAGS"
    CONST_NOTEBOOK_ENCODING = "NOTEBOOK_ENCODING"

    SPEC_MAPPINGS = {
        NotebookRuntime.CONST_OUTPUT_URI: CONST_OUTPUT_URI,
        NotebookRuntime.CONST_EXCLUDE_TAG: CONST_EXCLUDE_TAGS,
        NotebookRuntime.CONST_NOTEBOOK_ENCODING: CONST_NOTEBOOK_ENCODING,
    }

    def _translate_artifact(self, runtime: NotebookRuntime):
        source = runtime.source if runtime.source else runtime.notebook_uri
        return NotebookArtifact(source, runtime)

    def _translate_env(self, runtime: NotebookRuntime) -> dict:
        envs = super()._translate_env(runtime)

        if runtime.notebook:
            # runtime.notebook should always be a relative path from the root of the source.
            # In NotebookArtifact, when zipping the files,
            # a top level folder having the same name as the basename of runtime.source
            # is used to contain all the user artifacts.
            # The basename of runtime.source will also be used as the name of the artifact zip file.
            envs[self.CONST_NOTEBOOK_NAME] = os.path.join(
                os.path.basename(runtime.source), runtime.notebook
            )
        elif runtime.notebook_uri:
            # For running a single notebook.
            envs[self.CONST_NOTEBOOK_NAME] = os.path.basename(runtime.notebook_uri)
        else:
            raise ValueError(
                "Notebook not specified. "
                "Please specify the notebook using with_notebook_uri() or with_source() method."
            )

        envs[self.CONST_ENTRYPOINT] = NotebookArtifact.CONST_DRIVER_SCRIPT
        if runtime.notebook_encoding:
            envs[self.CONST_NOTEBOOK_ENCODING] = runtime.notebook_encoding
        if runtime.exclude_tag:
            envs[self.CONST_EXCLUDE_TAGS] = json.dumps(runtime.exclude_tag)
        if runtime.output_uri:
            envs[self.CONST_OUTPUT_URI] = runtime.output_uri
        return envs

    def _extract_envs(self, dsc_job) -> dict:
        """Extract the runtime specification from environment variables.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        spec = super()._extract_envs(dsc_job)
        envs = spec.pop(NotebookRuntime.CONST_ENV_VAR, {})
        if not (self.CONST_NOTEBOOK_NAME in envs and self.CONST_ENTRYPOINT in envs):
            raise IncompatibleRuntime()
        # Remove job run entrypoint since it is the same for notebook runtime.
        envs.pop(self.CONST_ENTRYPOINT)
        # Extract exclude tags
        exclude_tags = envs.pop(self.CONST_EXCLUDE_TAGS, None)
        if exclude_tags:
            # Exclude tags are in a JSON serialized string
            try:
                exclude_tags = json.loads(exclude_tags)
            except ValueError:
                # Ignore de-serialization error
                pass
            spec[NotebookRuntime.CONST_EXCLUDE_TAG] = exclude_tags

        # Extract notebook name
        notebook = envs.pop(self.CONST_NOTEBOOK_NAME)
        if "/" in notebook:
            # This indicate notebook is uploaded as part of a folder/zip
            # When the source is a folder, the notebook name will have the format of
            # folder/path/to/notebook.ipynb
            (
                spec[NotebookRuntime.CONST_SOURCE],
                spec[NotebookRuntime.CONST_ENTRYPOINT],
            ) = str(notebook).split("/", 1)
        else:
            # When the source is a single notebook, the notebook name will be the filename only.
            # notebook.ipynb
            spec[NotebookRuntime.CONST_NOTEBOOK_PATH] = notebook

        spec.update(self._extract_specs(envs, self.SPEC_MAPPINGS))
        spec[NotebookRuntime.CONST_ENV_VAR] = envs
        return spec


class GitPythonRuntimeHandler(CondaRuntimeHandler):
    """Runtime Handler for GitPythonRuntime"""

    RUNTIME_CLASS = GitPythonRuntime

    PATH_DELIMITER = ":"
    CONST_GIT_URL = "GIT_URL"
    CONST_GIT_BRANCH = "GIT_BRANCH"
    CONST_GIT_COMMIT = "GIT_COMMIT"
    CONST_GIT_CODE_DIR = "CODE_DIR"
    CONST_GIT_SSH_SECRET_ID = "GIT_SECRET_OCID"
    CONST_SKIP_METADATA = "SKIP_METADATA_UPDATE"

    CONST_ENTRYPOINT = "GIT_ENTRYPOINT"
    CONST_ENTRY_FUNCTION = "ENTRY_FUNCTION"
    CONST_PYTHON_PATH = "PYTHON_PATH"
    CONST_OUTPUT_DIR = "OUTPUT_DIR"
    CONST_OUTPUT_URI = "OUTPUT_URI"
    CONST_WORKING_DIR = "WORKING_DIR"

    CONST_JOB_ENTRYPOINT = "JOB_RUN_ENTRYPOINT"

    SPEC_MAPPINGS = {
        GitPythonRuntime.CONST_GIT_URL: CONST_GIT_URL,
        GitPythonRuntime.CONST_BRANCH: CONST_GIT_BRANCH,
        GitPythonRuntime.CONST_COMMIT: CONST_GIT_COMMIT,
        GitPythonRuntime.CONST_ENTRYPOINT: CONST_ENTRYPOINT,
        GitPythonRuntime.CONST_ENTRY_FUNCTION: CONST_ENTRY_FUNCTION,
        GitPythonRuntime.CONST_PYTHON_PATH: CONST_PYTHON_PATH,
        GitPythonRuntime.CONST_GIT_SSH_SECRET_ID: CONST_GIT_SSH_SECRET_ID,
        GitPythonRuntime.CONST_OUTPUT_DIR: CONST_OUTPUT_DIR,
        GitPythonRuntime.CONST_OUTPUT_URI: CONST_OUTPUT_URI,
        GitPythonRuntime.CONST_WORKING_DIR: CONST_WORKING_DIR,
    }

    def _translate_artifact(self, runtime: Runtime):
        """Specifies the driver script as the job artifact.
        runtime is not used in this method.

        Parameters
        ----------
        runtime : Runtime
            This is not used.

        Returns
        -------
        str
            Path to the git driver script.
        """
        return GitPythonArtifact()

    def _translate_env(self, runtime: GitPythonRuntime) -> dict:
        """Translate the environment variable.

        Parameters
        ----------
        runtime : GitPythonRuntime
            An instance of GitPythonRuntime

        Returns
        -------
        dict
            A dictionary containing environment variables for OCI data science job.
        """
        if not runtime.conda:
            raise ValueError(
                f"A conda pack is required for using the {runtime.__class__.__name__}. "
                "You can specify a service conda pack using with_service_conda()."
            )
        envs = super()._translate_env(runtime)
        envs.update(
            self._translate_specs(runtime, self.SPEC_MAPPINGS, self.PATH_DELIMITER)
        )
        if runtime.skip_metadata_update:
            envs[self.CONST_SKIP_METADATA] = "1"
        # Add entrypoint as the ADS driver is packed in a zip file.
        envs[self.CONST_JOB_ENTRYPOINT] = GitPythonArtifact.CONST_DRIVER_SCRIPT
        return envs

    def _extract_envs(self, dsc_job) -> dict:
        """Extract the environment variables from data science job.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        spec = super()._extract_envs(dsc_job)
        envs = spec.pop(CondaRuntime.CONST_ENV_VAR, {})

        if self.CONST_GIT_URL not in envs or self.CONST_ENTRYPOINT not in envs:
            raise IncompatibleRuntime()
        # Remove entrypoint as it's added by ADS
        envs.pop(self.CONST_JOB_ENTRYPOINT, None)
        spec.update(self._extract_specs(envs, self.SPEC_MAPPINGS))
        if GitPythonRuntime.CONST_PYTHON_PATH in spec:
            spec[GitPythonRuntime.CONST_PYTHON_PATH] = spec[
                GitPythonRuntime.CONST_PYTHON_PATH
            ].split(self.PATH_DELIMITER)
        if self.CONST_SKIP_METADATA in envs:
            envs.pop(self.CONST_SKIP_METADATA, None)
            spec[GitPythonRuntime.CONST_SKIP_METADATA] = True
        if envs:
            spec[ScriptRuntime.CONST_ENV_VAR] = envs
        return spec

    def _extract_artifact(self, dsc_job):
        """Git runtime uses the driver script as artifact. This will not be extracted."""
        return {}


class ContainerRuntimeHandler(RuntimeHandler):
    RUNTIME_CLASS = ContainerRuntime
    CMD_DELIMITER = ","
    CONST_CONTAINER_IMAGE = "CONTAINER_CUSTOM_IMAGE"
    CONST_CONTAINER_ENTRYPOINT = "CONTAINER_ENTRYPOINT"
    CONST_CONTAINER_CMD = "CONTAINER_CMD"

    def _translate_artifact(self, runtime: Runtime):
        """Specifies a dummy script as the job artifact.
        runtime is not used in this method.

        Parameters
        ----------
        runtime : Runtime
            This is not used.

        Returns
        -------
        str
            Path to the dummy script.
        """
        return os.path.join(
            os.path.dirname(__file__), "../../templates", "container.py"
        )

    def _translate_env(self, runtime: ContainerRuntime) -> dict:
        """Translate the environment variable.

        Parameters
        ----------
        runtime : GitPythonRuntime
            An instance of GitPythonRuntime

        Returns
        -------
        dict
            A dictionary containing environment variables for OCI data science job.
        """
        if not runtime.image:
            raise ValueError("Specify container image for ContainerRuntime.")
        envs = super()._translate_env(runtime)
        spec_mappings = {
            ContainerRuntime.CONST_IMAGE: self.CONST_CONTAINER_IMAGE,
            ContainerRuntime.CONST_ENTRYPOINT: self.CONST_CONTAINER_ENTRYPOINT,
            ContainerRuntime.CONST_CMD: self.CONST_CONTAINER_CMD,
        }
        envs.update(self._translate_specs(runtime, spec_mappings, self.CMD_DELIMITER))
        return envs

    @staticmethod
    def split_args(args: str) -> list:
        """Splits the cmd or entrypoint arguments for BYOC job into a list.
        BYOC jobs uses environment variables to store the values of cmd and entrypoint.
        In the values, comma(,) is used to separate cmd or entrypoint arguments.
        In YAML, the arguments are formatted into a list (Exec form).

        >>> ContainerRuntimeHandler.split_args("/bin/bash")
        ["/bin/bash"]
        >>> ContainerRuntimeHandler.split_args("-c,echo Hello World")
        ['-c', 'echo Hello World']

        Parameters
        ----------
        args : str
            Arguments in a comma separated string.

        Returns
        -------
        list
            Arguments in a list
        """
        if not args:
            return []
        return [
            arg.strip() for arg in args.split(ContainerRuntimeHandler.CMD_DELIMITER)
        ]

    def _extract_envs(self, dsc_job):
        """Extract the environment variables from data science job.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        dict
            A runtime specification dictionary for initializing a runtime.
        """
        spec = super()._extract_envs(dsc_job)
        envs = spec.pop(ContainerRuntime.CONST_ENV_VAR, {})
        if self.CONST_CONTAINER_IMAGE not in envs:
            raise IncompatibleRuntime()
        spec[ContainerRuntime.CONST_IMAGE] = envs.pop(self.CONST_CONTAINER_IMAGE)
        cmd = self.split_args(envs.pop(self.CONST_CONTAINER_CMD, ""))
        if cmd:
            spec[ContainerRuntime.CONST_CMD] = cmd
        entrypoint = self.split_args(envs.pop(self.CONST_CONTAINER_ENTRYPOINT, ""))
        if entrypoint:
            spec[ContainerRuntime.CONST_ENTRYPOINT] = entrypoint
        if envs:
            spec[ContainerRuntime.CONST_ENV_VAR] = envs
        return spec


class PyTorchDistributedRuntimeHandler(PythonRuntimeHandler):
    RUNTIME_CLASS = PyTorchDistributedRuntime
    CONST_WORKER_COUNT = "OCI__WORKER_COUNT"
    CONST_COMMAND = "OCI__LAUNCH_CMD"
    CONST_DEEPSPEED = "OCI__DEEPSPEED"

    GIT_SPEC_MAPPINGS = {
        cluster_config_helper.OCI__RUNTIME_URI: GitPythonRuntime.CONST_GIT_URL,
        cluster_config_helper.OCI__RUNTIME_GIT_BRANCH: GitPythonRuntime.CONST_BRANCH,
        cluster_config_helper.OCI__RUNTIME_GIT_COMMIT: GitPythonRuntime.CONST_COMMIT,
        cluster_config_helper.OCI__RUNTIME_GIT_SECRET_ID: GitPythonRuntime.CONST_GIT_SSH_SECRET_ID,
    }

    SPEC_MAPPINGS = PythonRuntimeHandler.SPEC_MAPPINGS
    SPEC_MAPPINGS.update(
        {
            PyTorchDistributedRuntime.CONST_COMMAND: CONST_COMMAND,
        }
    )

    def _translate_artifact(self, runtime: PyTorchDistributedRuntime):
        return PyTorchDistributedArtifact(runtime.source_uri, runtime)

    def _translate_env(self, runtime: PyTorchDistributedRuntime) -> dict:
        envs = super()._translate_env(runtime)
        replica = runtime.replica if runtime.replica else 1
        # WORKER_COUNT = REPLICA - 1 so that it will be same as distributed training
        envs[self.CONST_WORKER_COUNT] = str(replica - 1)
        envs[self.CONST_JOB_ENTRYPOINT] = PyTorchDistributedArtifact.CONST_DRIVER_SCRIPT
        if runtime.inputs:
            envs[driver_utils.CONST_ENV_INPUT_MAPPINGS] = json.dumps(runtime.inputs)
        if runtime.git:
            for env_key, spec_key in self.GIT_SPEC_MAPPINGS.items():
                if not runtime.git.get(spec_key):
                    continue
                envs[env_key] = runtime.git[spec_key]
        if runtime.dependencies:
            if PyTorchDistributedRuntime.CONST_PIP_PKG in runtime.dependencies:
                envs[driver_utils.CONST_ENV_PIP_PKG] = runtime.dependencies[
                    PyTorchDistributedRuntime.CONST_PIP_PKG
                ]
            if PyTorchDistributedRuntime.CONST_PIP_REQ in runtime.dependencies:
                envs[driver_utils.CONST_ENV_PIP_REQ] = runtime.dependencies[
                    PyTorchDistributedRuntime.CONST_PIP_REQ
                ]
        if runtime.use_deepspeed:
            envs[self.CONST_DEEPSPEED] = "1"
        return envs

    def _extract_envs(self, dsc_job) -> dict:
        spec = super()._extract_envs(dsc_job)
        envs = spec.pop(PythonRuntime.CONST_ENV_VAR, {})
        if self.CONST_WORKER_COUNT not in envs:
            raise IncompatibleRuntime()
        # Replicas
        spec[PyTorchDistributedRuntime.CONST_REPLICA] = (
            int(envs.pop(self.CONST_WORKER_COUNT)) + 1
        )
        # Git
        if cluster_config_helper.OCI__RUNTIME_URI in envs:
            git_spec = {}
            for env_key, spec_key in self.GIT_SPEC_MAPPINGS.items():
                if env_key in envs:
                    git_spec[spec_key] = envs.pop(env_key)
            spec[PyTorchDistributedRuntime.CONST_GIT] = git_spec
        # Inputs
        input_mappings = envs.pop(driver_utils.CONST_ENV_INPUT_MAPPINGS, None)
        if input_mappings:
            try:
                spec[PyTorchDistributedRuntime.CONST_INPUT] = json.loads(input_mappings)
            except ValueError:
                spec[PyTorchDistributedRuntime.CONST_INPUT] = input_mappings
        # Dependencies
        dep = {}
        if driver_utils.CONST_ENV_PIP_PKG in envs:
            dep[PyTorchDistributedRuntime.CONST_PIP_PKG] = envs.pop(
                driver_utils.CONST_ENV_PIP_PKG
            )
        if driver_utils.CONST_ENV_PIP_REQ in envs:
            dep[PyTorchDistributedRuntime.CONST_PIP_REQ] = envs.pop(
                driver_utils.CONST_ENV_PIP_REQ
            )
        if dep:
            spec[PyTorchDistributedRuntime.CONST_DEP] = dep
        if envs.pop(self.CONST_DEEPSPEED, None):
            spec[PyTorchDistributedRuntime.CONST_DEEPSPEED] = True
        # Envs
        if envs:
            spec[PythonRuntime.CONST_ENV_VAR] = envs
        return spec


class DataScienceJobRuntimeManager(RuntimeHandler):
    """This class is used by the DataScienceJob infrastructure to handle the runtime conversion.
    The translate() method determines the actual runtime handler by matching the RUNTIME_CLASS.
    The extract() method determines the actual runtime handler by checking if the runtime can be extracted.
    The order in runtime_handlers is used for extraction until a runtime is extracted.
    RuntimeHandler on the top of the list will have higher priority.
    If a runtime is a specify case of another runtime, the handler should be placed with higher priority.
    """

    runtime_handlers = [
        ContainerRuntimeHandler,
        PyTorchDistributedRuntimeHandler,
        GitPythonRuntimeHandler,
        NotebookRuntimeHandler,
        PythonRuntimeHandler,
        ScriptRuntimeHandler,
    ]

    def translate(self, runtime) -> dict:
        """Translates the runtime into a JSON payload for OCI API.
        This method determines the actual runtime handler by matching the RUNTIME_CLASS.

        Parameters
        ----------
        runtime : Runtime
            An instance of the runtime to be converted to a JSON payload.

        Returns
        -------
        dict
            JSON payload for defining a Data Science Job with OCI API
        """
        for runtime_handler in self.runtime_handlers:
            if runtime_handler.RUNTIME_CLASS == runtime.__class__:
                return runtime_handler(self.data_science_job).translate(runtime)
        raise NotImplementedError(
            f"{runtime.__class__.__name__} is not supported as the runtime of DataScienceJob."
        )

    def extract(self, dsc_job):
        """Extract the runtime from an OCI data science job object.

        This method determines the actual runtime handler by checking if the runtime can be extracted.

        Parameters
        ----------
        dsc_job : DSCJob or oci.datascience.models.Job
            The data science job containing runtime information.

        Returns
        -------
        Runtime:
            The runtime extracted from the data science job.
        """
        for runtime_handler in self.runtime_handlers:
            try:
                return runtime_handler(self.data_science_job).extract(dsc_job)
            except IncompatibleRuntime:
                pass
        raise NotImplementedError("Unable to extract runtime.")
