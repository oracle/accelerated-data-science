#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import datetime
import logging
import os
import traceback
from typing import Any, Dict, List, Optional

import fsspec
import oci
import oci.util as oci_util

from ads.common import utils
from ads.common.oci_datascience import DSCNotebookSession, OCIDataScienceMixin
from ads.common.oci_logging import OCILog
from ads.common.oci_resource import ResourceNotFoundError
from ads.config import COMPARTMENT_OCID, NB_SESSION_OCID, PROJECT_OCID
from ads.jobs.builders.base import Builder
from ads.jobs.builders.infrastructure.dsc_job import DataScienceJob, DSCJob
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    DataScienceJobRuntimeManager,
)
from ads.jobs.builders.runtimes.artifact import Artifact
from ads.jobs.builders.runtimes.python_runtime import (
    GitPythonRuntime,
    NotebookRuntime,
    PythonRuntime,
    ScriptRuntime,
)
from ads.pipeline.ads_pipeline_run import PipelineRun
from ads.pipeline.ads_pipeline_step import PipelineStep
from ads.pipeline.visualizer.base import GraphOrientation, PipelineVisualizer
from ads.pipeline.visualizer.graph_renderer import PipelineGraphRenderer

logger = logging.getLogger(__name__)

MAXIMUM_TIMEOUT = 1800
DEFAULT_WAITER_KWARGS = {"max_wait_seconds": MAXIMUM_TIMEOUT}
DEFAULT_OPERATION_KWARGS = {
    "delete_related_pipeline_runs": True,
    "delete_related_job_runs": True,
}
ALLOWED_OPERATION_KWARGS = [
    "allow_control_chars",
    "retry_strategy",
    "delete_related_job_runs",
    "delete_related_pipeline_runs",
    "if_match",
    "opc_request_id",
]
ALLOWED_WAITER_KWARGS = [
    "max_interval_seconds",
    "max_wait_seconds",
    "succeed_on_not_found",
    "wait_callback",
    "fetch_func",
]


class Pipeline(Builder):
    """Represents a Data Science Machine Learning Pipeline."""

    CONST_ID = "id"
    CONST_PIPELINE_ID = "pipelineId"
    CONST_DISPLAY_NAME = "displayName"
    CONST_STEP_DETAILS = "stepDetails"
    CONST_STEP_OVERRIDE_DETAILS = "stepOverrideDetails"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_PROJECT_ID = "projectId"
    CONST_LOG_GROUP_ID = "logGroupId"
    CONST_LOG_ID = "logId"
    CONST_SERVICE_LOG_ID = "serviceLogId"
    CONST_ENABLE_SERVICE_LOG = "enableServiceLog"
    CONST_ENVIRONMENT_VARIABLES = "environmentVariables"
    CONST_COMMAND_LINE_ARGUMENTS = "commandLineArguments"
    CONST_CREATED_BY = "createdBy"
    CONST_DESCRIPTION = "description"
    CONST_TYPE = "type"
    CONST_MAXIMUM_RUNTIME_IN_MINUTES = "maximumRuntimeInMinutes"
    CONST_ENABLE_LOGGING = "enableLogging"
    CONST_ENABLE_AUTO_LOG_CREATION = "enableAutoLogCreation"
    CONST_CONFIGURATION_DETAILS = "configurationDetails"
    CONST_CONFIGURATION_OVERRIDE_DETAILS = "configurationOverrideDetails"
    CONST_LOG_CONFIGURATION_DETAILS = "logConfigurationDetails"
    CONST_LOG_CONFIGURATION_OVERRIDE_DETAILS = "logConfigurationOverrideDetails"
    CONST_FREEFROM_TAGS = "freeformTags"
    CONST_DEFINED_TAGS = "definedTags"
    CONST_SYSTEM_TAGS = "systemTags"
    CONST_INFRA_CONFIG_DETAILS = "infrastructureConfigurationDetails"
    CONST_SHAPE_NAME = "shapeName"
    CONST_BLOCK_STORAGE_SIZE = "blockStorageSizeInGBs"
    CONST_SHAPE_CONFIG_DETAILS = "shapeConfigDetails"
    CONST_OCPUS = "ocpus"
    CONST_MEMORY_IN_GBS = "memoryInGBs"
    CONST_SERVICE_LOG_CATEGORY = "pipelinerunlog"
    CONST_SERVICE = "datascience"
    CONST_DAG = "dag"

    LIFECYCLE_STATE_CREATING = "CREATING"
    LIFECYCLE_STATE_ACTIVE = "ACTIVE"
    LIFECYCLE_STATE_DELETING = "DELETING"
    LIFECYCLE_STATE_FAILED = "FAILED"
    LIFECYCLE_STATE_DELETED = "DELETED"

    def __init__(self, name: str = None, spec: Dict = None, **kwargs) -> None:
        """Initialize a pipeline.

        Parameters
        ----------
        name: str
            The name of the pipeline, default to None. If a name is not provided, a randomly generated easy to remember
            name with timestamp will be generated, like 'strange-spider-2022-08-17-23:55.02'.
        spec : dict, optional
            Object specification, default to None
        kwargs: dict
            Specification as keyword arguments.
            If spec contains the same key as the one in kwargs, the value from kwargs will be used.

            - project_id: str
            - compartment_id: str
            - display_name: str
            - description: str
            - maximum_runtime_in_minutes: int
            - environment_variables: dict(str, str)
            - command_line_arguments: str
            - log_id: str
            - log_group_id: str
            - enable_service_log: bool
            - shape_name: str
            - block_storage_size_in_gbs: int
            - shape_config_details: dict
            - step_details: list[PipelineStep]
            - dag: list[str]
            - defined_tags: dict(str, dict(str, object))
            - freeform_tags: dict[str, str]

        Attributes
        ----------
        kind: str
            The kind of the object as showing in YAML.
        name: str
            The name of pipeline.
        id: str
            The id of pipeline.
        step_details: List[PipelineStep]
            The step details of pipeline.
        dag_details: List[str]
            The dag details of pipeline.
        log_group_id: str
            The log group id of pipeline.
        log_id: str
            The log id of pipeline.
        project_id: str
            The project id of pipeline.
        compartment_id: str
            The compartment id of pipeline.
        created_by: str
            The created by of pipeline.
        description: str
            The description of pipeline.
        environment_variable: dict
            The environment variables of pipeline.
        argument: str
            The command line argument of pipeline.
        maximum_runtime_in_minutes: int
            The maximum runtime in minutes of pipeline.
        shape_name: str
            The shape name of pipeline infrastructure.
        block_storage_size_in_gbs: int
            The block storage of pipeline infrastructure.
        shape_config_details: dict
            The shape config details of pipeline infrastructure.
        enable_service_log: bool
            The value to enable service log or not.
        service_log_id: str
            The service log id of pipeline.
        status: str
            The status of the pipeline.

        Methods
        -------
        with_name(self, name: str) -> Pipeline
            Sets the name of pipeline.
        with_id(self, id: str) -> Pipeline
            Sets the ocid of pipeline.
        with_step_details(self, step_details: List[PipelineStep]) -> Pipeline
            Sets the step details of pipeline.
        with_dag_details(self, dag_details: List[str]) -> Pipeline
            Sets the dag details of pipeline.
        with_log_group_id(self, log_group_id: str) -> Pipeline
            Sets the log group id of pipeline.
        with_log_id(self, log_id: str) -> Pipeline
            Sets the log id of pipeline.
        with_project_id(self, project_id: str) -> Pipeline
            Sets the project id of pipeline.
        with_compartment_id(self, compartment_id: str) -> Pipeline
            Sets the compartment id of pipeline.
        with_created_by(self, created_by: str) -> Pipeline
            Sets the created by of pipeline.
        with_description(self, description: str) -> Pipeline
            Sets the description of pipeline.
        with_environment_variable(self, **kwargs) -> Pipeline
            Sets the environment variables of pipeline.
        with_argument(self, *args, **kwargs) -> Pipeline
            Sets the command line arguments of pipeline.
        with_maximum_runtime_in_minutes(self, maximum_runtime_in_minutes: int) -> Pipeline
            Sets the maximum runtime in minutes of pipeline.
        with_freeform_tags(self, freeform_tags: Dict) -> Pipeline
            Sets the freeform tags of pipeline.
        with_defined_tags(self, defined_tags: Dict) -> Pipeline
            Sets the defined tags of pipeline.
        with_shape_name(self, shape_name: str) -> Pipeline
            Sets the shape name of pipeline infrastructure.
        with_block_storage_size_in_gbs(self, block_storage_size_in_gbs: int) -> Pipeline
            Sets the block storage size of pipeline infrastructure.
        with_shape_config_details(self, shape_config_details: Dict) -> Pipeline
            Sets the shape config details of pipeline infrastructure.
        with_enable_service_log(self, enable_service_log: bool) -> Pipeline
            Sets the value to enable the service log of pipeline.
        to_dict(self) -> dict:
            Serializes the pipeline specifications to a dictionary.
        from_dict(cls, obj_dict: dict):
            Initializes the object from a dictionary.
        create(self, delete_if_fail: bool = True) -> Pipeline
            Creates an ADS pipeline.
        show(self, rankdir: str = GraphOrientation.TOP_BOTTOM)
            Render pipeline with step information in a graph.
        to_svg(self, uri: str = None, rankdir: str = GraphOrientation.TOP_BOTTOM, **kwargs) -> str:
            Renders pipeline as graph into SVG.
        run(self, display_name: Optional[str] = None, project_id: Optional[str] = None, compartment_id: Optional[str] = None, configuration_override_details: Optional[dict] = None, log_configuration_override_details: Optional[dict] = None, step_override_details: Optional[list] = None, free_form_tags: Optional[dict] = None, defined_tags: Optional[dict] = None, system_tags: Optional[dict] = None) -> PipelineRun
            Creates and/or overrides an ADS pipeline run.
        delete(self, delete_related_pipeline_runs: Optional[bool] = True, delete_related_job_runs: Optional[bool] = True, max_wait_seconds: Optional[int] = MAXIMUM_TIMEOUT, **kwargs) -> Pipeline
            Deletes an ADS pipeline run.
        from_ocid(cls, ocid: str) -> Pipeline
            Creates an ADS pipeline from ocid.
        from_id(cls, id: str) -> Pipeline
            Creates an ADS pipeline from ocid.
        to_yaml(self, uri=None, **kwargs)
            Returns Pipeline serialized as a YAML string
        from_yaml(cls, yaml_string=None, uri=None, **kwargs)
            Creates an Pipeline from YAML string provided or from URI location containing YAML string
        list(cls, compartment_id: Optional[str] = None, **kwargs) -> List[Pipeline]
            List pipelines in a given compartment.
        run_list(self, **kwargs) -> List[PipelineRun]
            Gets a list of runs of the pipeline.

        Example
        -------
        Here is an example for creating and running a pipeline using builder:

        .. code-block:: python

            from ads.pipeline import Pipeline, CustomScriptStep, ScriptRuntime
            # Define an OCI Data Science pipeline
            pipeline = (
                Pipeline(name="<pipeline_name>")
                .with_compartment_id("<compartment_id>")
                .with_project_id("<project_id>")
                .with_log_group_id("<log_group_id>")
                .with_log_id("<log_id>")
                .with_description("<description>")
                .with_maximum_runtime_in_minutes(200)
                .with_argument("argument", key="value")
                .with_environment_variable(env="value")
                .with_freeform_tags({"key": "value"})
                .with_step_details([
                    (
                        PipelineStep(name="PipelineStepOne")
                        .with_job_id("<job_id>")
                        .with_description("<description>")
                    ),
                    (
                        PipelineStep(name="PipelineStepTwo")
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
                ])
                .with_dag_details(["PipelineStepOne >> PipelineStepTwo"])
            )
            # Create and Run the pipeline
            run = pipeline.create().run()
            # Stream the pipeline run outputs
            run.watch()

        See Also
        --------
        https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/pipeline/index.html
        """
        self.attribute_set = {
            "id",
            "createdBy",
            "projectId",
            "compartmentId",
            "description",
            "pipelineId",
            "displayName",
            "configurationDetails",
            "logConfigurationDetails",
            "infrastructureConfigurationDetails",
            "stepDetails",
            "lifecycleState",
            "freeformTags",
            "definedTags",
            "systemTags",
            "dag",
            "logId",
            "logGroupId",
        }

        self.attribute_map = {
            "projectId": "project_id",
            "compartmentId": "compartment_id",
            "displayName": "display_name",
            "description": "description",
            "maximumRuntimeInMinutes": "maximum_runtime_in_minutes",
            "environmentVariables": "environment_variables",
            "commandLineArguments": "command_line_arguments",
            "logId": "log_id",
            "logGroupId": "log_group_id",
            "enableServiceLog": "enable_service_log",
            "stepDetails": "step_details",
            "freeformTags": "freeform_tags",
            "definedTags": "defined_tags",
            "shapeName": "shape_name",
            "blockStorageSizeInGBs": "block_storage_size_in_gbs",
            "shapeConfigDetails": "shape_config_details",
            "dag": "dag",
            "id": "id",
        }

        self._artifact_content_map = {}
        super().__init__(spec=spec, **kwargs)
        name = (
            name
            or self.get_spec(self.CONST_DISPLAY_NAME)
            or utils.get_random_name_for_resource()
        )
        self.set_spec(self.CONST_DISPLAY_NAME, name)
        if "dag" in kwargs:
            self.with_dag(kwargs.get("dag"))
        elif spec and "dag" in spec:
            self.with_dag(spec.get("dag"))
        self.data_science_pipeline = None
        self.service_logging = None

    def _load_default_properties(self) -> Dict:
        """Load default properties from environment variables, notebook session, etc.

        Returns
        -------
        Dict
            A dictionary of default properties.
        """
        defaults = super()._load_default_properties()
        if COMPARTMENT_OCID:
            defaults[self.CONST_COMPARTMENT_ID] = COMPARTMENT_OCID
        if PROJECT_OCID:
            defaults[self.CONST_PROJECT_ID] = PROJECT_OCID

        if NB_SESSION_OCID:
            try:
                nb_session = DSCNotebookSession.from_ocid(NB_SESSION_OCID)
                nb_config = nb_session.notebook_session_configuration_details
                defaults[self.CONST_SHAPE_NAME] = nb_config.shape
                defaults[
                    self.CONST_BLOCK_STORAGE_SIZE
                ] = nb_config.block_storage_size_in_gbs

                if nb_config.notebook_session_shape_config_details:
                    notebook_shape_config_details = oci_util.to_dict(
                        nb_config.notebook_session_shape_config_details
                    )
                    defaults[self.CONST_SHAPE_CONFIG_DETAILS] = copy.deepcopy(
                        notebook_shape_config_details
                    )

            except Exception as e:
                logger.warning(
                    f"Error fetching details about Notebook "
                    f"session: {NB_SESSION_OCID}. {e}"
                )
                logger.debug(traceback.format_exc())

        return defaults

    @property
    def kind(self) -> str:
        """The kind of the object as showing in YAML.

        Returns
        -------
        str
            pipeline
        """
        return "pipeline"

    @property
    def name(self) -> str:
        """The name of the pipeline.

        Returns
        -------
        str
            The name of the pipeline.
        """
        return self.get_spec(self.CONST_DISPLAY_NAME)

    def with_name(self, name: str) -> "Pipeline":
        """Sets the name of pipeline.

        Parameters
        ----------
        name: str
            The name of pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_DISPLAY_NAME, name)

    @property
    def id(self) -> str:
        """The id of the pipeline.

        Returns
        -------
        str
            The id of the pipeline.
        """
        return self.get_spec(self.CONST_ID)

    def with_id(self, id: str) -> "Pipeline":
        """Sets the id of pipeline.

        Parameters
        ----------
        id: str
            The id of pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_ID, id)

    @property
    def step_details(self) -> List["PipelineStep"]:
        """The step details of the pipeline.

        Returns
        -------
        list
            The step details of the pipeline.
        """
        return self.get_spec(self.CONST_STEP_DETAILS)

    def with_step_details(self, step_details: List["PipelineStep"]) -> "Pipeline":
        """Sets the pipeline step details for the pipeline.

        Parameters
        ----------
        step_details: list
            A list of steps in the pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        self.set_spec(self.CONST_DAG, None)
        return self.set_spec(self.CONST_STEP_DETAILS, step_details)

    @property
    def dag(self) -> List[str]:
        """The dag details of the pipeline.

        Returns
        -------
        list
            The dag details of the pipeline.
        """
        return self.get_spec(self.CONST_DAG)

    def with_dag(self, dag: List[str]) -> "Pipeline":
        """Sets the pipeline dag details for the pipeline.

        Parameters
        ----------
        dag: list
            A list of dag representing step dependencies in the pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        self.set_spec(self.CONST_DAG, dag)

        if not self.step_details:
            raise ValueError("Pipeline step details must be specified first.")

        stepname_to_step_map = {x.name: x for x in self.step_details}
        updated_step_details = Pipeline._add_dag_to_node(dag, stepname_to_step_map)

        return self.set_spec(self.CONST_STEP_DETAILS, updated_step_details)

    @staticmethod
    def _add_dag_to_node(
        dag: List[str], stepname_to_step_map: dict
    ) -> List["PipelineStep"]:
        """Add dependencies to pipeline steps.

        Parameters
        ----------
        dag: list
            A list of dag representing step dependencies in the pipeline.
        stepname_to_step_map: dict
            A dict mapping PipelineStep name to the PipelineStep object.

        Returns
        -------
        List
            A list of PipelineStep.
        """

        dag_mapping = {x: [] for x in stepname_to_step_map}
        updated_step_details = []
        if dag:
            for dag_line in dag:
                dependency = [
                    x.strip().strip("()").split(",") for x in dag_line.split(">>")
                ]
                for i in range(len(dependency) - 1):
                    for node1 in dependency[i]:
                        for node2 in dependency[i + 1]:
                            node1 = node1.strip()
                            node2 = node2.strip()
                            if node1 not in stepname_to_step_map:
                                raise ValueError(
                                    f"Pipeline step with name {node1} does not exist. Please provide a valid step name."
                                )
                            if node2 not in stepname_to_step_map:
                                raise ValueError(
                                    f"Pipeline step with name {node2} does not exist. Please provide a valid step name."
                                )

                            dag_mapping[node2].append(node1)

        for node, prev_list in dag_mapping.items():
            node_list = [stepname_to_step_map[x] for x in prev_list]
            stepname_to_step_map[node]._with_depends_on(node_list)
            updated_step_details.append(stepname_to_step_map[node])

        return updated_step_details

    @property
    def log_group_id(self) -> str:
        """The log group id of the pipeline.

        Returns
        -------
        str:
            The log group id of the pipeline.
        """
        return self.get_spec(self.CONST_LOG_GROUP_ID)

    def with_log_group_id(self, log_group_id: str) -> "Pipeline":
        """Sets the log group id of the pipeline.

        Parameters
        ----------
        log_group_id: str
            The log group id of the pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_LOG_GROUP_ID, log_group_id)

    @property
    def log_id(self) -> str:
        """The log id of the pipeline.

        Returns
        -------
        str:
            The log id of the pipeline.
        """
        return self.get_spec(self.CONST_LOG_ID)

    def with_log_id(self, log_id: str) -> "Pipeline":
        """Sets the log id of the pipeline.

        Parameters
        ----------
        log_id: str
            The log id of the pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_LOG_ID, log_id)

    @property
    def project_id(self) -> str:
        """The project id of the pipeline.

        Returns
        -------
        str:
            The project id of the pipeline.
        """
        return self.get_spec(self.CONST_PROJECT_ID)

    def with_project_id(self, project_id: str) -> "Pipeline":
        """Sets the project id of the pipeline.

        Parameters
        ----------
        project_id: str
            The project id of the pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_PROJECT_ID, project_id)

    @property
    def compartment_id(self) -> str:
        """The compartment id of the pipeline.

        Returns
        -------
        str:
            The compartment id of the pipeline.
        """
        return self.get_spec(self.CONST_COMPARTMENT_ID)

    def with_compartment_id(self, compartment_id: str) -> "Pipeline":
        """Sets the compartment id of the pipeline.

        Parameters
        ----------
        compartment_id: str
            The compartment id of the pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    @property
    def created_by(self) -> str:
        """The id that creates the pipeline.

        Returns
        -------
        str:
            The id that creates the pipeline.
        """
        return self.get_spec(self.CONST_CREATED_BY)

    def with_created_by(self, created_by: str) -> "Pipeline":
        """Sets the id that creates the pipeline.

        Parameters
        ----------
        created_by: str
            The id that creates the pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_CREATED_BY, created_by)

    @property
    def description(self) -> str:
        """The description of pipeline.

        Returns
        -------
        str:
            The description of pipeline.
        """
        return self.get_spec(self.CONST_DESCRIPTION)

    def with_description(self, description: str) -> "Pipeline":
        """Sets the description of the pipeline.

        Parameters
        ----------
        description: str
            The description of the pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_DESCRIPTION, description)

    @property
    def environment_variable(self) -> dict:
        """The environment variables of the pipeline.

        Returns
        -------
        dict:
            The environment variables of the pipeline.
        """
        return self.get_spec(self.CONST_ENVIRONMENT_VARIABLES)

    def with_environment_variable(self, **kwargs) -> "Pipeline":
        """Sets environment variables of the pipeline.

        Parameters
        ----------
        kwargs:
            Keyword arguments.
            To add a keyword argument without value, set the value to None.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        if kwargs:
            environment_variable_dict = {}
            for k, v in kwargs.items():
                environment_variable_dict[k] = v
            self.set_spec(self.CONST_ENVIRONMENT_VARIABLES, environment_variable_dict)
        return self

    @property
    def argument(self) -> str:
        """The command line arguments of the pipeline.

        Returns
        -------
        str:
            The command line arguments of the pipeline.
        """
        return self.get_spec(self.CONST_COMMAND_LINE_ARGUMENTS)

    def with_argument(self, *args, **kwargs) -> "Pipeline":
        """Adds command line arguments to the pipeline.
        Existing arguments will be preserved.
        This method can be called (chained) multiple times to add various arguments.
        For example, pipeline.with_argument(key="val").with_argument("path/to/file")
        will result in: "--key val path/to/file"

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
            The Pipeline instance (self).

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
    def maximum_runtime_in_minutes(self) -> int:
        """The maximum runtime in minutes of the pipeline.

        Returns
        -------
        int:
            The maximum runtime minutes of the pipeline.
        """
        return self.get_spec(self.CONST_MAXIMUM_RUNTIME_IN_MINUTES)

    def with_maximum_runtime_in_minutes(
        self, maximum_runtime_in_minutes: int
    ) -> "Pipeline":
        """Sets the maximum runtime in minutes of the pipeline.

        Parameters
        ----------
        maximum_runtime_in_minutes: int
            The maximum_runtime_in_minutes of the pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(
            self.CONST_MAXIMUM_RUNTIME_IN_MINUTES, maximum_runtime_in_minutes
        )

    def with_freeform_tags(self, freeform_tags: Dict) -> "Pipeline":
        """Sets freeform tags of the pipeline.

        Parameters
        ----------
        freeform_tags: dict
            The freeform tags dictionary.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_FREEFROM_TAGS, freeform_tags)

    def with_defined_tags(self, defined_tags: Dict) -> "Pipeline":
        """Sets defined tags of the pipeline.

        Parameters
        ----------
        defined_tags: dict
            The defined tags dictionary.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_DEFINED_TAGS, defined_tags)

    @property
    def shape_name(self) -> str:
        """The shape name of pipeline infrastructure.

        Returns
        -------
        str:
            The shape name of the pipeline infrastructure.
        """
        return self.get_spec(self.CONST_SHAPE_NAME)

    def with_shape_name(self, shape_name: str) -> "Pipeline":
        """Sets the shape name of pipeline infrastructure.

        Parameters
        ----------
        shape_name: str
            The shape name of the pipeline infrastructure.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_SHAPE_NAME, shape_name)

    @property
    def block_storage_size_in_gbs(self) -> int:
        """The block storage size of pipeline infrastructure.

        Returns
        -------
        int:
            The block storage size of the pipeline infrastructure.
        """
        return self.get_spec(self.CONST_BLOCK_STORAGE_SIZE)

    def with_block_storage_size_in_gbs(
        self, block_storage_size_in_gbs: int
    ) -> "Pipeline":
        """Sets the block storage size of pipeline infrastructure.

        Parameters
        ----------
        block_storage_size_in_gbs: int
            The block storage size of pipeline infrastructure.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_BLOCK_STORAGE_SIZE, block_storage_size_in_gbs)

    @property
    def shape_config_details(self) -> dict:
        """The shape config details of pipeline infrastructure.

        Returns
        -------
        dict:
            The shape config details of the pipeline infrastructure.
        """
        return self.get_spec(self.CONST_SHAPE_CONFIG_DETAILS)

    def with_shape_config_details(
        self, memory_in_gbs: float, ocpus: float, **kwargs: Dict[str, Any]
    ) -> "Pipeline":
        """
        Sets the shape config details of pipeline infrastructure.
        Specify only when a flex shape is selected.
        For example `VM.Standard.E3.Flex` allows the memory_in_gbs and cpu count to be specified.

        Parameters
        ----------
        memory_in_gbs: float
            The size of the memory in GBs.
        ocpus: float
            The OCPUs count.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(
            self.CONST_SHAPE_CONFIG_DETAILS,
            {
                self.CONST_OCPUS: ocpus,
                self.CONST_MEMORY_IN_GBS: memory_in_gbs,
                **kwargs,
            },
        )

    @property
    def enable_service_log(self) -> bool:
        """Enables service log of pipeline.

        Returns
        -------
        bool:
            The bool value to enable service log of pipeline.
        """
        return self.get_spec(self.CONST_ENABLE_SERVICE_LOG)

    def with_enable_service_log(self, enable_service_log: bool) -> "Pipeline":
        """Sets the bool value to enable the service log of pipeline.

        Parameters
        ----------
        enable_service_log: bool
            The value to enable the service log of pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self).
        """
        return self.set_spec(self.CONST_ENABLE_SERVICE_LOG, enable_service_log)

    @property
    def service_log_id(self) -> str:
        """The service log id of pipeline.

        Returns
        -------
        str:
            The service log id of pipeline.
        """
        return self.get_spec(self.CONST_SERVICE_LOG_ID)

    def to_dict(self, **kwargs) -> dict:
        """Serializes the pipeline specifications to a dictionary.

        Returns
        -------
        dict
            A dictionary containing pipeline specifications.
        """
        dict_details = copy.deepcopy(super().to_dict(**kwargs))
        dict_details["spec"][self.CONST_DAG] = self.get_spec(self.CONST_DAG)

        step_details_list = []
        if self.step_details:
            for step in self.step_details:
                if isinstance(step, PipelineStep):
                    step = step.to_dict()

                if not isinstance(step, dict):
                    raise TypeError("Pipeline step is not a valid type")
                step_dict = copy.deepcopy(step)
                step_dict["spec"].pop("dependsOn", None)
                step_details_list.append(step_dict)

        dict_details["spec"][self.CONST_STEP_DETAILS] = step_details_list
        return dict_details

    @classmethod
    def from_dict(cls, obj_dict: dict):
        """Initializes the object from a dictionary."""
        temp_obj_dict = copy.deepcopy(obj_dict)
        step_mapping = {}
        for step in temp_obj_dict["spec"][cls.CONST_STEP_DETAILS]:
            pipeline_step = PipelineStep.from_dict(step)
            step_mapping[pipeline_step.name] = pipeline_step

        if cls.CONST_DAG not in temp_obj_dict["spec"]:
            temp_obj_dict["spec"][cls.CONST_DAG] = None

        step_details = Pipeline._add_dag_to_node(
            temp_obj_dict["spec"][cls.CONST_DAG], step_mapping
        )
        temp_obj_dict["spec"][cls.CONST_STEP_DETAILS] = step_details

        return cls(spec=temp_obj_dict["spec"])

    def create(self, delete_if_fail: bool = True) -> "Pipeline":
        """Creates an ADS pipeline.

        Returns
        -------
        Pipeline:
            The ADS Pipeline instance.
        """
        pipeline_details = self.__pipeline_details()
        self.data_science_pipeline = DataSciencePipeline(**pipeline_details).create(
            self.step_details, delete_if_fail
        )
        self.set_spec(self.CONST_ID, self.data_science_pipeline.id)
        if self.enable_service_log and not self.service_log_id:
            try:
                self.__create_service_log()
            except Exception as ex:
                logger.warning("Failed to create service log: %s", str(ex))
        return self

    def _show(self) -> PipelineVisualizer:
        """
        Prepeares `PipelineVisualizer` instance to render a graph.

        Returns
        -------
        PipelineVisualizer
        """
        return (
            PipelineVisualizer()
            .with_renderer(PipelineGraphRenderer(show_status=False))
            .with_pipeline(self)
        )

    def show(self, rankdir: str = GraphOrientation.TOP_BOTTOM) -> None:
        """
        Render pipeline with step information in a graph

        Returns
        -------
        None
        """
        self._show().render(rankdir=rankdir)

    def to_svg(
        self, uri: str = None, rankdir: str = GraphOrientation.TOP_BOTTOM, **kwargs
    ) -> str:
        """
        Renders pipeline as graph in svg string.

        Parameters
        ----------
        uri: (string, optional). Defaults to None.
            URI location to save the SVG string.
        rankdir: str, default to "TB".
            Direction of the rendered graph; allowed Values are {"TB", "LR"}.

        Returns
        -------
        str
            Graph in svg format.
        """
        return self._show().to_svg(uri=uri, rankdir=rankdir, **kwargs)

    def run(
        self,
        display_name: Optional[str] = None,
        project_id: Optional[str] = None,
        compartment_id: Optional[str] = None,
        configuration_override_details: Optional[dict] = None,
        log_configuration_override_details: Optional[dict] = None,
        step_override_details: Optional[list] = None,
        free_form_tags: Optional[dict] = None,
        defined_tags: Optional[dict] = None,
        system_tags: Optional[dict] = None,
    ) -> "PipelineRun":
        """Creates an ADS pipeline run.

        Parameters
        ----------
        display_name: str, optional
            The display name to override the one defined previously. Defaults to None.

        project_id: str, optional
            The project id to override the one defined previously. Defaults to None.

        compartment_id: str, optional
            The compartment id to override the one defined previously. Defaults to None.

        configuration_override_details: dict, optional
            The configuration details dictionary to override the one defined previously.
            Defaults to None.
            The configuration_override_details contains the following keys:
            * "type": str, only "DEFAULT" is allowed.
            * "environment_variables": dict, optional, the environment variables
            * "command_line_arguments": str, optional, the command line arguments
            * "maximum_runtime_in_minutes": int, optional, the maximum runtime allowed in minutes

        log_configuration_override_details: dict(str, str), optional
            The log configuration details dictionary to override the one defined previously.
            Defaults to None.
            The log_configuration_override_details contains the following keys:
            * "log_group_id": str, optional, the log group id
            * "log_id": str, optional, the log id

        step_override_details: list[PipelineStepOverrideDetails], optional
            The step details list to override the one defined previously.
            Defaults to None.
            The PipelineStepOverrideDetails is a dict which contains the following keys:
            * step_name: str, the name of step to override
            * step_configuration_details: dict, which contains:
                * "maximum_runtime_in_minutes": int, optional
                * "environment_variables": dict, optional
                * "command_line_arguments": str, optional

        free_form_tags: dict(str, str), optional
            The free from tags dictionary to override the one defined previously.
            Defaults to None.

        defined_tags: dict(str, dict(str, object)), optional
            The defined tags dictionary to override the one defined previously.
            Defaults to None.

        system_tags: dict(str, dict(str, object)), optional
            The system tags dictionary to override the one defined previously.
            Defaults to None.

        Example
        --------
        .. code-block:: python

            # Creates a pipeline run using pipeline configurations
            pipeline.run()

            # Creates a pipeline run by overriding pipeline configurations
            pipeline.run(
                display_name="OverrideDisplayName",
                configuration_override_details={
                    "maximum_runtime_in_minutes":30,
                    "type":"DEFAULT",
                    "environment_variables": {
                        "key": "value"
                    },
                    "command_line_arguments": "ARGUMENT --KEY VALUE",
                },
                log_configuration_override_details={
                    "log_group_id": "<log_group_id>"
                },
                step_override_details=[{
                    "step_name" : "<step_name>",
                    "step_configuration_details" : {
                        "maximum_runtime_in_minutes": 200,
                        "environment_variables": {
                            "1":"2"
                        },
                        "command_line_arguments": "argument --key value",
                    }
                }]
            )

        Returns
        -------
        PipelineRun:
            The ADS PipelineRun instance.
        """
        pipeline_details = self.__pipeline_details()
        self.__override_configurations(
            pipeline_details,
            display_name,
            project_id,
            compartment_id,
            configuration_override_details,
            log_configuration_override_details,
            step_override_details,
            free_form_tags,
            defined_tags,
            system_tags,
        )
        if not self.data_science_pipeline:
            self.data_science_pipeline = DataSciencePipeline(**pipeline_details)

        if self.enable_service_log:
            return self.data_science_pipeline.run(
                pipeline_details, self.service_logging
            )

        return self.data_science_pipeline.run(pipeline_details)

    def delete(
        self,
        delete_related_pipeline_runs: Optional[bool] = True,
        delete_related_job_runs: Optional[bool] = True,
        max_wait_seconds: Optional[int] = MAXIMUM_TIMEOUT,
        **kwargs,
    ) -> "Pipeline":
        """Deteles an ADS pipeline.

        Parameters
        ----------
        delete_related_pipeline_runs: bool, optional
            Specify whether to delete related PipelineRuns or not. Defaults to True.
        delete_related_job_runs: bool, optional
            Specify whether to delete related JobRuns or not. Defaults to True.
        max_wait_seconds: int, optional
            The maximum time to wait, in seconds. Defaults to 1800.

        kwargs: optional
        The kwargs to be executed when deleting the pipeline.
        The allowed keys are:
        * "allow_control_chars": bool, to indicate whether or not this request should
        allow control characters in the response object. By default, the response will
        not allow control characters in strings.
        * "retry_strategy": obj, to apply to this specific operation/call. This will
        override any retry strategy set at the client-level. This should be one of the
        strategies available in the :py:mod:`~oci.retry` module. This operation will not
        retry by default, users can also use the convenient :py:data:`~oci.retry.DEFAULT_RETRY_STRATEGY`
        provided by the SDK to enable retries for it. The specifics of the default retry
        strategy are described `here <https://docs.oracle.com/en-us/iaas/tools/python/latest/sdk_behaviors/retries.html>`__.
        To have this operation explicitly not perform any retries, pass an instance of :py:class:`~oci.retry.NoneRetryStrategy`.
        * "if_match": str, for optimistic concurrency control. In the PUT or DELETE call
        for a resource, set the `if-match` parameter to the value of the etag from a
        previous GET or POST response for that resource. The resource is updated or
        deleted only if the `etag` you provide matches the resource's current `etag` value.
        * "opc_request_id": str, unique Oracle assigned identifier for the request.
        If you need to contact Oracle about a particular request, then provide the request ID.
        * "max_interval_seconds": int, the maximum interval between queries, in seconds.
        * "succeed_on_not_found": bool, to determine whether or not the waiter should
        return successfully if the data we're waiting on is not found
        (e.g. a 404 is returned from the service). This defaults to False and so a 404 would
        cause an exception to be thrown by this function. Setting it to True may be useful in
        scenarios when waiting for a resource to be terminated/deleted since it is possible that
        the resource would not be returned by the a GET call anymore.
        * "wait_callback": A function which will be called each time that we have to do an initial
        wait (i.e. because the property of the resource was not in the correct state,
        or the ``evaluate_response`` function returned False). This function should take two
        arguments - the first argument is the number of times we have checked the resource,
        and the second argument is the result of the most recent check.
        * "fetch_func": A function to be called to fetch the updated state from the server.
        This can be used if the call to check for state needs to be more complex than a single
        GET request. For example, if the goal is to wait until an item appears in a list,
        fetch_func can be a function that paginates through a full list on the server.

        Returns
        -------
        Pipeline:
            The ADS Pipeline instance.
        """
        if not self.data_science_pipeline:
            self.data_science_pipeline = DataSciencePipeline()

        operation_kwargs = {
            "delete_related_pipeline_runs": delete_related_pipeline_runs,
            "delete_related_job_runs": delete_related_job_runs,
        }
        waiter_kwargs = {"max_wait_seconds": max_wait_seconds}
        for key, value in kwargs.items():
            if key in ALLOWED_OPERATION_KWARGS:
                operation_kwargs[key] = value
            elif key in ALLOWED_WAITER_KWARGS:
                waiter_kwargs[key] = value

        self.data_science_pipeline.delete(
            id=self.id, operation_kwargs=operation_kwargs, waiter_kwargs=waiter_kwargs
        )
        return self

    def download(
        self, to_dir: str, override_if_exists: Optional[bool] = False
    ) -> "Pipeline":
        """Downloads artifacts from pipeline.

        Parameters
        ----------
        to_dir : str
            Local directory to which the artifacts will be downloaded to.
        override_if_exists: bool, optional
            Bool to decide whether to override existing folder/file or not. Defaults to False.

        Returns
        -------
        Pipeline:
            The ADS Pipeline instance.
        """
        if not self.data_science_pipeline or not self.id:
            print("Pipeline hasn't been created.")
            return self

        if (
            self.data_science_pipeline.sync().lifecycle_state
            != self.LIFECYCLE_STATE_ACTIVE
        ):
            print("Pipeline hasn't been created or not in ACTIVE state.")
            return self

        pipeline_folder = os.path.join(to_dir, self.id)
        if not os.path.exists(pipeline_folder):
            print("Creating directory: " + pipeline_folder)
            os.mkdir(pipeline_folder)
        elif not override_if_exists:
            print(
                f"Folder {pipeline_folder} already exists. Set override_if_exists to True to override."
            )
            return self

        for step in self.step_details:
            if step.kind == "CUSTOM_SCRIPT":
                res = self.data_science_pipeline.client.get_step_artifact_content(
                    self.id, step.name
                )

                if not res or not res.data or not res.data.raw:
                    print(f"Failed to download {step.name} artifact.")
                    return self

                content_disposition = res.headers.get("Content-Disposition", "")
                artifact_name = str(content_disposition).replace(
                    "attachment; filename=", ""
                )
                step_folder = os.path.join(pipeline_folder, step.name)
                if not os.path.exists(step_folder):
                    print("Creating directory: " + step_folder)
                    os.mkdir(step_folder)
                elif not override_if_exists:
                    print(
                        f"Folder {step_folder} already exists. Set override_if_exists to True to override."
                    )
                    continue
                file_name = os.path.join(step_folder, artifact_name)
                with open(file_name, "wb") as f:
                    f.write(res.data.raw.read())

        return self

    def _populate_step_artifact_content(self):
        """Populates artifact information to CUSTOM_SCRIPT step.
        This method is only invoked when the existing pipeline needs to be loaded.
        """
        if (
            not self.data_science_pipeline
            or not self.id
            or self.status != self.LIFECYCLE_STATE_ACTIVE
        ):
            return

        for step in self.step_details:
            if step.kind == "CUSTOM_SCRIPT":
                artifact_name = self._artifact_content_map.get(step.name)
                if not artifact_name:
                    res = self.data_science_pipeline.client.get_step_artifact_content(
                        self.id, step.name
                    )
                    content_disposition = res.headers.get("Content-Disposition", "")
                    artifact_name = str(content_disposition).replace(
                        "attachment; filename=", ""
                    )
                    self._artifact_content_map[step.name] = artifact_name
                if isinstance(step.runtime, ScriptRuntime):
                    step.runtime.with_script(artifact_name)
                elif isinstance(step.runtime, PythonRuntime):
                    step.runtime.with_working_dir(artifact_name)
                elif isinstance(step.runtime, NotebookRuntime):
                    step.runtime.with_notebook(artifact_name)
                elif isinstance(step.runtime, GitPythonRuntime):
                    step.runtime.with_source(artifact_name)

    @classmethod
    def from_ocid(cls, ocid: str) -> "Pipeline":
        """Creates a pipeline by OCID.

        Parameters
        ----------
        ocid: str
            The OCID of pipeline.

        Returns
        -------
        Pipeline:
            The Pipeline instance.
        """
        pipeline = DataSciencePipeline.from_ocid(ocid).build_ads_pipeline()
        pipeline._populate_step_artifact_content()
        return pipeline

    @classmethod
    def from_id(cls, id: str) -> "Pipeline":
        """Creates a pipeline by OCID.

        Parameters
        ----------
        id: str
            The OCID of pipeline.

        Returns
        -------
        Pipeline:
            The Pipeline instance.
        """
        return cls.from_ocid(id)

    def __create_service_log(self) -> "Pipeline":
        """Creates a service log for pipeline.

        Returns
        -------
        Pipeline:
            The ADS Pipeline instance.
        """
        if not self.log_group_id:
            raise ValueError(
                "Log group OCID is not specified for this pipeline. Call with_log_group_id to add it."
            )
        if not self.id:
            raise ValueError("Pipeline is not created yet. Call the create method.")

        oci_service = oci.logging.models.OciService(
            service=self.CONST_SERVICE,
            resource=self.id,
            category=self.CONST_SERVICE_LOG_CATEGORY,
        )

        archiving = oci.logging.models.Archiving(is_enabled=False)

        configuration = oci.logging.models.Configuration(
            source=oci_service, compartment_id=self.compartment_id, archiving=archiving
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        service_logging = OCILog(
            display_name=self.name + f"-{timestamp}",
            log_group_id=self.log_group_id,
            log_type="SERVICE",
            configuration=configuration,
            annotation="service",
        ).create()

        self.__set_service_logging_resource(service_logging)
        self.set_spec(self.CONST_SERVICE_LOG_ID, self.service_logging.id)
        return self

    def __set_service_logging_resource(self, service_logging: OCILog):
        """Sets the service logging for pipeline.

        Parameters
        ----------
        service_logging: OCILog
            An OCILog instance containing the service logging resources.
        """
        self.service_logging = service_logging

    def _convert_step_details_to_dag(
        self, step_details: List["PipelineStep"] = []
    ) -> List[str]:
        """Converts step_details to the DAG representation.

        Parameters
        ----------
        step_details: list
            A list of PipelineStep objects, default to empty list.

        Returns
        -------
        List
            A list of str representing step dependencies.
        """
        dag_list = []
        for step in step_details:
            if isinstance(step, PipelineStep):
                step = step.to_dict()["spec"]
                step_name = step["name"]
            else:
                step_name = step["stepName"]

            if not step["dependsOn"]:
                continue
            if len(step["dependsOn"]) == 1:
                dag = step["dependsOn"][0] + " >> " + step_name
            else:
                dag = "(" + ", ".join(step["dependsOn"]) + ") >> " + step_name
            dag_list.append(dag)

        return dag_list

    def __pipeline_details(self) -> dict:
        """Converts pipeline attributes to a dictionary.

        Returns
        -------
        dict:
            A dictionary that contains pipeline details.
        """
        pipeline_details = copy.deepcopy(self._spec)
        pipeline_details.pop(self.CONST_ENABLE_SERVICE_LOG, None)
        pipeline_details.pop(self.CONST_SERVICE_LOG_ID, None)
        pipeline_configuration_details = self.__pipeline_configuration_details(
            pipeline_details
        )
        if pipeline_configuration_details:
            pipeline_details[
                self.CONST_CONFIGURATION_DETAILS
            ] = pipeline_configuration_details

        pipeline_log_configuration_details = self.__pipeline_log_configuration_details(
            pipeline_details
        )
        if pipeline_log_configuration_details:
            pipeline_details[
                self.CONST_LOG_CONFIGURATION_DETAILS
            ] = pipeline_log_configuration_details

        pipeline_infrastructure_configuration_details = (
            self.__pipeline_infrastructure_configuration_details(pipeline_details)
        )
        if pipeline_infrastructure_configuration_details:
            pipeline_details[
                self.CONST_INFRA_CONFIG_DETAILS
            ] = pipeline_infrastructure_configuration_details

        if self.id:
            pipeline_details[self.CONST_PIPELINE_ID] = self.id
            pipeline_details.pop(self.CONST_ID)

        step_details_list = self.__step_details(pipeline_details)
        pipeline_details[self.CONST_STEP_DETAILS] = step_details_list

        return pipeline_details

    def __pipeline_configuration_details(self, pipeline_details: Dict) -> dict:
        """Converts pipeline configuration details to a dictionary.

        Parameters
        ----------
        pipeline_details: dict
            A dictionary that contains pipeline details.

        Returns
        -------
        dict:
            A dictionary that contains pipeline configuration details.
        """
        pipeline_configuration_details = {}
        if self.maximum_runtime_in_minutes:
            pipeline_configuration_details[
                self.CONST_MAXIMUM_RUNTIME_IN_MINUTES
            ] = self.maximum_runtime_in_minutes
            pipeline_details.pop(self.CONST_MAXIMUM_RUNTIME_IN_MINUTES)
        if self.environment_variable:
            pipeline_configuration_details[
                self.CONST_ENVIRONMENT_VARIABLES
            ] = self.environment_variable
            pipeline_details.pop(self.CONST_ENVIRONMENT_VARIABLES)
        if self.argument:
            pipeline_configuration_details[
                self.CONST_COMMAND_LINE_ARGUMENTS
            ] = self.argument
            pipeline_details.pop(self.CONST_COMMAND_LINE_ARGUMENTS)
        pipeline_configuration_details[self.CONST_TYPE] = "DEFAULT"
        return pipeline_configuration_details

    def __pipeline_log_configuration_details(self, pipeline_details: Dict) -> dict:
        """Converts pipeline log configuration details to a dictionary.

        Parameters
        ----------
        pipeline_details: dict
            A dictionary that contains pipeline details.

        Returns
        -------
        dict:
            A dictionary that contains pipeline log configuration details.
        """
        pipeline_log_configuration_details = {}
        if self.log_id:
            pipeline_log_configuration_details[self.CONST_LOG_ID] = self.log_id
            if not self.log_group_id:
                try:
                    log_obj = OCILog.from_ocid(self.log_id)
                except ResourceNotFoundError:
                    raise ResourceNotFoundError(
                        f"Unable to determine log group ID for Log ({self.log_id})."
                        " The log resource may not exist or You may not have the required permission."
                        " Try to avoid this by specifying the log group ID."
                    )
                self.with_log_group_id(log_obj.log_group_id)

        if self.log_group_id:
            pipeline_log_configuration_details[
                self.CONST_LOG_GROUP_ID
            ] = self.log_group_id

        if self.log_id:
            pipeline_log_configuration_details[self.CONST_ENABLE_LOGGING] = True
            pipeline_log_configuration_details[
                self.CONST_ENABLE_AUTO_LOG_CREATION
            ] = False
            pipeline_details.pop(self.CONST_LOG_ID)
            pipeline_details.pop(self.CONST_LOG_GROUP_ID, None)
        else:
            if self.log_group_id:
                pipeline_log_configuration_details[self.CONST_ENABLE_LOGGING] = True
                pipeline_log_configuration_details[
                    self.CONST_ENABLE_AUTO_LOG_CREATION
                ] = True
                pipeline_details.pop(self.CONST_LOG_GROUP_ID)
            else:
                pipeline_log_configuration_details[self.CONST_ENABLE_LOGGING] = False
                pipeline_log_configuration_details[
                    self.CONST_ENABLE_AUTO_LOG_CREATION
                ] = False
        return pipeline_log_configuration_details

    def __pipeline_infrastructure_configuration_details(
        self, pipeline_details: Dict
    ) -> dict:
        pipeline_infrastructure_details = {}
        if self.shape_name:
            pipeline_infrastructure_details[self.CONST_SHAPE_NAME] = self.shape_name
            pipeline_details.pop(self.CONST_SHAPE_NAME)
        if self.block_storage_size_in_gbs:
            pipeline_infrastructure_details[
                self.CONST_BLOCK_STORAGE_SIZE
            ] = self.block_storage_size_in_gbs
            pipeline_details.pop(self.CONST_BLOCK_STORAGE_SIZE)
        if self.shape_config_details:
            pipeline_infrastructure_details[
                self.CONST_SHAPE_CONFIG_DETAILS
            ] = self.shape_config_details
            pipeline_details.pop(self.CONST_SHAPE_CONFIG_DETAILS)

        return pipeline_infrastructure_details

    def __step_details(self, pipeline_details: Dict) -> list:
        """Converts pipeline step details to a dictionary.

        Parameters
        ----------
        pipeline_details: dict
            A dictionary that contains pipeline details.

        Returns
        -------
        list:
            A list that contains pipeline step details.
        """
        step_details_list = []
        if self.step_details:
            for step in self.step_details:
                step_details = copy.deepcopy(step._spec)
                step_details["stepName"] = step.name
                step_details.pop("name", None)
                if not step.depends_on:
                    step_details[step.CONST_DEPENDS_ON] = []
                if not step.job_id:
                    step_infrastructure_configuration_details = (
                        self.__step_infrastructure_configuration_details(step)
                    )
                    step_details[
                        step.CONST_STEP_INFRA_CONFIG_DETAILS
                    ] = step_infrastructure_configuration_details
                    step_details.pop(step.CONST_INFRASTRUCTURE, None)
                    step_details.pop(step.CONST_RUNTIME, None)

                step_configuration_details = self.__step_configuration_details(
                    pipeline_details, step
                )
                step_details[
                    step.CONST_STEP_CONFIG_DETAILS
                ] = step_configuration_details
                step_details.pop(self.CONST_MAXIMUM_RUNTIME_IN_MINUTES, None)
                step_details.pop(self.CONST_ENVIRONMENT_VARIABLES, None)
                step_details.pop(self.CONST_COMMAND_LINE_ARGUMENTS, None)
                step_details_list.append(step_details)
        return step_details_list

    def __step_infrastructure_configuration_details(self, step) -> dict:
        step_infrastructure_configuration_details = {}
        step_infrastructure_configuration_details[
            "blockStorageSizeInGBs"
        ] = step.infrastructure.block_storage_size
        step_infrastructure_configuration_details[
            "shapeName"
        ] = step.infrastructure.shape_name
        step_infrastructure_configuration_details[
            "shapeConfigDetails"
        ] = step.infrastructure.shape_config_details
        return step_infrastructure_configuration_details

    def __step_configuration_details(self, pipeline_details: Dict, step) -> dict:
        step_configuration_details = {}
        step_configuration_details[self.CONST_TYPE] = "DEFAULT"
        if step.runtime:
            payload = DataScienceJobRuntimeManager(step.infrastructure).translate(
                step.runtime
            )
            if "job_configuration_details" in payload:
                job_configuration_details = payload["job_configuration_details"]
                if "environment_variables" in job_configuration_details:
                    step_configuration_details[
                        self.CONST_ENVIRONMENT_VARIABLES
                    ] = job_configuration_details["environment_variables"]
                if "command_line_arguments" in job_configuration_details:
                    step_configuration_details[
                        self.CONST_COMMAND_LINE_ARGUMENTS
                    ] = job_configuration_details["command_line_arguments"]
                if "maximum_runtime_in_minutes" in job_configuration_details:
                    step_configuration_details[
                        self.CONST_MAXIMUM_RUNTIME_IN_MINUTES
                    ] = job_configuration_details["maximum_runtime_in_minutes"]
        elif step.CONST_STEP_CONFIG_DETAILS in step._spec:
            step_configuration_details = step._spec[step.CONST_STEP_CONFIG_DETAILS]

        if len(step_configuration_details) == 1:
            if step.environment_variable:
                step_configuration_details[
                    self.CONST_ENVIRONMENT_VARIABLES
                ] = step.environment_variable
            if step.argument:
                step_configuration_details[
                    self.CONST_COMMAND_LINE_ARGUMENTS
                ] = step.argument
            if step.maximum_runtime_in_minutes:
                step_configuration_details[
                    self.CONST_MAXIMUM_RUNTIME_IN_MINUTES
                ] = step.maximum_runtime_in_minutes

        if len(step_configuration_details) == 1:
            if self.CONST_CONFIGURATION_DETAILS in pipeline_details:
                step_configuration_details = pipeline_details[
                    self.CONST_CONFIGURATION_DETAILS
                ]

        return step_configuration_details

    def __override_configurations(
        self,
        pipeline_details,
        display_name,
        project_id,
        compartment_id,
        configuration_override_details,
        log_configuration_override_details,
        step_override_details,
        free_form_tags,
        defined_tags,
        system_tags,
    ) -> dict:
        if display_name:
            pipeline_details[self.CONST_DISPLAY_NAME] = display_name

        if project_id:
            pipeline_details[self.CONST_PROJECT_ID] = project_id

        if compartment_id:
            pipeline_details[self.CONST_COMPARTMENT_ID] = compartment_id

        if configuration_override_details:
            configuration_override_details[self.CONST_TYPE] = "DEFAULT"
            pipeline_details[
                self.CONST_CONFIGURATION_OVERRIDE_DETAILS
            ] = self._standardize_spec(configuration_override_details)

        if log_configuration_override_details:
            pipeline_details[
                self.CONST_LOG_CONFIGURATION_OVERRIDE_DETAILS
            ] = self._standardize_spec(log_configuration_override_details)
            log_configuration_override_details = pipeline_details[
                self.CONST_LOG_CONFIGURATION_OVERRIDE_DETAILS
            ]

            if (
                self.CONST_LOG_ID in log_configuration_override_details
                and self.CONST_LOG_GROUP_ID not in log_configuration_override_details
            ):
                try:
                    log_obj = OCILog.from_ocid(
                        log_configuration_override_details[self.CONST_LOG_ID]
                    )
                except ResourceNotFoundError:
                    raise ResourceNotFoundError(
                        f"Unable to determine log group ID for Log ({log_configuration_override_details[self.CONST_LOG_ID]})."
                        " The log resource may not exist or You may not have the required permission."
                        " Try to avoid this by specifying the log group ID."
                    )
                if log_obj and log_obj.log_group_id:
                    log_configuration_override_details[
                        self.CONST_LOG_GROUP_ID
                    ] = log_obj.log_group_id

            if self.CONST_LOG_ID in log_configuration_override_details:
                log_configuration_override_details[self.CONST_ENABLE_LOGGING] = True
                log_configuration_override_details[
                    self.CONST_ENABLE_AUTO_LOG_CREATION
                ] = False
            else:
                if self.CONST_LOG_GROUP_ID in log_configuration_override_details:
                    log_configuration_override_details[self.CONST_ENABLE_LOGGING] = True
                    log_configuration_override_details[
                        self.CONST_ENABLE_AUTO_LOG_CREATION
                    ] = True
                else:
                    log_configuration_override_details[
                        self.CONST_ENABLE_LOGGING
                    ] = False
                    log_configuration_override_details[
                        self.CONST_ENABLE_AUTO_LOG_CREATION
                    ] = False

        if step_override_details:
            step_override_details_list = []
            for step in step_override_details:
                step_detail = {}
                step_detail["stepName"] = step["step_name"]
                step_detail["stepConfigurationDetails"] = self._standardize_spec(
                    step["step_configuration_details"]
                )
                step_override_details_list.append(step_detail)
            pipeline_details[
                self.CONST_STEP_OVERRIDE_DETAILS
            ] = step_override_details_list

        if free_form_tags:
            pipeline_details[self.CONST_FREEFROM_TAGS] = free_form_tags

        if defined_tags:
            pipeline_details[self.CONST_DEFINED_TAGS] = defined_tags

        if system_tags:
            pipeline_details[self.CONST_SYSTEM_TAGS] = system_tags

    # TODO: Needs to improve the validation logic
    # Ticket: https://jira.oci.oraclecorp.com/browse/ODSC-31996
    # @classmethod
    # def from_yaml(cls, uri: str) -> "Pipeline":
    #     pipeline_schema = {}
    #     schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema")
    #     with open(
    #         os.path.join(schema_path, "pipeline_schema.json")
    #     ) as pipeline_schema_file:
    #         pipeline_schema = json.load(pipeline_schema_file)

    #     cs_step_schema = {}
    #     with open(
    #         os.path.join(schema_path, "cs_step_schema.json")
    #     ) as cs_step_schema_file:
    #         cs_step_schema = json.load(cs_step_schema_file)

    #     ml_step_schema = {}
    #     with open(
    #         os.path.join(schema_path, "ml_step_schema.json")
    #     ) as ml_step_schema_file:
    #         ml_step_schema = json.load(ml_step_schema_file)

    #     yaml_dict = yaml.load(Pipeline._read_from_file(uri=uri), Loader=yaml.FullLoader)

    #     pipeline_validator = Validator(pipeline_schema)
    #     if not pipeline_validator.validate(yaml_dict):
    #         raise ValueError(pipeline_validator.errors)

    #     step_details = yaml_dict["spec"]["stepDetails"]
    #     if len(step_details) == 0:
    #         raise ValueError("Pipeline must have at least one step.")

    #     ml_step_validator = Validator(ml_step_schema)
    #     cs_step_validator = Validator(cs_step_schema)
    #     for step in step_details:
    #         if not ml_step_validator.validate(step) and not cs_step_validator.validate(
    #             step
    #         ):
    #             if ml_step_validator.errors:
    #                 raise ValueError(ml_step_validator.errors)
    #             else:
    #                 raise ValueError(cs_step_validator.errors)

    #     return super().from_yaml(uri=uri)

    @classmethod
    def list(cls, compartment_id: Optional[str] = None, **kwargs) -> List["Pipeline"]:
        """
        List pipelines in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to None.
            The OCID of compartment.
            If `None`, the value will be taken from the environment variables.
        kwargs
            Additional keyword arguments for filtering pipelines.
            - project_id: str
            - lifecycle_state: str. Allowed values: "CREATING", "ACTIVE", "DELETING", "FAILED", "DELETED"
            - created_by: str
            - limit: int

        Returns
        -------
        List[Pipeline]
            The list of pipelines.
        """
        result = []
        for item in DataSciencePipeline.list_resource(compartment_id, **kwargs):
            pipeline = item.build_ads_pipeline()
            pipeline._populate_step_artifact_content()
            result.append(pipeline)
        return result

    def run_list(self, **kwargs) -> List[PipelineRun]:
        """Gets a list of runs of the pipeline.

        Returns
        -------
        List[PipelineRun]
            A list of pipeline run instances.
        """
        return PipelineRun.list(
            compartment_id=self.compartment_id, pipeline_id=self.id, **kwargs
        )

    @property
    def status(self) -> Optional[str]:
        """Status of the pipeline.

        Returns
        -------
        str
            Status of the pipeline.
        """
        if self.data_science_pipeline:
            return self.data_science_pipeline.lifecycle_state
        return None

    def init(self) -> "Pipeline":
        """Initializes a starter specification for the Pipeline.

        Returns
        -------
        Pipeline
            The Pipeline instance (self)
        """
        return (
            self.build()
            .with_compartment_id(self.compartment_id or "{Provide a compartment OCID}")
            .with_project_id(self.project_id or "{Provide a project OCID}")
        )


class DataSciencePipeline(OCIDataScienceMixin, oci.data_science.models.Pipeline):
    @classmethod
    def from_ocid(cls, ocid: str) -> "DataSciencePipeline":
        """Gets a datascience pipeline by OCID.

        Parameters
        ----------
        ocid: str
            The OCID of the datascience pipeline.

        Returns
        -------
        DataSciencePipeline
            An instance of DataSciencePipeline.
        """
        return super().from_ocid(ocid)

    def build_ads_pipeline(self) -> "Pipeline":
        """Builds an ADS pipeline from OCI datascience pipeline.

        Returns
        -------
        Pipeline:
            ADS Pipeline instance.
        """
        pipeline_details = self.to_dict()
        ads_pipeline = Pipeline(pipeline_details["displayName"])
        ads_pipeline.data_science_pipeline = self

        for key in pipeline_details:
            if key in ads_pipeline.attribute_set:
                if key == "stepDetails":
                    step_details = []
                    for step in pipeline_details[key]:
                        step_details.append(self.build_ads_pipeline_step(step))
                    ads_pipeline.set_spec(key, step_details)
                elif key in ["freeformTags", "systemTags", "definedTags"]:
                    ads_pipeline.set_spec(key, pipeline_details[key])
                elif type(pipeline_details[key]) is dict:
                    for attribute in pipeline_details[key]:
                        ads_pipeline.set_spec(
                            attribute, pipeline_details[key][attribute]
                        )
                else:
                    ads_pipeline.set_spec(key, pipeline_details[key])

        dag_list = ads_pipeline._convert_step_details_to_dag(
            pipeline_details["stepDetails"]
        )
        ads_pipeline.set_spec(Pipeline.CONST_DAG, dag_list)

        return ads_pipeline

    def build_ads_pipeline_step(self, step: Dict) -> "PipelineStep":
        """Builds an ADS pipeline step from OCI pipeline response.

        Parameters
        ----------
        step: dict
            A dictionary that contains the information of a pipeline step.

        Returns
        -------
        Pipeline:
            ADS PipelineStep instance.
        """
        ads_pipeline_step = PipelineStep(step["stepName"])

        for key in step:
            if key in ads_pipeline_step.attribute_set:
                infrastructure = DataScienceJob()
                if key == ads_pipeline_step.CONST_STEP_INFRA_CONFIG_DETAILS:
                    for attribute in step[key]:
                        infrastructure.set_spec(attribute, step[key][attribute])
                    ads_pipeline_step.set_spec(
                        ads_pipeline_step.CONST_INFRASTRUCTURE, infrastructure
                    )
                elif key == ads_pipeline_step.CONST_STEP_CONFIG_DETAILS:
                    if step["stepType"] == "CUSTOM_SCRIPT":
                        job_configuration_details_dict = {}
                        for attribute in step[key]:
                            job_configuration_details_dict[
                                infrastructure.CONST_JOB_TYPE
                            ] = "DEFAULT"
                            job_configuration_details_dict[attribute] = step[key][
                                attribute
                            ]
                        dsc_job = DSCJob(
                            job_configuration_details=job_configuration_details_dict
                        )
                        runtime = DataScienceJobRuntimeManager(infrastructure).extract(
                            dsc_job
                        )
                        ads_pipeline_step.set_spec(
                            ads_pipeline_step.CONST_RUNTIME, runtime
                        )
                    else:
                        for attribute in step[key]:
                            ads_pipeline_step.set_spec(attribute, step[key][attribute])
                else:
                    ads_pipeline_step.set_spec(key, step[key])
        return ads_pipeline_step

    def create(self, step_details: List, delete_if_fail: bool) -> str:
        """Creates an OCI pipeline.

        Parameters
        ----------
        step_details: list
            List of pipeline step details.

        Returns
        -------
        str:
            The id of OCI pipeline.
        """
        response = self.client.create_pipeline(
            self.to_oci_model(oci.data_science.models.CreatePipelineDetails)
        )
        self.update_from_oci_model(response.data)
        try:
            self.upload_artifact(step_details)
        except Exception as ex:
            if delete_if_fail:
                self.delete(self.id)
            raise ex
        self.step_details = step_details
        return self

    def upload_artifact(self, step_details: List) -> "DataSciencePipeline":
        """Uploads artifacts to pipeline.

        Parameters
        ----------
        step_details: list
            List of pipeline step details.

        Returns
        -------
        DataSciencePipeline:
            DataSciencePipeline instance.
        """
        for step in step_details:
            if step.runtime:
                payload = DataScienceJobRuntimeManager(step.infrastructure).translate(
                    step.runtime
                )
                target_artifact = payload["artifact"]
                if issubclass(target_artifact.__class__, Artifact):
                    with target_artifact as artifact:
                        self.create_step_artifact(artifact.path, step.name)
                else:
                    self.create_step_artifact(target_artifact, step.name)
        return self

    def create_step_artifact(
        self, artifact_path: str, step_name: str
    ) -> "DataSciencePipeline":
        """Creates step artifact.

        Parameters
        ----------
        artifact_path: str
            Local path to artifact.
        step_name: str
            Pipeline step name.

        Returns
        -------
        DataSciencePipeline:
            DataSciencePipeline instance.
        """
        with fsspec.open(artifact_path, "rb") as f:
            self.client.create_step_artifact(
                self.id,
                step_name,
                f,
                content_disposition=f"attachment; filename={os.path.basename(artifact_path)}",
            )
        return self

    def run(
        self, pipeline_details: Dict, service_logging: OCILog = None
    ) -> "PipelineRun":
        """Runs an OCI pipeline.

        Parameters
        ----------
        pipeline_details: dict
            A dictionary that contains pipeline details.
        service_logging: OCILog instance.
            The OCILog instance.

        Returns
        -------
        PipelineRun:
            PipelineRun instance.
        """
        data_science_pipeline_run = PipelineRun(**pipeline_details)
        if service_logging:
            data_science_pipeline_run._set_service_logging_resource(service_logging)
        data_science_pipeline_run.create()

        return data_science_pipeline_run

    def delete(
        self,
        id: str,
        operation_kwargs: Dict = DEFAULT_OPERATION_KWARGS,
        waiter_kwargs: Dict = DEFAULT_WAITER_KWARGS,
    ) -> "DataSciencePipeline":
        """Deletes an OCI pipeline.

        Parameters
        ----------
        id: str
            The ocid of pipeline.
        Parameters
        ----------
        operation_kwargs: dict, optional
            The operational kwargs to be executed when deleting the pipeline.
            Defaults to: {"delete_related_pipeline_runs": True, "delete_related_job_runs": True},
            which will delete the corresponding pipeline runs and job runs.

            The allowed keys are:
            * "delete_related_pipeline_runs": bool, to specify whether to delete related
            PipelineRuns or not.
            * "delete_related_job_runs": bool, to specify whether to delete related JobRuns or not.
            * "allow_control_chars": bool, to indicate whether or not this request should
            allow control characters in the response object. By default, the response will not
            allow control characters in strings
            * "retry_strategy": obj, to apply to this specific operation/call. This will
            override any retry strategy set at the client-level. This should be one of the
            strategies available in the :py:mod:`~oci.retry` module. This operation will not retry
            by default, users can also use the convenient :py:data:`~oci.retry.DEFAULT_RETRY_STRATEGY`
            provided by the SDK to enable retries for it. The specifics of the default retry strategy
            are described `here <https://docs.oracle.com/en-us/iaas/tools/python/latest/sdk_behaviors/retries.html>`__.
            To have this operation explicitly not perform any retries, pass an instance of :py:class:`~oci.retry.NoneRetryStrategy`.
            * "if_match": str, for optimistic concurrency control. In the PUT or DELETE call
            for a resource, set the `if-match` parameter to the value of the etag from a previous
            GET or POST response for that resource. The resource is updated or deleted only if the
            `etag` you provide matches the resource's current `etag` value.
            * "opc_request_id": str, unique Oracle assigned identifier for the request. If you need
            to contact Oracle about a particular request, then provide the request ID.

        waiter_kwargs: dict, optional
            The waiter kwargs to be passed when deleting the pipeline.
            Defaults to: {"max_wait_seconds": 1800}, which will allow a maximum wait time to 1800 seconds to delete the pipeline.
            The allowed keys are:
            * "max_wait_seconds": int, the maximum time to wait, in seconds.
            * "max_interval_seconds": int, the maximum interval between queries, in seconds.
            * "succeed_on_not_found": bool, to determine whether or not the waiter should return
            successfully if the data we're waiting on is not found (e.g. a 404 is returned from the service).
            This defaults to False and so a 404 would cause an exception to be thrown by this function.
            Setting it to True may be useful in scenarios when waiting for a resource to be
            terminated/deleted since it is possible that the resource would not be returned by the a GET call anymore.
            * "wait_callback": A function which will be called each time that we have to do an initial
            wait (i.e. because the property of the resource was not in the correct state,
            or the ``evaluate_response`` function returned False). This function should take two
            arguments - the first argument is the number of times we have checked the resource,
            and the second argument is the result of the most recent check.
            * "fetch_func": A function to be called to fetch the updated state from the server.
            This can be used if the call to check for state needs to be more complex than a single
            GET request. For example, if the goal is to wait until an item appears in a list,
            fetch_func can be a function that paginates through a full list on the server.

        Returns
        -------
        DataSciencePipeline:
            DataSciencePipeline instance.
        """
        self.client_composite.delete_pipeline_and_wait_for_state(
            pipeline_id=id,
            wait_for_states=[
                oci.data_science.models.WorkRequest.STATUS_SUCCEEDED,
                oci.data_science.models.WorkRequest.STATUS_FAILED,
            ],
            operation_kwargs=operation_kwargs,
            waiter_kwargs=waiter_kwargs,
        )
        return self.sync()
