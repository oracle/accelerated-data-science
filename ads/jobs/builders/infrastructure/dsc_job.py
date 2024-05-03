#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from __future__ import annotations

import datetime
import inspect
import logging
import os
import re
import time
import traceback
import uuid
from io import DEFAULT_BUFFER_SIZE
from string import Template
from typing import Any, Dict, List, Optional, Union

import fsspec
import oci
import oci.data_science
import oci.util as oci_util
from oci.data_science.models import JobInfrastructureConfigurationDetails
from oci.exceptions import ServiceError
import yaml
from ads.common import utils
from ads.common.oci_datascience import DSCNotebookSession, OCIDataScienceMixin
from ads.common.oci_logging import OCILog
from ads.common.oci_resource import ResourceNotFoundError
from ads.jobs.builders.infrastructure.base import Infrastructure, RunInstance
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    DataScienceJobRuntimeManager,
)
from ads.jobs.builders.infrastructure.utils import get_value
from ads.jobs.builders.runtimes.artifact import Artifact
from ads.jobs.builders.runtimes.container_runtime import ContainerRuntime
from ads.jobs.builders.runtimes.python_runtime import GitPythonRuntime

from ads.common.dsc_file_system import (
    OCIFileStorage,
    DSCFileSystemManager,
    OCIObjectStorage,
)
from ads.common.decorator.utils import class_or_instance_method

logger = logging.getLogger(__name__)

SLEEP_INTERVAL = 3
WAIT_SECONDS_AFTER_FINISHED = 90
MAXIMUM_MOUNT_COUNT = 5
FILE_STORAGE_TYPE = "FILE_STORAGE"
OBJECT_STORAGE_TYPE = "OBJECT_STORAGE"


class DSCJob(OCIDataScienceMixin, oci.data_science.models.Job):
    """Represents an OCI Data Science Job
    This class contains all attributes of the oci.data_science.models.Job.
    The main purpose of this class is to link the oci.data_science.models.Job model and the related client methods.
    Mainly, linking the Job model (payload) to Create/Update/Get/List/Delete methods.

    A DSCJob can be initialized by unpacking a the properties stored in a dictionary (payload):

    .. code-block:: python

        job_properties = {
            "display_name": "my_job",
            "job_infrastructure_configuration_details": {"shape_name": "VM.MY_SHAPE"}
        }
        job = DSCJob(**job_properties)

    The properties can also be OCI REST API payload, in which the keys are in camel format.

    .. code-block:: python

        job_payload = {
            "projectId": "<project_ocid>",
            "compartmentId": "<compartment_ocid>",
            "displayName": "<job_name>",
            "jobConfigurationDetails": {
                "jobType": "DEFAULT",
                "commandLineArguments": "pos_arg1 pos_arg2 --key1 val1 --key2 val2",
                "environmentVariables": {
                    "KEY1": "VALUE1",
                    "KEY2": "VALUE2",
                    # User specifies conda env via env var
                    "CONDA_ENV_TYPE" : "service",
                    "CONDA_ENV_SLUG" : "mlcpuv1"
                }
            },
            "jobInfrastructureConfigurationDetails": {
                "jobInfrastructureType": "STANDALONE",
                "shapeName": "VM.Standard.E3.Flex",
                "jobShapeConfigDetails": {
                    "memoryInGBs": 16,
                    "ocpus": 1
                },
                "blockStorageSizeInGBs": "100",
                "subnetId": "<subnet_ocid>"
            }
        }
        job = DSCJob(**job_payload)
    """

    DEFAULT_INFRA_TYPE = (
        JobInfrastructureConfigurationDetails.JOB_INFRASTRUCTURE_TYPE_ME_STANDALONE
    )

    CONST_DEFAULT_BLOCK_STORAGE_SIZE = 50

    def __init__(self, artifact: Union[str, Artifact] = None, **kwargs) -> None:
        """Initialize a DSCJob object.

        Parameters
        ----------
        artifact: str or Artifact
            Job artifact, which can be a path or an Artifact object. Defaults to None.
        kwargs:
            Same as kwargs in oci.data_science.models.Job.
            Keyword arguments are passed into OCI Job model to initialize the properties.

        """
        self._artifact = artifact

        super().__init__(**kwargs)
        if not self.job_configuration_details:
            self.job_configuration_details = {
                "jobType": "DEFAULT",
            }
        if not self.job_infrastructure_configuration_details:
            self.job_infrastructure_configuration_details = {}

    @property
    def artifact(self) -> Union[str, Artifact]:
        """Job artifact.

        Returns
        -------
        str or Artifact
            When creating a job, this be a path or an Artifact object.
            When loading the job from OCI, this will be the filename of the job artifact.
        """
        if self.id and self._artifact is None:
            try:
                res = self.client.head_job_artifact(self.id)
                content = res.headers.get("content-disposition")
                if content and "filename=" in content:
                    self._artifact = content.split("filename=", 1)[-1]
            except ServiceError:
                self._artifact = ""
        return self._artifact

    @artifact.setter
    def artifact(self, artifact: Union[str, Artifact]):
        """Sets the job artifact."""
        self._artifact = artifact

    def _load_infra_from_notebook(self, nb_config):
        """Loads the infrastructure configuration from notebook configuration."""
        infra = self.job_infrastructure_configuration_details
        nb_shape_config_details = oci_util.to_dict(
            getattr(nb_config, "notebook_session_shape_config_details", None) or {}
        )
        if isinstance(infra, dict):
            shape_name = infra.get("shapeName", nb_config.shape)

            # Ignore notebook shape config details if shape names do not match.
            if shape_name != nb_config.shape:
                nb_shape_config_details = {}

            infra_type = infra.get("jobInfrastructureType")
            block_storage = infra.get(
                "blockStorageSizeInGBs", nb_config.block_storage_size_in_gbs
            )
            subnet_id = infra.get(
                "subnetId",
                (
                    nb_config.subnet_id
                    if infra_type
                    != JobInfrastructureConfigurationDetails.JOB_INFRASTRUCTURE_TYPE_ME_STANDALONE
                    else None
                ),
            )
            job_shape_config_details = infra.get("jobShapeConfigDetails", {})
            memory_in_gbs = job_shape_config_details.get(
                "memoryInGBs", nb_shape_config_details.get("memory_in_gbs")
            )
            ocpus = job_shape_config_details.get(
                "ocpus", nb_shape_config_details.get("ocpus")
            )
        else:
            shape_name = (
                infra.shape_name
                if getattr(infra, "shape_name", None)
                else nb_config.shape
            )
            # Ignore notebook shape config details if shape names do not match.
            if shape_name != nb_config.shape:
                nb_shape_config_details = {}

            infra_type = getattr(infra, "job_infrastructure_type", None)

            block_storage = (
                infra.block_storage_size_in_gbs
                if getattr(infra, "block_storage_size_in_gbs", None)
                else nb_config.block_storage_size_in_gbs
            )
            subnet_id = (
                infra.subnet_id
                if getattr(infra, "subnet_id", None)
                else (
                    nb_config.subnet_id
                    if infra_type
                    != JobInfrastructureConfigurationDetails.JOB_INFRASTRUCTURE_TYPE_ME_STANDALONE
                    else None
                )
            )
            job_shape_config_details = oci_util.to_dict(
                getattr(infra, "job_shape_config_details", {}) or {}
            )
            memory_in_gbs = job_shape_config_details.get(
                "memory_in_gbs", nb_shape_config_details.get("memory_in_gbs")
            )
            ocpus = job_shape_config_details.get(
                "ocpus", nb_shape_config_details.get("ocpus")
            )

        self.job_infrastructure_configuration_details = {
            "jobInfrastructureType": infra_type,
            "shapeName": shape_name,
            "blockStorageSizeInGBs": block_storage,
        }
        # ADS does not provide explicit API for setting infrastructure type.
        # If subnet is configured, the type will be set to STANDALONE,
        # otherwise ME_STANDALONE
        if subnet_id:
            self.job_infrastructure_configuration_details.update(
                {
                    "subnetId": subnet_id,
                    "jobInfrastructureType": JobInfrastructureConfigurationDetails.JOB_INFRASTRUCTURE_TYPE_STANDALONE,
                }
            )
        else:
            self.job_infrastructure_configuration_details.update(
                {
                    "jobInfrastructureType": self.DEFAULT_INFRA_TYPE,
                }
            )

        # Specify shape config details
        if memory_in_gbs or ocpus:
            self.job_infrastructure_configuration_details.update(
                {
                    "jobShapeConfigDetails": {
                        "memoryInGBs": memory_in_gbs,
                        "ocpus": ocpus,
                    }
                }
            )

    def load_properties_from_env(self) -> None:
        """Loads default properties from the environment"""
        if "NB_SESSION_OCID" in os.environ:
            try:
                nb_session = DSCNotebookSession.from_ocid(os.environ["NB_SESSION_OCID"])
            except Exception:
                logger.debug("Failed to load config from notebook.")
                logger.debug(traceback.format_exc())
                # If there is an error loading the notebook infra configurations.
                # Ignore it by setting nb_session to None
                # This will skip loading the default configure.
                nb_session = None
            if nb_session:
                nb_config = getattr(
                    nb_session, "notebook_session_config_details", None
                ) or getattr(nb_session, "notebook_session_configuration_details", None)

                if nb_config:
                    self._load_infra_from_notebook(nb_config)
                if self.project_id is None:
                    self.project_id = nb_session.project_id
        super().load_properties_from_env()

    def load_defaults(self) -> DSCJob:
        self.load_properties_from_env()
        if not self.job_infrastructure_configuration_details:
            self.job_infrastructure_configuration_details = {}
        # Convert the dict to JobInfrastructureConfigurationDetails object
        if isinstance(self.job_infrastructure_configuration_details, dict):
            # Default networking
            if not self.job_infrastructure_configuration_details.get(
                "jobInfrastructureType"
            ):
                self.job_infrastructure_configuration_details[
                    "jobInfrastructureType"
                ] = self.DEFAULT_INFRA_TYPE
            self.job_infrastructure_configuration_details = self.deserialize(
                self.job_infrastructure_configuration_details,
                JobInfrastructureConfigurationDetails.__name__,
            )

        # Default block storage size
        if not self.job_infrastructure_configuration_details.block_storage_size_in_gbs:
            self.job_infrastructure_configuration_details.block_storage_size_in_gbs = (
                self.CONST_DEFAULT_BLOCK_STORAGE_SIZE
            )
        return self

    def _create_with_oci_api(self) -> None:
        oci_model = self.to_oci_model(oci.data_science.models.CreateJobDetails)
        logger.debug(oci_model)
        res = self.client.create_job(oci_model)
        self.update_from_oci_model(res.data)
        if self.lifecycle_state == "ACTIVE":
            return
        try:
            if issubclass(self.artifact.__class__, Artifact):
                with self.artifact as artifact:
                    self.upload_artifact(artifact.path)
            else:
                self.upload_artifact()
        except Exception as ex:
            # Delete the job if upload artifact is failed.
            self.delete()
            raise ex

    def create(self) -> DSCJob:
        """Create the job on OCI Data Science platform

        Returns
        -------
        DSCJob
            The DSCJob instance (self), which allows chaining additional method.

        """
        if not self.display_name:
            if self.artifact:
                timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H:%M.%S")
                self.display_name = (
                    os.path.basename(str(self.artifact)).split(".")[0] + f"-{timestamp}"
                )
            else:
                # Set default display_name if not specified - randomly generated easy to remember name generated
                self.display_name = utils.get_random_name_for_resource()
        try:
            self.load_defaults()
        except Exception:
            logger.exception("Failed to load default properties.")
        # Check compartment ID and project ID before calling the OCI API
        if not self.compartment_id:
            raise ValueError("Specify compartment ID for data science job.")
        if not self.project_id:
            raise ValueError("Specify project ID for data science job.")
        self._create_with_oci_api()
        return self

    def update(self) -> DSCJob:
        """Updates the Data Science Job."""
        raise NotImplementedError("Updating Job is not supported at the moment.")

    def delete(self, force_delete: bool = False) -> DSCJob:
        """Deletes the job and the corresponding job runs.

        Parameters
        ----------
        force_delete : bool, optional, defaults to False
            the deletion fails when associated job runs are in progress, but if force_delete to true, then
            the job run will be canceled, then it will be deleted. In this case, delete job has to wait till
            job has been canceled.

        Returns
        -------
        DSCJob
            The DSCJob instance (self), which allows chaining additional method.

        """
        runs = self.run_list()
        for run in runs:
            if force_delete:
                if run.lifecycle_state in [
                    DataScienceJobRun.LIFECYCLE_STATE_ACCEPTED,
                    DataScienceJobRun.LIFECYCLE_STATE_IN_PROGRESS,
                    DataScienceJobRun.LIFECYCLE_STATE_NEEDS_ATTENTION,
                ]:
                    run.cancel(wait_for_completion=True)
            run.delete()
        self.client.delete_job(self.id)
        return self

    def upload_artifact(self, artifact_path: str = None) -> DSCJob:
        """Uploads the job artifact to OCI

        Parameters
        ----------
        artifact_path : str, optional
            Local path to the job artifact file to be uploaded, by default None.
            If artifact_path is None, the path in self.artifact will be used.

        Returns
        -------
        DSCJob
            The DSCJob instance (self), which allows chaining additional method.

        """
        if not artifact_path:
            artifact_path = self.artifact
        with fsspec.open(artifact_path, "rb") as f:
            self.client.create_job_artifact(
                self.id,
                f,
                content_disposition=f"attachment; filename={os.path.basename(artifact_path)}",
            )
        return self

    def download_artifact(self, artifact_path: str) -> DSCJob:
        """Downloads the artifact from OCI

        Parameters
        ----------
        artifact_path : str
            Local path to store the job artifact.

        Returns
        -------
        DSCJob
            The DSCJob instance (self), which allows chaining additional method.

        """
        res = self.client.get_job_artifact_content(self.id)
        with open(artifact_path, "wb") as f:
            for chunk in res.data.iter_content(chunk_size=DEFAULT_BUFFER_SIZE * 16):
                f.write(chunk)
        return self

    def run_list(self, **kwargs) -> list[DataScienceJobRun]:
        """Lists the runs of this job.

        Parameters
        ----------
        **kwargs :
            Keyword arguments to te passed into the OCI list_job_runs() for filtering the job runs.

        Returns
        -------
        list
            A list of DSCJobRun objects

        """
        items = oci.pagination.list_call_get_all_results(
            self.client.list_job_runs, self.compartment_id, job_id=self.id, **kwargs
        ).data
        return [DataScienceJobRun(**self.auth).from_oci_model(item) for item in items]

    def run(self, **kwargs) -> DataScienceJobRun:
        """Runs the job

        Parameters
        ----------
        **kwargs :
            Keyword arguments for initializing a Data Science Job Run.
            The keys can be any keys in supported by OCI JobConfigurationDetails and JobRun, including:
            * hyperparameter_values: dict(str, str)
            * environment_variables: dict(str, str)
            * command_line_arguments: str
            * maximum_runtime_in_minutes: int
            * display_name: str
            * freeform_tags: dict(str, str)
            * defined_tags: dict(str, dict(str, object))

        If display_name is not specified, it will be generated as "<JOB_NAME>-run-<TIMESTAMP>".

        Returns
        -------
        DSCJobRun
            An instance of DSCJobRun, which can be used to monitor the job run.

        """
        if not self.id:
            self.create()

        swagger_types = (
            oci.data_science.models.DefaultJobConfigurationDetails().swagger_types.keys()
        )
        config_kwargs = {}
        keys = list(kwargs.keys())
        for key in keys:
            if key in swagger_types:
                config_kwargs[key] = kwargs.pop(key)

        # remove timestamp from the job name (added in default names, when display_name not specified by user)
        if self.display_name:
            try:
                datetime.datetime.strptime(self.display_name[-19:], "%Y-%m-%d-%H:%M.%S")
                self.display_name = self.display_name[:-20]
            except ValueError:
                pass

        job_attrs = dict(
            project_id=self.project_id,
            display_name=self.display_name
            + "-run-"
            + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M.%S"),
            job_id=self.id,
            compartment_id=self.compartment_id,
        )

        for key, val in job_attrs.items():
            if not kwargs.get(key):
                kwargs[key] = val

        if config_kwargs:
            config_kwargs["jobType"] = "DEFAULT"
            config_override = kwargs.get("job_configuration_override_details", {})
            config_override.update(config_kwargs)
            kwargs["job_configuration_override_details"] = config_override

        wait = kwargs.pop("wait", False)
        run = DataScienceJobRun(**kwargs, **self.auth).create()
        if wait:
            return run.watch()
        return run


class DataScienceJobRun(
    OCIDataScienceMixin, oci.data_science.models.JobRun, RunInstance
):
    """Represents a Data Science Job run"""

    _DETAILS_LINK = (
        "https://console.{region}.oraclecloud.com/data-science/job-runs/{id}"
    )

    TERMINAL_STATES = [
        oci.data_science.models.JobRun.LIFECYCLE_STATE_SUCCEEDED,
        oci.data_science.models.JobRun.LIFECYCLE_STATE_FAILED,
        oci.data_science.models.JobRun.LIFECYCLE_STATE_CANCELED,
        oci.data_science.models.JobRun.LIFECYCLE_STATE_DELETED,
    ]

    def create(self) -> DataScienceJobRun:
        """Creates a job run"""
        self.load_properties_from_env()
        res = self.client.create_job_run(
            self.to_oci_model(oci.data_science.models.CreateJobRunDetails)
        )
        self.update_from_oci_model(res.data)
        return self

    @property
    def status(self) -> str:
        """Lifecycle status

        Returns
        -------
        str
            Status in a string.
        """
        return self.lifecycle_state

    @property
    def log_id(self) -> str:
        """The log ID from OCI logging service containing the logs from the job run."""
        if not self.log_details:
            return None
        return self.log_details.log_id

    @property
    def log_group_id(self) -> str:
        """The log group ID from OCI logging service containing the logs from the job run."""
        if not self.log_details:
            return None
        return self.log_details.log_group_id

    @property
    def logging(self) -> OCILog:
        """The OCILog object containing the logs from the job run"""
        if not self.log_id:
            raise ValueError("Log OCID is not specified for this job run.")
        # Specifying log group ID when initializing OCILog can reduce the number of API calls.
        auth = self.auth
        if "client_kwargs" in auth and isinstance(auth["client_kwargs"], dict):
            auth["client_kwargs"].pop("service_endpoint", None)
        return OCILog(
            id=self.log_id, log_group_id=self.log_details.log_group_id, **auth
        )

    @property
    def exit_code(self):
        """The exit code of the job run from the lifecycle details.
        Note that,
        None will be returned if the job run is not finished or failed without exit code.
        0 will be returned if job run succeeded.
        """
        if self.lifecycle_state == self.LIFECYCLE_STATE_SUCCEEDED:
            return 0
        if not self.lifecycle_details:
            return None
        match = re.search(r"exit code (\d+)", self.lifecycle_details)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    @staticmethod
    def _format_log(message: str, date_time: datetime.datetime) -> dict:
        """Formats a message as log record with datetime.
        This is used to add additional logs to show job run status change.

        Parameters
        ----------
        message : str
            Log message.
        date_time : datetime or str
            Timestamp for the message

        Returns
        -------
        dict
            log record as a dictionary, including id, time and message as keys.
        """
        if isinstance(date_time, datetime.datetime):
            date_time = date_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return {
            "id": str(uuid.uuid4()),
            "message": message,
            "time": date_time,
        }

    def logs(self, limit: int = None) -> list:
        """Gets the logs of the job run.

        Parameters
        ----------
        limit : int, optional
            Limit the number of logs to be returned.
            Defaults to None. All logs will be returned.

        Returns
        -------
        list
            A list of log records. Each log record is a dictionary with the following keys: id, time, message.
        """
        if self.time_accepted:
            log_messages = self.logging.tail(
                source=self.id, limit=limit, time_start=self.time_accepted
            )
        else:
            log_messages = []
        if self.time_started:
            log_messages.insert(
                0, self._format_log("Job Run STARTED", self.time_started)
            )
        if self.time_accepted:
            log_messages.insert(
                0, self._format_log("Job Run ACCEPTED", self.time_accepted)
            )
        if self.time_finished:
            log_messages.append(
                self._format_log("Job Run FINISHED", self.time_finished)
            )
        return log_messages

    def _job_run_status_text(self) -> str:
        details = f", {self.lifecycle_details}" if self.lifecycle_details else ""
        return f"Job Run {self.lifecycle_state}" + details

    def _check_and_print_status(self, prev_status) -> str:
        status = self._job_run_status_text()
        if status != prev_status:
            if self.lifecycle_state in self.TERMINAL_STATES and self.time_finished:
                timestamp = self.time_finished.strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - {status}")
        return status

    def wait(self, interval: float = SLEEP_INTERVAL):
        """Waits for the job run until if finishes.

        Parameters
        ----------
        interval : float
            Time interval in seconds between each request to update the logs.
            Defaults to 3 (seconds).

        """
        self.sync()
        while self.status not in self.TERMINAL_STATES:
            time.sleep(interval)
            self.sync()
        return self

    def watch(
        self,
        interval: float = SLEEP_INTERVAL,
        wait: float = WAIT_SECONDS_AFTER_FINISHED,
    ) -> DataScienceJobRun:
        """Watches the job run until it finishes.
        Before the job start running, this method will output the job run status.
        Once the job start running,
        the logs will be streamed until the job is success, failed or cancelled.

        Parameters
        ----------
        interval : float
            Time interval in seconds between each request to update the logs.
            Defaults to 3 (seconds).
        wait : float
            Time in seconds to keep updating the logs after the job run finished.
            It may take some time for logs to appear in OCI logging service
            after the job run is finished.
            Defaults to 90 (seconds).

        """

        def stop_condition():
            """Stops the log streaming once the job is in a terminal state."""
            self.sync()
            if self.lifecycle_state not in self.TERMINAL_STATES:
                return False
            # Stop if time_finished is not available.
            if not self.time_finished:
                return True
            # Stop only if time_finished is over 2 minute ago.
            # This is for the time delay between job run stopped and the logs appear in oci logging.
            if (
                datetime.datetime.now(self.time_finished.tzinfo)
                - datetime.timedelta(seconds=wait)
                > self.time_finished
            ):
                return True
            return False

        if not self.log_id and not self.log_group_id:
            print(
                "Logging is not configured for the job. Watch() will only show job status."
            )

        print(f"Job OCID: {self.job.id}")
        print(f"Job Run OCID: {self.id}")

        status = ""
        while not stop_condition():
            status = self._check_and_print_status(status)
            # Break and stream logs if job has log ID and started.
            # Otherwise, keep watching the status until job terminates.
            if self.time_started and self.log_id:
                break
            time.sleep(interval)

        if self.log_id and self.time_accepted:
            count = self.logging.stream(
                source=self.id,
                interval=interval,
                stop_condition=stop_condition,
                time_start=self.time_accepted,
            )
            if not count:
                print(
                    "No logs in the last 14 days. Please set time_start to see older logs."
                )

        self._check_and_print_status(status)

        return self

    def cancel(self, wait_for_completion: bool = True) -> DataScienceJobRun:
        """Cancels a job run

        Parameters
        ----------
        wait_for_completion: bool
            Whether to wait for job run to be cancelled before proceeding.
            Defaults to True.

        Returns
        -------
        self
            The job run instance.
        """
        self.client.cancel_job_run(self.id)
        if wait_for_completion:
            while (
                self.lifecycle_state
                != oci.data_science.models.JobRun.LIFECYCLE_STATE_CANCELED
            ):
                self.sync()
                time.sleep(SLEEP_INTERVAL)
        return self

    def __repr__(self) -> str:
        """Displays the object as YAML."""
        return self.to_yaml()

    def to_yaml(self) -> str:
        """Serializes the object into YAML string.

        Returns
        -------
        str
            YAML stored in a string.
        """
        # Here the job YAML is used as the base for the job run
        job_dict = self.job.to_dict()

        # Update infrastructure from job run
        run_dict = self.to_dict()
        infra_specs = [
            run_dict,
            run_dict.get("jobInfrastructureConfigurationDetails", {}),
            run_dict.get("logDetails", {}),
        ]
        for infra_spec in infra_specs:
            for key in infra_spec:
                if key in job_dict["spec"]["infrastructure"]["spec"]:
                    job_dict["spec"]["infrastructure"]["spec"][key] = infra_spec[key]

        # Update runtime from job run
        from ads.jobs import Job

        job = Job(**self.auth).from_dict(job_dict)
        envs = job.runtime.envs
        run_config_override = run_dict.get("jobConfigurationOverrideDetails", {})
        envs.update(run_config_override.get("environmentVariables", {}))
        job.runtime.with_environment_variable(**envs)
        if run_config_override.get("commandLineArguments"):
            job.runtime.set_spec(
                "args",
                run_config_override.get("commandLineArguments"),
            )

        # Update kind, id and name
        run_dict = job.to_dict()
        run_dict["kind"] = "jobRun"
        run_dict["spec"]["id"] = self.id
        run_dict["spec"]["name"] = self.display_name
        return yaml.safe_dump(run_dict)

    @property
    def job(self):
        """The job instance of this run.

        Returns
        -------
        Job
            An ADS Job instance
        """
        from ads.jobs import Job

        return Job(**self.auth).from_datascience_job(self.job_id)

    def download(self, to_dir):
        """Downloads files from job run output URI to local.

        Parameters
        ----------
        to_dir : str
            Local directory to which the files will be downloaded to.

        Returns
        -------
        DataScienceJobRun
            The job run instance (self)
        """
        self.job.download(to_dir)
        return self

    def delete(self, force_delete: bool = False):
        if force_delete:
            self.cancel(wait_for_completion=True)
        super().delete()
        return


# This is for backward compatibility
DSCJobRun = DataScienceJobRun


class DataScienceJob(Infrastructure):
    """Represents the OCI Data Science Job infrastructure.

    To configure the infrastructure for a Data Science Job::

        infrastructure = (
            DataScienceJob()
            # Configure logging for getting the job run outputs.
            .with_log_group_id("<log_group_ocid>")
            # Log resource will be auto-generated if log ID is not specified.
            .with_log_id("<log_ocid>")
            # If you are in an OCI data science notebook session,
            # the following configurations are not required.
            # Configurations from the notebook session will be used as defaults.
            .with_compartment_id("<compartment_ocid>")
            .with_project_id("<project_ocid>")
            .with_subnet_id("<subnet_ocid>")
            .with_shape_name("VM.Standard.E3.Flex")
            # Shape config details are applicable only for the flexible shapes.
            .with_shape_config_details(memory_in_gbs=16, ocpus=1)
            # Minimum/Default block storage size is 50 (GB).
            .with_block_storage_size(50)
            # A list of file systems to be mounted
            .with_storage_mount(
                {
                    "src" : "<mount_target_ip_address>:<export_path>",
                    "dest" : "<destination_directory_name>"
                }
            )
            # Tags
            .with_freeform_tag(my_tag="my_value")
            .with_defined_tag(**{"Operations": {"CostCenter": "42"}})
        )

    """

    CONST_PROJECT_ID = "projectId"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_DISPLAY_NAME = "displayName"
    CONST_JOB_TYPE = "jobType"
    CONST_JOB_INFRA = "jobInfrastructureType"
    CONST_SHAPE_NAME = "shapeName"
    CONST_BLOCK_STORAGE = "blockStorageSize"
    CONST_SUBNET_ID = "subnetId"
    CONST_SHAPE_CONFIG_DETAILS = "shapeConfigDetails"
    CONST_MEMORY_IN_GBS = "memoryInGBs"
    CONST_OCPUS = "ocpus"
    CONST_LOG_ID = "logId"
    CONST_LOG_GROUP_ID = "logGroupId"
    CONST_STORAGE_MOUNT = "storageMount"
    CONST_FREEFORM_TAGS = "freeformTags"
    CONST_DEFINED_TAGS = "definedTags"

    attribute_map = {
        CONST_PROJECT_ID: "project_id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_DISPLAY_NAME: "display_name",
        CONST_JOB_TYPE: "job_type",
        CONST_JOB_INFRA: "job_infrastructure_type",
        CONST_SHAPE_NAME: "shape_name",
        CONST_BLOCK_STORAGE: "block_storage_size",
        CONST_SUBNET_ID: "subnet_id",
        CONST_SHAPE_CONFIG_DETAILS: "shape_config_details",
        CONST_LOG_ID: "log_id",
        CONST_LOG_GROUP_ID: "log_group_id",
        CONST_STORAGE_MOUNT: "storage_mount",
        CONST_FREEFORM_TAGS: "freeform_tags",
        CONST_DEFINED_TAGS: "defined_tags",
    }

    shape_config_details_attribute_map = {
        CONST_MEMORY_IN_GBS: "memory_in_gbs",
        CONST_OCPUS: "ocpus",
    }

    payload_attribute_map = {
        CONST_PROJECT_ID: "project_id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_DISPLAY_NAME: "display_name",
        CONST_JOB_TYPE: "job_configuration_details.job_type",
        CONST_JOB_INFRA: "job_infrastructure_configuration_details.job_infrastructure_type",
        CONST_SHAPE_NAME: "job_infrastructure_configuration_details.shape_name",
        CONST_BLOCK_STORAGE: "job_infrastructure_configuration_details.block_storage_size_in_gbs",
        CONST_SUBNET_ID: "job_infrastructure_configuration_details.subnet_id",
        CONST_SHAPE_CONFIG_DETAILS: "job_infrastructure_configuration_details.job_shape_config_details",
        CONST_LOG_ID: "job_log_configuration_details.log_id",
        CONST_LOG_GROUP_ID: "job_log_configuration_details.log_group_id",
    }

    snake_to_camel_map = {
        v.split(".", maxsplit=1)[-1]: k for k, v in payload_attribute_map.items()
    }

    storage_mount_type_dict = {
        FILE_STORAGE_TYPE: OCIFileStorage,
        OBJECT_STORAGE_TYPE: OCIObjectStorage,
    }

    auth = {}

    @staticmethod
    def standardize_spec(spec):
        if not spec:
            return {}

        attribute_map = {
            **DataScienceJob.attribute_map,
            **DataScienceJob.shape_config_details_attribute_map,
        }
        snake_to_camel_map = {v: k for k, v in attribute_map.items()}
        snake_to_camel_map = {
            **{v: k for k, v in attribute_map.items()},
            **DataScienceJob.snake_to_camel_map,
        }

        for key in list(spec.keys()):
            if key not in attribute_map and key.lower() in snake_to_camel_map:
                value = spec.pop(key)
                if isinstance(value, dict):
                    spec[snake_to_camel_map[key.lower()]] = (
                        DataScienceJob.standardize_spec(value)
                    )
                else:
                    spec[snake_to_camel_map[key.lower()]] = value
        return spec

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        """Initializes a data science job infrastructure

        Parameters
        ----------
        spec : dict, optional
            Object specification, by default None
        kwargs: dict
            Specification as keyword arguments.
            If spec contains the same key as the one in kwargs, the value from kwargs will be used.
        """
        # Saves a copy of the auth object from the class to the instance.
        # Future changes to the class level Job.auth will not affect the auth of existing instances.
        self.auth = self.auth.copy()
        for key in ["config", "signer", "client_kwargs"]:
            if kwargs.get(key):
                self.auth[key] = kwargs.pop(key)

        self.standardize_spec(spec)
        self.standardize_spec(kwargs)
        super().__init__(spec=spec, **kwargs)
        if not self.job_type:
            self.with_job_type("DEFAULT")
        self.dsc_job = DSCJob(**self.auth)
        self.runtime = None
        self._name = None

    @property
    def name(self) -> str:
        """Display name of the job"""
        if self.dsc_job:
            self._name = self.dsc_job.display_name
        return self._name

    @name.setter
    def name(self, value: str):
        """Sets the display name of the job

        Parameters
        ----------
        value : str
            The display name of the job
        """
        self._name = value
        if self.dsc_job:
            self.dsc_job.display_name = value

    @property
    def job_id(self) -> Optional[str]:
        """The OCID of the job"""
        if self.dsc_job:
            return self.dsc_job.id
        return None

    @property
    def status(self) -> Optional[str]:
        """Status of the job.

        Returns
        -------
        str
            Status of the job.
        """
        if self.dsc_job:
            return self.dsc_job.lifecycle_state
        return None

    def with_project_id(self, project_id: str) -> DataScienceJob:
        """Sets the project OCID

        Parameters
        ----------
        project_id : str
            The project OCID

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)

        """
        return self.set_spec(self.CONST_PROJECT_ID, project_id)

    @property
    def project_id(self) -> Optional[str]:
        """Project OCID"""
        return self.get_spec(self.CONST_PROJECT_ID)

    def with_compartment_id(self, compartment_id: str) -> DataScienceJob:
        """Sets the compartment OCID

        Parameters
        ----------
        compartment_id : str
            The compartment OCID

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)

        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    @property
    def compartment_id(self) -> Optional[str]:
        """The compartment OCID"""
        return self.get_spec(self.CONST_COMPARTMENT_ID)

    def with_job_type(self, job_type: str) -> DataScienceJob:
        """Sets the job type

        Parameters
        ----------
        job_type : str
            Job type as string

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)

        """
        return self.set_spec(self.CONST_JOB_TYPE, job_type)

    @property
    def job_type(self) -> Optional[str]:
        """Job type"""
        return self.get_spec(self.CONST_JOB_TYPE)

    def with_job_infrastructure_type(self, infrastructure_type: str) -> DataScienceJob:
        """Sets the job infrastructure type

        Parameters
        ----------
        infrastructure_type : str
            Job infrastructure type as string

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)

        """
        return self.set_spec(self.CONST_JOB_INFRA, infrastructure_type)

    @property
    def job_infrastructure_type(self) -> Optional[str]:
        """Job infrastructure type"""
        return self.get_spec(self.CONST_JOB_INFRA)

    def with_shape_name(self, shape_name: str) -> DataScienceJob:
        """Sets the shape name for running the job

        Parameters
        ----------
        shape_name : str
            Shape name

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)

        """
        return self.set_spec(self.CONST_SHAPE_NAME, shape_name)

    @property
    def shape_name(self) -> Optional[str]:
        """Shape name"""
        return self.get_spec(self.CONST_SHAPE_NAME)

    def with_block_storage_size(self, size_in_gb: int) -> DataScienceJob:
        """Sets the block storage size in GB

        Parameters
        ----------
        size_in_gb : int
            Block storage size in GB

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)

        """
        return self.set_spec(self.CONST_BLOCK_STORAGE, size_in_gb)

    @property
    def block_storage_size(self) -> int:
        """Block storage size for the job"""
        return self.get_spec(self.CONST_BLOCK_STORAGE)

    def with_subnet_id(self, subnet_id: str) -> DataScienceJob:
        """Sets the subnet ID

        Parameters
        ----------
        subnet_id : str
            Subnet ID

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)

        """
        return self.set_spec(self.CONST_SUBNET_ID, subnet_id)

    @property
    def subnet_id(self) -> str:
        """Subnet ID"""
        return self.get_spec(self.CONST_SUBNET_ID)

    def with_shape_config_details(
        self, memory_in_gbs: float, ocpus: float, **kwargs: Dict[str, Any]
    ) -> DataScienceJob:
        """Sets the details for the job run shape configuration.
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
        DataScienceJob
            The DataScienceJob instance (self)
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
    def shape_config_details(self) -> Dict:
        """The details for the job run shape configuration."""
        return self.get_spec(self.CONST_SHAPE_CONFIG_DETAILS)

    def with_log_id(self, log_id: str) -> DataScienceJob:
        """Sets the log OCID for the data science job.
        If log ID is specified, setting the log group ID (with_log_group_id()) is not strictly needed.
        ADS will look up the log group ID automatically.
        However, this may require additional permission,
        and the look up may not be available for newly created log group.
        Specifying both log ID (with_log_id()) and log group ID (with_log_group_id())
        can avoid such lookup and speed up the job creation.

        Parameters
        ----------
        log_id : str
            Log resource OCID.

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)
        """
        return self.set_spec(self.CONST_LOG_ID, log_id)

    @property
    def log_id(self) -> str:
        """Log OCID for the data science job.

        Returns
        -------
        str
            Log OCID
        """
        return self.get_spec(self.CONST_LOG_ID)

    def with_log_group_id(self, log_group_id: str) -> DataScienceJob:
        """Sets the log group OCID for the data science job.
        If log group ID is specified but log ID is not,
        a new log resource will be created automatically for each job run to store the logs.

        Parameters
        ----------
        log_group_id : str
            Log Group OCID

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)
        """
        return self.set_spec(self.CONST_LOG_GROUP_ID, log_group_id)

    @property
    def log_group_id(self) -> str:
        """Log group OCID of the data science job

        Returns
        -------
        str
            Log group OCID
        """
        return self.get_spec(self.CONST_LOG_GROUP_ID)

    def with_storage_mount(self, *storage_mount: List[dict]) -> DataScienceJob:
        """Sets the file systems to be mounted for the data science job.
        A maximum number of 5 file systems are allowed to be mounted for a single data science job.

        Parameters
        ----------
        storage_mount : List[dict]
            A list of file systems to be mounted.

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)
        """
        storage_mount_list = []
        for item in storage_mount:
            if not isinstance(item, dict):
                raise ValueError(
                    "Parameter `storage_mount` should be a list of dictionaries."
                )
            storage_mount_list.append(item)
        if len(storage_mount_list) > MAXIMUM_MOUNT_COUNT:
            raise ValueError(
                f"A maximum number of {MAXIMUM_MOUNT_COUNT} file systems are allowed to be mounted at this time for a job."
            )
        return self.set_spec(self.CONST_STORAGE_MOUNT, storage_mount_list)

    @property
    def storage_mount(self) -> List[dict]:
        """Files systems that have been mounted for the data science job

        Returns
        -------
        list
            A list of file systems that have been mounted
        """
        return self.get_spec(self.CONST_STORAGE_MOUNT, [])

    def with_freeform_tag(self, **kwargs) -> DataScienceJob:
        """Sets freeform tags

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)
        """
        return self.set_spec(self.CONST_FREEFORM_TAGS, kwargs)

    def with_defined_tag(self, **kwargs) -> DataScienceJob:
        """Sets defined tags

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)
        """
        return self.set_spec(self.CONST_DEFINED_TAGS, kwargs)

    @property
    def freeform_tags(self) -> dict:
        """Freeform tags"""
        return self.get_spec(self.CONST_FREEFORM_TAGS, {})

    @property
    def defined_tags(self) -> dict:
        """Defined tags"""
        return self.get_spec(self.CONST_DEFINED_TAGS, {})

    def _prepare_log_config(self) -> dict:
        if not self.log_group_id and not self.log_id:
            return None
        # Look up log group ID if only the log ID is specified
        if self.log_id and not self.log_group_id:
            try:
                log_obj = OCILog.from_ocid(self.log_id)
            except ResourceNotFoundError as exc:
                raise ResourceNotFoundError(
                    f"Unable to determine log group ID for Log ({self.log_id})."
                    " The log resource may not exist or You may not have the required permission."
                    " Try to avoid this by specifying the log group ID."
                ) from exc
            self.with_log_group_id(log_obj.log_group_id)

        if self.log_group_id and not self.log_id:
            enable_auto_log_creation = True
        else:
            enable_auto_log_creation = False

        log_config = {
            "enable_logging": True,
            "enable_auto_log_creation": enable_auto_log_creation,
        }
        if self.log_id:
            log_config["log_id"] = self.log_id

        if self.log_group_id:
            log_config["log_group_id"] = self.log_group_id
        return log_config

    def _update_from_dsc_model(
        self, dsc_job: oci.data_science.models.Job, overwrite: bool = True
    ) -> DataScienceJob:
        """Update the properties from an OCI data science job model.

        Parameters
        ----------
        dsc_job: oci.data_science.models.Job
            An OCI data science job model.

        overwrite: bool
            Whether to overwrite the existing values.
            If this is set to False, only the empty/None properties will be updated.

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)
        """
        sub_level = {
            self.CONST_SHAPE_CONFIG_DETAILS: self.shape_config_details_attribute_map
        }
        self.dsc_job = dsc_job

        for infra_attr, dsc_attr in self.payload_attribute_map.items():
            value = get_value(dsc_job, dsc_attr)
            if not value:
                continue
            if infra_attr not in sub_level:
                if overwrite or not self._spec.get(infra_attr):
                    self._spec[infra_attr] = value
            else:
                sub_spec = self._spec.get(infra_attr, {})
                self._spec[infra_attr] = {}
                for sub_infra_attr, sub_dsc_attr in sub_level[infra_attr].items():
                    sub_value = get_value(value, sub_dsc_attr)
                    if not sub_value:
                        continue
                    if overwrite or not sub_spec.get(sub_infra_attr):
                        sub_spec[sub_infra_attr] = sub_value
                if sub_spec:
                    self._spec[infra_attr] = sub_spec

        self._update_storage_mount_from_dsc_model(dsc_job, overwrite)
        return self

    def _update_storage_mount_from_dsc_model(
        self, dsc_job: oci.data_science.models.Job, overwrite: bool = True
    ) -> DataScienceJob:
        """Update the mount storage properties from an OCI data science job model.

        Parameters
        ----------
        dsc_job: oci.data_science.models.Job
            An OCI data science job model.

        overwrite: bool
            Whether to overwrite the existing values.
            If this is set to False, only the empty/None properties will be updated.

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)
        """
        storage_mount_list = get_value(
            dsc_job, "job_storage_mount_configuration_details_list"
        )
        if storage_mount_list:
            storage_mount = [
                self.storage_mount_type_dict[
                    file_system.storage_type
                ].update_from_dsc_model(file_system)
                for file_system in storage_mount_list
                if file_system.storage_type in self.storage_mount_type_dict
            ]
            if overwrite or not self.get_spec(self.CONST_STORAGE_MOUNT):
                self.set_spec(self.CONST_STORAGE_MOUNT, storage_mount)
        return self

    def _update_job_infra(self, dsc_job: DSCJob) -> DataScienceJob:
        """Updates the job infrastructure from a DSCJob object.

        Parameters
        ----------
        dsc_job : DSCJob
            A DSCJob instance.

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)

        """
        attr_map = {
            self.CONST_JOB_INFRA: "jobInfrastructureType",
            self.CONST_SHAPE_NAME: "shapeName",
            self.CONST_SUBNET_ID: "subnetId",
            self.CONST_BLOCK_STORAGE: "blockStorageSizeInGBs",
            self.CONST_SHAPE_CONFIG_DETAILS: "jobShapeConfigDetails",
        }

        if not dsc_job.job_infrastructure_configuration_details:
            dsc_job.job_infrastructure_configuration_details = {}

        for snake_attr, camel_attr in attr_map.items():
            value = self.get_spec(snake_attr)
            if value:
                dsc_job.job_infrastructure_configuration_details[camel_attr] = value

        if not dsc_job.job_infrastructure_configuration_details.get(
            "shapeName", ""
        ).endswith("Flex") and dsc_job.job_infrastructure_configuration_details.get(
            "jobShapeConfigDetails"
        ):
            raise ValueError(
                "Shape config is not required for non flex shape from user end."
            )

        if dsc_job.job_infrastructure_configuration_details.get("subnetId"):
            dsc_job.job_infrastructure_configuration_details[
                "jobInfrastructureType"
            ] = JobInfrastructureConfigurationDetails.JOB_INFRASTRUCTURE_TYPE_STANDALONE

        if self.storage_mount:
            if not hasattr(oci.data_science.models, "StorageMountConfigurationDetails"):
                raise EnvironmentError(
                    "Storage mount hasn't been supported in the current OCI SDK installed."
                )
            dsc_job.job_storage_mount_configuration_details_list = [
                DSCFileSystemManager.initialize(file_system)
                for file_system in self.storage_mount
            ]
        return self

    def build(self) -> DataScienceJob:
        self.dsc_job.load_defaults()

        try:
            self.dsc_job.load_defaults()
        except Exception:
            logger.exception("Failed to load default properties.")

        self._update_from_dsc_model(self.dsc_job, overwrite=False)
        return self

    def init(self, **kwargs) -> DataScienceJob:
        """Initializes a starter specification for the DataScienceJob.

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)
        """
        return (
            self.build()
            .with_compartment_id(self.compartment_id or "{Provide a compartment OCID}")
            .with_project_id(self.project_id or "{Provide a project OCID}")
            .with_subnet_id(
                self.subnet_id
                or "{Provide a subnet OCID or remove this field if you use a default networking}"
            )
        )

    def create(self, runtime, **kwargs) -> DataScienceJob:
        """Creates a job with runtime.

        Parameters
        ----------
        runtime : Runtime
            An ADS job runtime.

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)

        """
        if not runtime:
            raise ValueError("Set a valid runtime.")
        payload = DataScienceJobRuntimeManager(self).translate(runtime)
        # Add infra properties to payload
        for attr in ["project_id", "compartment_id"]:
            if getattr(self, attr):
                payload[attr] = getattr(self, attr)

        if self.name:
            display_name = Template(self.name).safe_substitute(runtime.envs)
        elif isinstance(runtime, GitPythonRuntime) or isinstance(
            runtime, ContainerRuntime
        ):
            display_name = utils.get_random_name_for_resource()
        else:
            display_name = None

        payload["display_name"] = display_name
        payload["job_log_configuration_details"] = self._prepare_log_config()
        if not payload.get("freeform_tags"):
            payload["freeform_tags"] = self.freeform_tags
        if not payload.get("defined_tags"):
            payload["defined_tags"] = self.defined_tags

        self.dsc_job = DSCJob(**payload, **self.auth)
        # Set Job infra to user values after DSCJob initialized the defaults
        self._update_job_infra(self.dsc_job)
        self.dsc_job.create()
        # Update the model from infra after job creation.
        self._update_from_dsc_model(self.dsc_job)
        return self

    def run(
        self,
        name=None,
        args=None,
        env_var=None,
        freeform_tags=None,
        defined_tags=None,
        wait=False,
        **kwargs,
    ) -> DataScienceJobRun:
        """Runs a job on OCI Data Science job

        Parameters
        ----------
        name : str, optional
            The name of the job run, by default None.
        args : str, optional
            Command line arguments for the job run, by default None.
        env_var : dict, optional
            Environment variable for the job run, by default None
        freeform_tags : dict, optional
            Freeform tags for the job run, by default None
        defined_tags : dict, optional
            Defined tags for the job run, by default None
        wait : bool, optional
            Indicate if this method should wait for the run to finish before it returns, by default False.
        kwargs
            additional keyword arguments

        Returns
        -------
        DataScienceJobRun
            A Data Science Job Run instance.

        """
        # Runtime in the infrastructure will be None if the job is not created.
        if not self.runtime:
            raise RuntimeError(
                "Job is not created. Call create() to create the job first."
            )

        if not freeform_tags:
            freeform_tags = {}
        runtime_freeform_tags = self.runtime.freeform_tags
        if runtime_freeform_tags:
            freeform_tags.update(runtime_freeform_tags)

        if not defined_tags:
            defined_tags = {}
        runtime_defined_tags = self.runtime.defined_tags
        if runtime_defined_tags:
            defined_tags.update(runtime_defined_tags)

        if name:
            envs = self.runtime.envs
            if env_var:
                envs.update(env_var)
            name = Template(name).safe_substitute(envs)

        kwargs = dict(
            display_name=name,
            command_line_arguments=args,
            environment_variables=env_var,
            freeform_tags=freeform_tags,
            defined_tags=defined_tags,
            wait=wait,
            **kwargs,
        )
        # A Runtime class may define customized run() method.
        # Use the customized method if the run() method is defined by the runtime.
        # Otherwise, use the default run() method defined in this class.
        if hasattr(self.runtime, "run"):
            return self.runtime.run(self.dsc_job, **kwargs)
        return self.dsc_job.run(**kwargs)

    def delete(self) -> None:
        """Deletes a job"""
        self.dsc_job.delete()

    def run_list(self, **kwargs) -> List[DataScienceJobRun]:
        """Gets a list of job runs.

        Parameters
        ----------
        **kwargs :
            Keyword arguments for filtering the job runs.
            These arguments will be passed to OCI API.


        Returns
        -------
        List[DSCJobRun]:
            A list of job runs.

        """
        return self.dsc_job.run_list(**kwargs)

    @classmethod
    def from_dsc_job(cls, dsc_job: DSCJob) -> DataScienceJob:
        """Initialize a DataScienceJob instance from a DSCJob

        Parameters
        ----------
        dsc_job : DSCJob
            An instance of DSCJob

        Returns
        -------
        DataScienceJob
            An instance of DataScienceJob

        """
        instance = cls()
        instance._update_from_dsc_model(dsc_job)
        instance.runtime = DataScienceJobRuntimeManager(instance).extract(dsc_job)
        return instance

    @class_or_instance_method
    def from_id(cls, job_id: str) -> DataScienceJob:
        """Gets an existing job using Job OCID

        Parameters
        ----------
        job_id : str
            Job OCID


        Returns
        -------
        DataScienceJob
            An instance of DataScienceJob

        """
        return cls.from_dsc_job(DSCJob(**cls.auth).from_ocid(job_id))

    @class_or_instance_method
    def from_dict(cls, obj_dict: dict):
        """Initialize the object from a Python dictionary"""
        if inspect.isclass(cls):
            job_cls = cls
        else:
            job_cls = cls.__class__
        return job_cls(spec=obj_dict.get("spec"), **cls.auth)

    @class_or_instance_method
    def list_jobs(cls, compartment_id: str = None, **kwargs) -> List[DataScienceJob]:
        """Lists all jobs in a compartment.

        Parameters
        ----------
        compartment_id : str, optional
            The compartment ID for running the jobs, by default None.
            This is optional in a OCI Data Science notebook session.
            If this is not specified, the compartment ID of the notebook session will be used.
        **kwargs :
            Keyword arguments to be passed into OCI list_jobs API for filtering the jobs.

        Returns
        -------
        List[DataScienceJob]
            A list of DataScienceJob object.

        """
        return [
            cls.from_dsc_job(job)
            for job in DSCJob(**cls.auth).list_resource(compartment_id, **kwargs)
        ]

    @class_or_instance_method
    def instance_shapes(cls, compartment_id: str = None, **kwargs) -> list:
        """Lists the supported shapes for running jobs in a compartment.

        Parameters
        ----------
        compartment_id : str, optional
            The compartment ID for running the jobs, by default None.
            This is optional in a OCI Data Science notebook session.
            If this is not specified, the compartment ID of the notebook session will be used.

        Returns
        -------
        list
            A list of oci.data_science.models.JobShapeSummary objects
            containing the information of the supported shapes.

        Examples
        --------
        To get a list of shape names::

            shapes = DataScienceJob.fast_launch_shapes(
                compartment_id=os.environ["PROJECT_COMPARTMENT_OCID"]
            )
            shape_names = [shape.name for shape in shapes]

        """
        shapes = oci.pagination.list_call_get_all_results(
            DSCJob(**cls.auth).init_client().list_job_shapes,
            DSCJob.check_compartment_id(compartment_id),
            **kwargs,
        ).data
        return shapes

    @class_or_instance_method
    def fast_launch_shapes(cls, compartment_id: str = None, **kwargs) -> list:
        """Lists the supported fast launch shapes for running jobs in a compartment.

        Parameters
        ----------
        compartment_id : str, optional
            The compartment ID for running the jobs, by default None.
            This is optional in a OCI Data Science notebook session.
            If this is not specified, the compartment ID of the notebook session will be used.

        Returns
        -------
        list
            A list of oci.data_science.models.FastLaunchJobConfigSummary objects
            containing the information of the supported shapes.

        Examples
        --------
        To get a list of shape names::

            shapes = DataScienceJob.fast_launch_shapes(
                compartment_id=os.environ["PROJECT_COMPARTMENT_OCID"]
            )
            shape_names = [shape.shape_name for shape in shapes]

        """
        shapes = oci.pagination.list_call_get_all_results(
            DSCJob(**cls.auth).init_client().list_fast_launch_job_configs,
            DSCJob.check_compartment_id(compartment_id),
            **kwargs,
        ).data
        return shapes
