#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from __future__ import annotations

import datetime
import os
import time
import uuid
from io import DEFAULT_BUFFER_SIZE
from typing import Dict, List, Optional, Union

import fsspec
import oci.data_science
import yaml
from ads.common.oci_datascience import DSCNotebookSession, OCIDataScienceMixin
from ads.common.oci_logging import OCILog
from ads.common.oci_resource import ResourceNotFoundError
from ads.jobs.builders.runtimes.artifact import Artifact
from ads.jobs.builders.infrastructure.base import Infrastructure, RunInstance
from ads.jobs.builders.infrastructure.utils import get_value
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    DataScienceJobRuntimeManager,
)


class DSCJob(OCIDataScienceMixin, oci.data_science.models.Job):
    """Represents an OCI Data Science Job
    This class contains all attributes of the oci.data_science.models.Job.
    The main purpose of this class is to link the oci.data_science.models.Job model and the related client methods.
    Mainly, linking the Job model (payload) to Create/Update/Get/List/Delete methods.

    A DSCJob can be initialized by unpacking a the properties stored in a dictionary (payload):

    .. code-block:: python

        job_properties = {
            "display_name": "my_job,
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
                "shapeName": "VM.Standard2.1",
                "blockStorageSizeInGBs": "100",
                "subnetId": "<subnet_ocid>"
            }
        }
        job = DSCJob(**job_payload)

    """

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
        self.job_configuration_details = {
            "jobType": "DEFAULT",
        }
        self.job_infrastructure_configuration_details = {
            "jobInfrastructureType": "STANDALONE",
        }
        super().__init__(**kwargs)

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
            res = self.client.head_job_artifact(self.id)
            content = res.headers.get("content-disposition")
            if content and "filename=" in content:
                self._artifact = content.split("filename=", 1)[-1]
        return self._artifact

    @artifact.setter
    def artifact(self, artifact: Union[str, Artifact]):
        """Sets the job artifact"""
        self._artifact = artifact

    def load_properties_from_env(self) -> None:
        """Loads default properties from the environment"""
        if "NB_SESSION_OCID" in os.environ:
            try:
                nb_session = DSCNotebookSession.from_ocid(os.environ["NB_SESSION_OCID"])
            except:
                # If there is an error loading the notebook infra configurations.
                # Ignore it by setting nb_session to None
                # This will skip loadding the default configure.
                nb_session = None
            if nb_session:
                nb_config = nb_session.notebook_session_configuration_details

                infra = self.job_infrastructure_configuration_details
                if isinstance(infra, dict):
                    infra_type = infra.get("jobInfrastructureType", "STANDALONE")
                    shape_name = infra.get("shapeName", nb_config.shape)
                    block_storage = infra.get(
                        "blockStorageSizeInGBs", nb_config.block_storage_size_in_gbs
                    )
                    subnet_id = infra.get("subnetId", nb_config.subnet_id)
                else:
                    infra_type = (
                        infra.job_infrastructure_type
                        if infra.job_infrastructure_type
                        else "STANDALONE"
                    )
                    shape_name = (
                        infra.shape_name if infra.shape_name else nb_config.shape
                    )
                    block_storage = (
                        infra.block_storage_size_in_gbs
                        if infra.block_storage_size_in_gbs
                        else nb_config.block_storage_size_in_gbs
                    )
                    subnet_id = (
                        infra.subnet_id if infra.subnet_id else nb_config.subnet_id
                    )

                self.job_infrastructure_configuration_details = {
                    "jobInfrastructureType": infra_type,
                    "shapeName": shape_name,
                    "blockStorageSizeInGBs": block_storage,
                    "subnetId": subnet_id,
                }
                if self.project_id is None:
                    self.project_id = nb_session.project_id

        super().load_properties_from_env()

    def _create_with_oci_api(self) -> None:
        res = self.client.create_job(
            self.to_oci_model(oci.data_science.models.CreateJobDetails)
        )
        try:
            self.update_from_oci_model(res.data)
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
        if not self.artifact:
            raise ValueError("Artifact is required to create the job.")
        if not self.display_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            self.display_name = (
                os.path.basename(str(self.artifact)).split(".")[0] + f"-{timestamp}"
            )
        self.load_properties_from_env()
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

    def delete(self) -> DSCJob:
        """Deletes the job and the corresponding job runs.

        Returns
        -------
        DSCJob
            The DSCJob instance (self), which allows chaining additional method.

        """
        runs = self.run_list()
        for run in runs:
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
        return [DataScienceJobRun.from_oci_model(item) for item in items]

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

        If display_name is not specified, it will be generated as "<JOB_NAME>-run-<TIMESTAMP>"

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

        if not kwargs.get("display_name"):
            kwargs["display_name"] = (
                self.display_name
                + "-run-"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M")
            )

        job_attrs = dict(
            project_id=self.project_id,
            display_name=self.display_name
            + "-run-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M"),
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
        run = DataScienceJobRun(**kwargs).create()
        if wait:
            return run.watch()
        return run

    @classmethod
    def from_ocid(cls, ocid) -> DSCJob:
        """Gets a job by OCID

        Parameters
        ----------
        ocid : str
            The OCID of the job.

        Returns
        -------
        DSCJob
            An instance of DSCJob.

        """
        instance = super().from_ocid(ocid)
        res = instance.client.get_job_artifact_content(instance.id)
        # {
        # 'Date': 'Wed, 02 Jun 2021 20:19:06 GMT',
        # 'opc-request-id': 'x',
        # 'Accept-Ranges': 'bytes',
        # 'ETag': 'f320fd3d-fddb-4703-9d74-9736b23c875a--gzip',
        # 'Content-Disposition': 'attachment; filename=ads_my_script.py',
        # 'Content-Encoding': 'gzip',
        # 'Vary': 'Accept-Encoding',
        # 'Last-Modified': 'Wed, 02 Jun 2021 19:55:57 GMT',
        # 'Content-Type': 'application/octet-stream',
        # 'Content-MD5': 'dDE6hVW0wIScavVYqzPnTw==',
        # 'X-Content-Type-Options': 'nosniff',
        # 'Content-Length': '3562'}
        content_disposition = res.headers.get("Content-Disposition", "")
        instance.artifact = str(content_disposition).replace(
            "attachment; filename=", ""
        )
        return instance


class DataScienceJobRun(
    OCIDataScienceMixin, oci.data_science.models.JobRun, RunInstance
):
    """Represents a Data Science Job run"""

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
        return OCILog(id=self.log_id, log_group_id=self.log_details.log_group_id)

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
            date_time = date_time.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "id": str(uuid.uuid4()),
            "time": str(date_time),
            "message": message,
        }

    def logs(self, limit: int = 100) -> list:
        """Gets the logs of the job run.

        Parameters
        ----------
        limit : int, optional
            Limit the number of logs to be returned, by default 100

        Returns
        -------
        list
            A list of log records. Each log record is a dictionary with the following keys: id, time, message.
        """
        log_messages = self.logging.tail(source=self.id, limit=limit)
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
            timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - {status}")
        return status

    def watch(self, interval=3) -> DataScienceJobRun:
        """Watches the job run until it finishes.
        Before the job start running, this method will output the job run status.
        Once the job start running, the logs will be streamed until the job is success, failed or cancelled.

        Parameters
        ----------
        interval : int
             (Default value = 3)
        Time interval between each request to update the logs.

        """

        def stop_condition():
            """Stops the log streaming once the job is in a terminal state."""
            self.sync()
            return self.lifecycle_state in self.TERMINAL_STATES

        if not self.log_id and not self.log_group_id:
            print(
                "Logging is not configured for the job. Watch() will only show job status."
            )

        status = ""
        while not stop_condition():
            status = self._check_and_print_status(status)
            # Break and stream logs if job has log ID and started.
            # Otherwise, keep watching the status until job terminates.
            if self.time_started and self.log_id:
                break
            time.sleep(interval)

        if self.log_id:
            self.logging.stream(
                source=self.id, interval=interval, stop_condition=stop_condition
            )

        self._check_and_print_status(status)

        return self

    def cancel(self) -> DataScienceJobRun:
        """Cancels a job run
        This method will wait for the job run to be canceled before returning.

        Returns
        -------
        self
            The job run instance.
        """
        self.client.cancel_job_run(self.id)
        while self.lifecycle_state != "CANCELED":
            self.sync()
            time.sleep(3)
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
        return yaml.safe_dump(self.to_dict())

    @property
    def job(self):
        """The job instance of this run.

        Returns
        -------
        Job
            An ADS Job instance
        """
        from ads.jobs import Job

        return Job.from_datascience_job(self.job_id)

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


# This is for backward compatibility
DSCJobRun = DataScienceJobRun


class DataScienceJob(Infrastructure):
    """Represents the OCI Data Science Job infrastructure."""

    CONST_PROJECT_ID = "projectId"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_DISPLAY_NAME = "displayName"
    CONST_JOB_TYPE = "jobType"
    CONST_JOB_INFRA = "jobInfrastructureType"
    CONST_SHAPE_NAME = "shapeName"
    CONST_BLOCK_STORAGE = "blockStorageSize"
    CONST_SUBNET_ID = "subnetId"
    CONST_LOG_ID = "logId"
    CONST_LOG_GROUP_ID = "logGroupId"

    attribute_map = {
        CONST_PROJECT_ID: "project_id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_DISPLAY_NAME: "display_name",
        CONST_JOB_TYPE: "job_configuration_details.job_type",
        CONST_JOB_INFRA: "job_infrastructure_configuration_details.job_infrastructure_type",
        CONST_SHAPE_NAME: "job_infrastructure_configuration_details.shape_name",
        CONST_BLOCK_STORAGE: "job_infrastructure_configuration_details.block_storage_size_in_gbs",
        CONST_SUBNET_ID: "job_infrastructure_configuration_details.subnet_id",
        CONST_LOG_ID: "job_log_configuration_details.log_id",
        CONST_LOG_GROUP_ID: "job_log_configuration_details.log_group_id",
    }

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
        super().__init__(spec=spec, **kwargs)
        if not self.job_type:
            self.with_job_type("DEFAULT")
        if not self.job_infrastructure_type:
            self.with_job_infrastructure_type("STANDALONE")
        self.dsc_job = DSCJob()
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

    def _prepare_log_config(self) -> dict:
        if not self.log_group_id and not self.log_id:
            return None
        # Look up log group ID if only the log ID is specified
        if self.log_id and not self.log_group_id:
            try:
                log_obj = OCILog.from_ocid(self.log_id)
            except ResourceNotFoundError:
                raise ResourceNotFoundError(
                    f"Unable to determine log group ID for Log ({self.log_id})."
                    " The log resource may not exist or You may not have the required permission."
                    " Try to avoid this by specifying the log group ID."
                )
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
        self, dsc_job: oci.data_science.models.Job
    ) -> DataScienceJob:
        """Update the properties from an OCI data science job model.

        Parameters
        ----------
        dsc_job: oci.data_science.models.Job
            An OCI data science job model.

        Returns
        -------
        DataScienceJob
            The DataScienceJob instance (self)

        """
        self.dsc_job = dsc_job
        for infra_attr, dsc_attr in self.attribute_map.items():
            value = get_value(dsc_job, dsc_attr)
            if value:
                self._spec[infra_attr] = get_value(dsc_job, dsc_attr)
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
        }
        for snake_attr, camel_attr in attr_map.items():
            if self.get_spec(snake_attr):
                if not dsc_job.job_infrastructure_configuration_details:
                    dsc_job.job_infrastructure_configuration_details = {
                        self.CONST_JOB_INFRA: "STANDALONE",
                        self.CONST_BLOCK_STORAGE: 50,
                    }
                dsc_job.job_infrastructure_configuration_details[
                    camel_attr
                ] = self.get_spec(snake_attr)
        return self

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
        payload["display_name"] = self.name
        payload["job_log_configuration_details"] = self._prepare_log_config()

        self.dsc_job = DSCJob(**payload)
        # Set Job infra to user values after DSCJob initialized the defaults
        self._update_job_infra(self.dsc_job)
        self.dsc_job.create()
        # Update the model from infra after job creation.
        self._update_from_dsc_model(self.dsc_job)
        return self

    def run(
        self, name=None, args=None, env_var=None, freeform_tags=None, wait=False
    ) -> DataScienceJobRun:
        """Runs a job on OCI Data Science job

        Parameters
        ----------
        name : str, optional
            The name of the job run, by default None
        args : str, optional
            Command line arguments for the job run, by default None.
        env_var : dict, optional
            Environment variable for the job run, by default None
        freeform_tags : dict, optional
            Freeform tags for the job run, by default None
        wait : bool, optional
            Indicate if this method should wait for the run to finish before it returns, by default False.

        Returns
        -------
        DSCJobRun
            A Data Science Job Run instance.

        """
        # Runtime in the infrastructure will be None if the job is not created.
        if not self.runtime:
            raise RuntimeError(
                "Job is not created. Call create() to create the job first."
            )
        tags = self.runtime.freeform_tags
        if not tags:
            tags = {}
        if freeform_tags:
            tags.update(freeform_tags)

        return self.dsc_job.run(
            display_name=name,
            command_line_arguments=args,
            environment_variables=env_var,
            freeform_tags=tags,
            wait=wait,
        )

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

    @classmethod
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
        return cls.from_dsc_job(DSCJob.from_ocid(job_id))

    @classmethod
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
            for job in DSCJob.list_resource(compartment_id, **kwargs)
        ]

    @classmethod
    def instance_shapes(cls, compartment_id: str = None) -> list:
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
            A list of dictionaries containing the information of the supported shapes.
        """
        shapes = oci.pagination.list_call_get_all_results(
            DSCJob.init_client().list_job_shapes,
            DSCJob.check_compartment_id(compartment_id),
        ).data
        return shapes
