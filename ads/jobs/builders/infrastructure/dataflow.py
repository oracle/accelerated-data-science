#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import copy
import datetime
import io
import json
import logging
import os
import time
import urllib.parse
from typing import Dict, List, Optional

import fsspec
import oci.data_flow
import ocifs
import yaml
from ads.common.auth import default_signer
from ads.common.oci_client import OCIClientFactory
from ads.common.oci_mixin import OCIModelMixin
from ads.config import OCI_REGION_METADATA
from ads.jobs.builders.infrastructure.base import Infrastructure, RunInstance
from ads.jobs.builders.infrastructure.utils import batch_convert_case, normalize_config
from ads.jobs.builders.runtimes.python_runtime import DataFlowRuntime
from IPython.display import HTML, display
from oci.data_flow.models import CreateApplicationDetails, CreateRunDetails
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataFlowApp(OCIModelMixin, oci.data_flow.models.Application):
    @classmethod
    def init_client(cls, **kwargs) -> oci.data_flow.data_flow_client.DataFlowClient:
        return cls._init_client(
            client=oci.data_flow.data_flow_client.DataFlowClient, **kwargs
        )

    @property
    def client(self) -> oci.data_flow.data_flow_client.DataFlowClient:
        return super().client

    def create(self) -> "DataFlowApp":
        """
        Create a Data Flow application.

        Returns
        -------
        DataFlowApp
            a DataFlowApp instance

        """
        resp = self.client.create_application(
            self.to_oci_model(CreateApplicationDetails)
        )
        logger.debug(f"Created a DataFlow Application {resp.data}")
        for k, v in json.loads(repr(resp.data)).items():
            setattr(self, k, v)
        return self

    def delete(self) -> None:
        """
        Delete a Data Flow application.

        Returns
        -------
        None
        """
        self.application = self.client.delete_application(self.id)
        self.lifecycle_state = oci.data_flow.models.Application.LIFECYCLE_STATE_DELETED

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


class DataFlowRun(OCIModelMixin, oci.data_flow.models.Run, RunInstance):

    TERMINATED_STATES = [
        oci.data_flow.models.Run.LIFECYCLE_STATE_CANCELED,
        oci.data_flow.models.Run.LIFECYCLE_STATE_FAILED,
        oci.data_flow.models.Run.LIFECYCLE_STATE_SUCCEEDED,
    ]

    @classmethod
    def init_client(cls, **kwargs) -> oci.data_flow.data_flow_client.DataFlowClient:
        return cls._init_client(
            client=oci.data_flow.data_flow_client.DataFlowClient, **kwargs
        )

    @property
    def client(self) -> oci.data_flow.data_flow_client.DataFlowClient:
        return super().client

    def create(self) -> "DataFlowRun":
        """
        Create a Data Flow run.

        Returns
        -------
        DataFlowRun
            a DataFlowRun instance
        """
        resp = self.client.create_run(self.to_oci_model(CreateRunDetails))
        logger.debug(f"Created a DataFlow Run {resp.data}")
        for k, v in json.loads(repr(resp.data)).items():
            setattr(self, k, v)
        return self

    @property
    def status(self) -> str:
        """
        Show status (lifecycle state) of a run.

        Returns
        -------
        str
            status of the run
        """
        if self.id:
            resp = self.client.get_run(self.id)
            self.lifecycle_state = resp.data.lifecycle_state
        return self.lifecycle_state

    @property
    def logs(self) -> "DataFlowLogs":
        """
        Show logs from a run.
        There are three types of logs: application log, driver log and executor log,
        each with stdout and stderr separately.
        To access each type of logs,
        >>> dfr.logs.application.stdout
        >>> dfr.logs.driver.stderr

        Returns
        -------
        DataFlowLogs
            an instance of DataFlowLogs
        """
        return DataFlowLogs(run_id=self.id)

    def wait(self, interval: int = 3) -> "DataFlowRun":
        """
        Wait for a run to terminate.

        Parameters
        ----------
        interval: int, optional
            interval to wait before probing again

        Returns
        -------
        DataFlowRun
            a DataFlowRun instance
        """
        current = self.status
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"{timestamp} - {self.status}")
        while current not in self.TERMINATED_STATES:
            time.sleep(interval)
            if self.status != current:
                timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"{timestamp} - {self.status}")
                current = self.status
        return self

    def watch(self, interval: int = 3) -> "DataFlowRun":
        """This is an alias of `wait()` method. It waits for a run to terminate.

        Parameters
        ----------
        interval: int, optional
            interval to wait before probing again

        Returns
        -------
        DataFlowRun
            a DataFlowRun instance
        """
        logger.info(
            "This is an alias of `wait`. Logs are available after run completes."
        )
        return self.wait(interval=interval)

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
        run = self.to_dict()
        run["lifecycleState"] = self.status
        return yaml.safe_dump(run)

    def delete(self) -> None:
        """
        Cancel a Data Flow run if it is not yet terminated.

        Returns
        -------
        None
        """
        if self.status not in self.TERMINATED_STATES:
            self.client.delete_run(self.id)
            self.lifecycle_state = oci.data_flow.models.Run.LIFECYCLE_STATE_CANCELING

    @property
    def run_details_link(self):
        """
        Link to run details page in OCI console

        Returns
        -------
        DisplayHandle
            html display
        """
        signer = default_signer()
        if "region" in signer["config"]:
            region = signer["config"]["region"]
        else:
            region = json.loads(OCI_REGION_METADATA)["regionIdentifier"]
        return (
            f"https://console.{region}.oraclecloud.com/data-flow/runs/details/{self.id}"
        )


class _Log:
    def __init__(self, run_id, src_type, output_type):
        self.run_id = run_id
        self.src_type = src_type
        self.output_type = output_type
        self._cache = None

    @property
    def client(self):
        return OCIClientFactory(**default_signer()).dataflow

    def _get_logs(self, n=None, reverse=False):
        if not self._cache:
            logs = self.client.list_run_logs(self.run_id).data
            lines = []
            for log in logs:
                if log.type == self.output_type and self.src_type == log.source:
                    lines.extend(
                        self.client.get_run_log(self.run_id, log.name)
                        .data.text.lstrip("\x00")
                        .rstrip("\n")
                        .splitlines()
                    )
            self._cache = lines
        if n and reverse:
            return "\n".join(self._cache[-n:])
        elif n and not reverse:
            return "\n".join(self._cache[:n])
        else:
            return "\n".join(self._cache)

    def __repr__(self):
        return self._get_logs()

    def head(self, n=None):
        return self._get_logs(n=n)

    def tail(self, n=None):
        return self._get_logs(n=n, reverse=True)

    def download(self, path):
        with open(path, mode="w") as f:
            f.write(self._get_logs())


class _DataFlowLog:
    def __init__(self, run_id, src_type):
        self.run_id = run_id
        self.src_type = src_type

    @property
    def stdout(self):
        return _Log(self.run_id, self.src_type, "STDOUT")

    @property
    def stderr(self):
        return _Log(self.run_id, self.src_type, "STDERR")


class DataFlowLogs:
    def __init__(self, run_id):
        self.run_id = run_id

    @property
    def application(self):
        return _DataFlowLog(run_id=self.run_id, src_type="APPLICATION")

    @property
    def driver(self):
        return _DataFlowLog(run_id=self.run_id, src_type="DRIVER")

    @property
    def executor(self):
        return _DataFlowLog(run_id=self.run_id, src_type="EXECUTOR")


class DataFlow(Infrastructure):
    def __init__(self, spec: dict = None):
        defaults = self._load_default_properties()
        if spec is None:
            super(DataFlow, self).__init__(defaults)
        else:
            spec = {k: v for k, v in spec.items() if f"with_{k}" in self.__dir__()}
            defaults.update(spec)
            super(DataFlow, self).__init__(defaults)
        self.df_app = DataFlowApp(**self._spec)
        self.runtime = None
        self._name = None

    @staticmethod
    def _load_default_properties() -> dict:
        """
        Load default properties from environment variables, notebook session, etc.

        Returns
        -------
        dict
            a dictionary of default properties
        """
        allowed_shapes = [
            "VM.Standard2.1",
            "VM.Standard2.2",
            "VM.Standard2.4",
            "VM.Standard2.8",
            "VM.Standard2.16",
        ]
        defaults = {}
        if "NB_SESSION_COMPARTMENT_OCID" in os.environ:
            defaults["compartment_id"] = os.environ["NB_SESSION_COMPARTMENT_OCID"]
        if "NB_SESSION_OCID" in os.environ:
            dsc_client = OCIClientFactory(**default_signer()).data_science
            try:
                nb_session = dsc_client.get_notebook_session(
                    os.environ["NB_SESSION_OCID"]
                ).data
                nb_config = nb_session.notebook_session_configuration_details
                if nb_config.shape in allowed_shapes:
                    defaults["driver_shape"] = nb_config.shape
                    logger.debug(f"Set driver shape to {nb_config.shape}")
                    defaults["executor_shape"] = nb_config.shape
                    logger.debug(f"Set executor shape to {nb_config.shape}")

            except Exception as e:
                logger.warning(
                    f"Error fetching details about Notebook session: {os.environ['NB_SESSION_OCID']}. {e}"
                )

        defaults["language"] = "PYTHON"
        defaults["spark_version"] = "2.4.4"
        defaults["num_executors"] = 1
        logger.debug("Set spark version to be 2.4.4.")
        return defaults

    @property
    def name(self) -> str:
        """Display name of the job"""
        return self._name

    @name.setter
    def name(self, value: str):
        """Sets the display name of the job

        Parameters
        ----------
        value : str
            The display name of the job
        """
        if self.df_app:
            self.df_app.display_name = value
        self._name = value

    @property
    def job_id(self) -> Optional[str]:
        """The OCID of the job"""
        return self.get_spec("id")

    def with_compartment_id(self, id: str) -> "DataFlow":
        """
        Set compartment id for a Data Flow job.

        Parameters
        ----------
        id: str
            compartment id

        Returns
        -------
        DataFlow
            the Data Flow instance itself
        """
        return self.set_spec("compartment_id", id)

    def with_configuration(self, configs: dict) -> "DataFlow":
        """
        Set configuration for a Data Flow job.

        Parameters
        ----------
        configs: dict
            dictionary of configurations

        Returns
        -------
        DataFlow
            the Data Flow instance itself
        """
        return self.set_spec("configuration", configs)

    def with_driver_shape(self, shape: str) -> "DataFlow":
        """
        Set driver shape for a Data Flow job.

        Parameters
        ----------
        shape: str
            driver shape

        Returns
        -------
        DataFlow
            the Data Flow instance itself
        """
        return self.set_spec("driver_shape", shape)

    def with_executor_shape(self, shape: str) -> "DataFlow":
        """
        Set executor shape for a Data Flow job.

        Parameters
        ----------
        shape: str
            executor shape

        Returns
        -------
        DataFlow
            the Data Flow instance itself
        """
        return self.set_spec("executor_shape", shape)

    def with_language(self, lang: str) -> "DataFlow":
        """
        Set language for a Data Flow job.

        Parameters
        ----------
        lang: str
            language for the job

        Returns
        -------
        DataFlow
            the Data Flow instance itself
        """
        return self.set_spec("language", lang)

    def with_metastore_id(self, id: str) -> "DataFlow":
        """
        Set Hive metastore id for a Data Flow job.

        Parameters
        ----------
        id: str
            metastore id

        Returns
        -------
        DataFlow
            the Data Flow instance itself
        """

        return self.set_spec("metastore_id", id)

    def with_logs_bucket_uri(self, uri: str) -> "DataFlow":
        """
        Set logs bucket uri for a Data Flow job.

        Parameters
        ----------
        uri: str
            uri to logs bucket

        Returns
        -------
        DataFlow
            the Data Flow instance itself
        """
        return self.set_spec("logs_bucket_uri", uri)

    def with_num_executors(self, n: int) -> "DataFlow":
        """
        Set number of executors for a Data Flow job.

        Parameters
        ----------
        n: int
            number of executors

        Returns
        -------
        DataFlow
            the Data Flow instance itself
        """
        return self.set_spec("num_executors", n)

    def with_spark_version(self, ver: str) -> "DataFlow":
        """
        Set spark version for a Data Flow job.
        Currently supported versions are 2.4.4 and 3.0.2
        Documentation: https://docs.oracle.com/en-us/iaas/data-flow/using/dfs_getting_started.htm#before_you_begin

        Parameters
        ----------
        ver: str
            spark version

        Returns
        -------
        DataFlow
            the Data Flow instance itself
        """
        return self.set_spec("spark_version", ver)

    def with_warehouse_bucket_uri(self, uri: str) -> "DataFlow":
        """
        Set warehouse bucket uri for a Data Flow job.

        Parameters
        ----------
        uri: str
            uri to warehouse bucket

        Returns
        -------
        DataFlow
            the Data Flow instance itself
        """
        return self.set_spec("warehouse_bucket_uri", uri)

    def with_id(self, id: str) -> "DataFlow":
        """
        Set id for a Data Flow job.

        Parameters
        ----------
        id: str
            id of a job

        Returns
        -------
        DataFlow
            the Data Flow instance itself
        """
        return self.set_spec("id", id)

    def __getattr__(self, item):
        if f"with_{item}" in self.__dir__():
            return self.get_spec(item)
        raise AttributeError(f"Attribute {item} not found.")

    def create(self, runtime: DataFlowRuntime, **kwargs) -> "DataFlow":
        """
        Create a Data Flow job given a runtime.

        Parameters
        ----------
        runtime
            runtime to bind to the Data Flow job
        kwargs
            additional keyword arguments

        Returns
        -------
        DataFlow
            a Data Flow job instance
        """
        if not self.name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            self.name = f"dataflow-{timestamp}"
        payload = copy.deepcopy(self._spec)
        if not runtime.script_uri:
            raise ValueError("script uri must be specified in runtime.")
        if runtime.script_uri.split(":")[0] != "oci":
            if runtime.script_bucket:
                runtime.with_script_uri(
                    self._upload_file(runtime.script_uri, runtime.script_bucket)
                )
            else:
                raise ValueError(
                    "script bucket must be specified if script uri given is local."
                )
        if runtime.archive_uri and runtime.archive_uri.split(":")[0] != "oci":
            if runtime.archive_bucket:
                runtime.with_archive_uri(
                    self._upload_file(runtime.archive_uri, runtime.archive_bucket)
                )
            else:
                raise ValueError(
                    "archive bucket must be specified if archive uri given is local."
                )
        payload.update(
            {
                "display_name": self.name,
                "file_uri": runtime.script_uri,
                "freeform_tags": runtime.freeform_tags,
                "archive_uri": runtime.archive_uri,
            }
        )
        if not payload.get("compartment_id", None):
            raise ValueError(
                "Compartment id is required. Specify compartment id via 'with_compartment_id()'."
            )
        payload.pop("id", None)
        logger.debug(f"Creating a DataFlow Application with payload {payload}")
        self.df_app = DataFlowApp(**payload).create()
        self.with_id(self.df_app.id)
        return self

    @staticmethod
    def _upload_file(local_path, bucket):
        signer = default_signer()
        os_client = OCIClientFactory(**signer).object_storage
        namespace = os_client.get_namespace().data
        parsed = urllib.parse.urlparse(bucket)
        chunk_size = io.DEFAULT_BUFFER_SIZE
        if parsed.scheme == "oci":
            dst_path = os.path.join(bucket, os.path.basename(local_path))
        else:
            dst_path = f"oci://{bucket}@{namespace}/{os.path.basename(local_path)}"
        logger.debug(f"Uploading {local_path} to {dst_path}")
        with fsspec.open(dst_path, mode="wb", **signer) as fto:
            file_system_clz = fsspec.get_filesystem_class(
                urllib.parse.urlparse(local_path).scheme or "file"
            )
            file_size = file_system_clz().info(local_path)["size"]
            with fsspec.open(local_path, mode="rb", encoding=None) as fread:
                with tqdm.wrapattr(
                    fread,
                    "read",
                    desc=f"Uploading file: {local_path} to {dst_path}",
                    total=file_size,
                ) as ffrom:
                    while True:
                        chunk = ffrom.read(chunk_size)
                        if chunk:
                            fto.write(chunk)
                        else:
                            break
        return dst_path

    def run(
        self,
        name: str = None,
        args: List[str] = None,
        env_vars: Dict[str, str] = None,
        freeform_tags: Dict[str, str] = None,
        wait: bool = False,
        **kwargs,
    ) -> DataFlowRun:
        """
        Run a Data Flow job.

        Parameters
        ----------
        name: str, optional
            name of the run
        args: List[str], optional
            list of command line arguments
        env_vars: Dict[str, str], optional
            dictionary of environment variables (not used for data flow)
        freeform_tags: Dict[str, str], optional
            freeform tags
        wait: bool, optional
            whether to wait for a run to terminate
        kwargs
            additional keyword arguments

        Returns
        -------
        DataFlowRun
            a DataFlowRun instance
        """
        payload = kwargs
        if "application_id" not in payload:
            payload["application_id"] = self.id
        if "compartment_id" not in payload:
            payload["compartment_id"] = self.get_spec("compartment_id")
        payload["display_name"] = (
            name
            if name
            else f"{self.name}-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        )
        payload["arguments"] = args if args and len(args) > 0 else None
        payload["freeform_tags"] = freeform_tags
        payload.pop("spark_version", None)
        logger.debug(f"Creating a DataFlow Run with payload {payload}")
        run = DataFlowRun(**payload).create()
        if wait:
            interval = kwargs["interval"] if "interval" in kwargs else 3
            run.wait(interval)
        return run

    def run_list(self, **kwargs) -> List[DataFlowRun]:
        """
        List runs associated with a Data Flow job.

        Parameters
        ----------
        kwargs
         additional arguments for filtering runs.

        Returns
        -------
        List[DataFlowRun]
            list of DataFlowRun instances
        """
        items = oci.pagination.list_call_get_all_results(
            self.df_app.client.list_runs,
            self.compartment_id,
            application_id=self.id,
            **kwargs,
        ).data
        return [DataFlowRun.from_oci_model(item) for item in items]

    def delete(self):
        """
        Delete a Data Flow job and canceling associated runs.

        Returns
        -------
        None
        """
        runs = self.run_list()
        for run in runs:
            run.delete()
        self.df_app.delete()

    @classmethod
    def from_id(cls, id: str) -> "DataFlow":
        """
        Load a Data Flow job given an id.

        Parameters
        ----------
        id: str
            id of the Data Flow job to load

        Returns
        -------
        DataFlow
            a Data Flow job instance
        """
        payload = batch_convert_case(
            yaml.safe_load(repr(DataFlowApp.from_ocid(id))), "snake"
        )
        config = normalize_config(payload)
        name = config.pop("display_name")
        ins = cls(config)
        rt = DataFlowRuntime()
        for k, v in config.items():
            if hasattr(DataFlowRuntime, f"with_{k}"):
                getattr(rt, f"with_{k}")(v)
        ins.runtime = rt
        ins.name = name
        return ins

    @classmethod
    def list_jobs(cls, compartment_id: str = None, **kwargs) -> List["DataFlow"]:
        """
        List Data Flow jobs in a given compartment.

        Parameters
        ----------
        compartment_id: str
            id of that compartment
        kwargs
            additional keyword arguments for filtering jobs

        Returns
        -------
        List[DataFlow]
            list of Data Flow jobs
        """
        return [
            cls.from_id(job.id)
            for job in DataFlowApp.list_resource(compartment_id, **kwargs)
        ]

    def to_dict(self) -> dict:
        """
        Serialize job to a dictionary.

        Returns
        -------
        dict
            serialized job as a dictionary
        """
        return {
            "kind": self.kind,
            "type": self.type,
            "spec": batch_convert_case(self._spec, "camel"),
        }

    @classmethod
    def from_dict(cls, config: dict) -> "DataFlow":
        """
        Load a Data Flow job instance from a dictionary of configurations.

        Parameters
        ----------
        config: dict
            dictionary of configurations

        Returns
        -------
        DataFlow
            a Data Flow job instance
        """
        return cls(spec=batch_convert_case(config["spec"], "snake"))

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
