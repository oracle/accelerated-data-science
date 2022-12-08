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
from typing import Any, Dict, List, Optional

import fsspec
import oci.data_flow
import yaml
from ads.common import utils
from ads.common.auth import default_signer
from ads.common.oci_client import OCIClientFactory
from ads.common.oci_mixin import OCIModelMixin
from ads.common.utils import camel_to_snake, batch_convert_case
from ads.config import OCI_REGION_METADATA
from ads.jobs.builders.infrastructure.base import Infrastructure, RunInstance
from ads.jobs.builders.infrastructure.utils import normalize_config
from ads.jobs.builders.runtimes.python_runtime import DataFlowRuntime
from ads.model.runtime.env_info import InferenceEnvInfo
from oci.data_flow.models import CreateApplicationDetails, CreateRunDetails
import oci.util as oci_util
from tqdm import tqdm

logger = logging.getLogger(__name__)

CONDA_PACK_SUFFIX = "#conda"


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
        print(f"{timestamp} - {self.status}")
        while current not in self.TERMINATED_STATES:
            time.sleep(interval)
            if self.status != current:
                timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"{timestamp} - {self.status}")
                print(f"{timestamp} - {self.status}")
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

    CONST_COMPARTMENT_ID = "compartment_id"
    CONST_CONFIG = "configuration"
    CONST_EXECUTE = "execute"
    CONST_DRIVER_SHAPE = "driver_shape"
    CONST_EXECUTOR_SHAPE = "executor_shape"
    CONST_LANGUAGE = "language"
    CONST_METASTORE_ID = "metastore_id"
    CONST_BUCKET_URI = "logs_bucket_uri"
    CONST_NUM_EXECUTORS = "num_executors"
    CONST_SPARK_VERSION = "spark_version"
    CONST_WAREHOUSE_BUCKET_URI = "warehouse_bucket_uri"
    CONST_DRIVER_SHAPE_CONFIG = "driver_shape_config"
    CONST_EXECUTOR_SHAPE_CONFIG = "executor_shape_config"
    CONST_MEMORY_IN_GBS = "memory_in_gbs"
    CONST_OCPUS = "ocpus"
    CONST_ID = "id"

    attribute_map = {
        CONST_COMPARTMENT_ID: "compartmentId",
        CONST_CONFIG: CONST_CONFIG,
        CONST_EXECUTE: CONST_EXECUTE,
        CONST_DRIVER_SHAPE: "driverShape",
        CONST_EXECUTOR_SHAPE: "executorShape",
        CONST_METASTORE_ID: "metastoreId",
        CONST_BUCKET_URI: "logsBucketUri",
        CONST_NUM_EXECUTORS: "numExecutors",
        CONST_SPARK_VERSION: "sparkVersion",
        CONST_WAREHOUSE_BUCKET_URI: "warehouseBucketUri",
        CONST_DRIVER_SHAPE_CONFIG: "driverShapeConfig",
        CONST_EXECUTOR_SHAPE_CONFIG: "executorShapeConfig",
        CONST_MEMORY_IN_GBS: "memoryInGBs",
        CONST_OCPUS: CONST_OCPUS,
        CONST_ID: CONST_ID,
    }

    def __init__(self, spec: dict = None, **kwargs):
        defaults = self._load_default_properties()
        spec = self._standardize_spec(spec)
        kwargs = self._standardize_spec(kwargs)
        if spec is None:
            super(DataFlow, self).__init__(defaults, **kwargs)
        else:
            spec = {
                k: v
                for k, v in spec.items()
                if f"with_{camel_to_snake(k)}" in self.__dir__() and v is not None
            }
            defaults.update(spec)
            super(DataFlow, self).__init__(defaults, **kwargs)
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
                defaults["driver_shape"] = nb_config.shape
                logger.debug(f"Set driver shape to {nb_config.shape}")
                defaults["executor_shape"] = nb_config.shape
                logger.debug(f"Set executor shape to {nb_config.shape}")
                if nb_config.notebook_session_shape_config_details:
                    notebook_shape_config_details = oci_util.to_dict(
                        nb_config.notebook_session_shape_config_details
                    )
                    defaults["driver_shape_config"] = copy.deepcopy(
                        notebook_shape_config_details
                    )
                    logger.debug(
                        f"Set driver shape config to {nb_config.notebook_session_shape_config_details}"
                    )
                    defaults["executor_shape_config"] = copy.deepcopy(
                        notebook_shape_config_details
                    )
                    logger.debug(
                        f"Set executor shape config to {nb_config.notebook_session_shape_config_details}"
                    )

            except Exception as e:
                logger.warning(
                    f"Error fetching details about Notebook session: {os.environ['NB_SESSION_OCID']}. {e}"
                )

        defaults["language"] = "PYTHON"
        defaults["spark_version"] = "3.2.1"
        defaults["num_executors"] = 1
        logger.debug("Set spark version to be 3.2.1.")
        logger.debug("Set number of executors to be 1.")
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
        return self.set_spec(self.CONST_COMPARTMENT_ID, id)

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
        return self.set_spec(self.CONST_CONFIG, configs)

    def with_execute(self, exec: str) -> "DataFlow":
        """
        Set command for spark-submit.

        Parameters
        ----------
        exec: str
            str of commands

        Returns
        -------
        DataFlow
            the Data Flow instance itself
        """
        return self.set_spec(self.CONST_EXECUTE, exec)

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
        return self.set_spec(self.CONST_DRIVER_SHAPE, shape)

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
        return self.set_spec(self.CONST_EXECUTOR_SHAPE, shape)

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
        return self.set_spec(self.CONST_LANGUAGE, lang)

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

        return self.set_spec(self.CONST_METASTORE_ID, id)

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
        return self.set_spec(self.CONST_BUCKET_URI, uri)

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
        return self.set_spec(self.CONST_NUM_EXECUTORS, n)

    def with_spark_version(self, ver: str) -> "DataFlow":
        """
        Set spark version for a Data Flow job.
        Currently supported versions are 2.4.4, 3.0.2 and 3.2.1
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
        return self.set_spec(self.CONST_SPARK_VERSION, ver)

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
        return self.set_spec(self.CONST_WAREHOUSE_BUCKET_URI, uri)

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
        return self.set_spec(self.CONST_ID, id)

    def with_driver_shape_config(
        self, memory_in_gbs: float, ocpus: float, **kwargs: Dict[str, Any]
    ) -> "DataFlow":
        """
        Sets the driver shape config details of Data Flow job infrastructure.
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
        DataFlow
            the Data Flow instance itself.
        """
        return self.set_spec(
            self.CONST_DRIVER_SHAPE_CONFIG,
            {
                self.CONST_OCPUS: ocpus,
                self.CONST_MEMORY_IN_GBS: memory_in_gbs,
                **kwargs,
            },
        )

    def with_executor_shape_config(
        self, memory_in_gbs: float, ocpus: float, **kwargs: Dict[str, Any]
    ) -> "DataFlow":
        """
        Sets the executor shape config details of Data Flow job infrastructure.
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
        DataFlow
            the Data Flow instance itself.
        """
        return self.set_spec(
            self.CONST_EXECUTOR_SHAPE_CONFIG,
            {
                self.CONST_OCPUS: ocpus,
                self.CONST_MEMORY_IN_GBS: memory_in_gbs,
                **kwargs,
            },
        )

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
        # Set default display_name if not specified - randomly generated easy to remember name
        if not self.name:
            self.name = utils.get_random_name_for_resource()
        payload = copy.deepcopy(self._spec)
        runtime.convert(overwrite=kwargs.get("overwrite", False))
        if not runtime.script_uri:
            raise ValueError("script uri must be specified in runtime.")
        overwrite = kwargs.get("overwrite", False)
        if runtime.script_uri.split(":")[0] != "oci":
            if runtime.script_bucket:
                runtime.with_script_uri(
                    self._upload_file(
                        runtime.script_uri, runtime.script_bucket, overwrite
                    )
                )
            else:
                raise ValueError(
                    "script bucket must be specified if script uri given is local."
                )
        if runtime.archive_uri and runtime.archive_uri.split(":")[0] != "oci":
            if runtime.archive_bucket:
                runtime.with_archive_uri(
                    self._upload_file(
                        runtime.archive_uri, runtime.archive_bucket, overwrite
                    )
                )
            else:
                raise ValueError(
                    "archive bucket must be specified if archive uri given is local."
                )
        if runtime.conda:
            conda_type = runtime.conda.get(runtime.CONST_CONDA_TYPE)
            if conda_type == runtime.CONST_CONDA_TYPE_CUSTOM:
                conda_uri = runtime.conda.get(runtime.CONST_CONDA_URI)
            elif conda_type == runtime.CONST_CONDA_TYPE_SERVICE:
                conda_slug = runtime.conda.get(runtime.CONST_CONDA_SLUG)
                env_info = InferenceEnvInfo.from_slug(env_slug=conda_slug)
                conda_uri = env_info.inference_env_path
                # TODO: Re-visit if we can enable a global read policy
                raise NotImplementedError(
                    "Service Conda Packs not yet supported. Please use a published pack."
                )
            else:
                raise ValueError(f"Conda built type not understood: {conda_type}.")
            runtime_config = runtime.configuration or dict()
            runtime_config.update(
                {
                    "spark.archives": conda_uri.replace(" ", "%20") + CONDA_PACK_SUFFIX,
                    "dataflow.auth": "resource_principal",
                }
            )
            runtime.with_configuration(runtime_config)
        payload.update(
            {
                "display_name": self.name,
                "file_uri": runtime.script_uri,
                "freeform_tags": runtime.freeform_tags,
                "archive_uri": runtime.archive_uri,
                "configuration": runtime.configuration,
            }
        )
        if len(runtime.args) > 0:
            payload["arguments"] = runtime.args
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
    def _upload_file(local_path, bucket, overwrite=False):
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

            if not overwrite:
                remote_file_system_clz = fsspec.get_filesystem_class(
                    urllib.parse.urlparse(dst_path).scheme or "file"
                )
                remote_file_system = remote_file_system_clz(**default_signer())
                if remote_file_system.exists(dst_path):
                    raise FileExistsError(
                        f"{dst_path} exists. Please use a new file name."
                    )

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
            name of the run. If a name is not provided, a randomly generated easy to remember name
            with timestamp will be generated, like 'strange-spider-2022-08-17-23:55.02'.
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
        # Set default display_name if not specified - randomly generated easy to remember name generated
        payload["display_name"] = name if name else utils.get_random_name_for_resource()
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
