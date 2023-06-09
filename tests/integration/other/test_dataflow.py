#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import os
from uuid import uuid4

import pytest
import yaml
import oci
import logging
import tempfile

from ads.jobs.ads_job import Job
from ads.jobs.builders.infrastructure.dataflow import (
    DataFlowApp,
    DataFlowRun,
    DataFlow,
    logger,
)
from ads.jobs.builders.runtimes.python_runtime import (
    DataFlowRuntime,
    DataFlowNotebookRuntime,
)
from ads.common import auth as authutil
from ads.common import oci_client as oc
from tests.integration.config import secrets
import hashlib

logger.setLevel(logging.DEBUG)

SAMPLE_PAYLOAD = dict(
    arguments=[f"test-df-{str(uuid4())}"],
    compartment_id=secrets.common.COMPARTMENT_ID,
    display_name="test-df",
    driver_shape="VM.Standard2.1",
    executor_shape="VM.Standard2.1",
    file_uri=f"oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-dataflow/test-dataflow.py",
    num_executors=1,
    spark_version="3.2.1",
    language="PYTHON",
    logs_bucket_uri=f"oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/",
)

SAMPLE_CONDA_PAYLOAD = dict(
    type="published",
    uri=f"oci://{secrets.other.BUCKET_1}@{secrets.common.NAMESPACE}/conda_environments/cpu/PySpark 3.0 and Data Flow/v2/pyspark3_0anddataflowvv2",
)

SAMPLE_CONFIGURATION_PAYLOAD = {
    "spark.archives": SAMPLE_CONDA_PAYLOAD["uri"] + "#conda",  # .replace(" ", "%20")
    "dataflow.auth": "resource_principal",
}

SAMPLE_SHAPE_PAYLOAD = dict(
    driver_shape="VM.Standard3.Flex",
    driver_shape_config={"ocpus": 2, "memory_in_gbs": 16},
    executor_shape="VM.Standard3.Flex",
    executor_shape_config={"ocpus": 4, "memory_in_gbs": 32},
)


class TestDataFlowRun:
    @pytest.fixture(scope="class")
    def df(self):
        df = DataFlowApp(**SAMPLE_PAYLOAD).create()
        assert (
            df.lifecycle_state
            == oci.data_flow.models.Application.LIFECYCLE_STATE_ACTIVE
        )
        yield df
        df.delete()
        assert (
            df.lifecycle_state
            == oci.data_flow.models.Application.LIFECYCLE_STATE_DELETED
        )

    def test_create_delete(self, df):
        payload = copy.deepcopy(SAMPLE_PAYLOAD)
        payload.pop("spark_version")
        payload["application_id"] = df.id

        payload["arguments"] = ["test-df", "-v"]
        payload["configuration"] = SAMPLE_CONFIGURATION_PAYLOAD
        dfr = DataFlowRun(**payload).create()
        assert dfr.status == oci.data_flow.models.Run.LIFECYCLE_STATE_ACCEPTED
        dfr.wait()
        assert dfr.logs.application.stdout.head(n=1) == "record 0"
        assert (
            dfr.logs.application.stdout.tail(n=1)
            == '{"city":"Berlin","zipcode":"10405","lat_long":"52.52907092467378,13.412843393984936"}'
        )
        assert dfr.status == oci.data_flow.models.Run.LIFECYCLE_STATE_SUCCEEDED

        payload["arguments"] = ["test-df", "-v", "-l", "20"]
        dfr = DataFlowRun(**payload).create()
        dfr.delete()
        assert dfr.status == oci.data_flow.models.Run.LIFECYCLE_STATE_CANCELING
        dfr.watch()
        assert dfr.status == oci.data_flow.models.Run.LIFECYCLE_STATE_CANCELED

        payload["arguments"] = ["-a"]
        dfr = DataFlowRun(**payload).create()
        dfr.wait()
        assert dfr.status == oci.data_flow.models.Run.LIFECYCLE_STATE_FAILED

        dfr2 = DataFlowRun(**yaml.safe_load(dfr.to_yaml()))
        assert dfr2.id == dfr.id


class TestDataFlow:
    @pytest.fixture(scope="class")
    def df(self):
        df = (
            DataFlow()
            .with_compartment_id(SAMPLE_PAYLOAD["compartment_id"])
            .with_driver_shape(SAMPLE_PAYLOAD["driver_shape"])
            .with_executor_shape(SAMPLE_PAYLOAD["executor_shape"])
            .with_logs_bucket_uri(SAMPLE_PAYLOAD["logs_bucket_uri"])
        )
        rt = (
            DataFlowRuntime()
            .with_script_uri(SAMPLE_PAYLOAD["file_uri"])
            .with_custom_conda(SAMPLE_CONDA_PAYLOAD["uri"])
        )
        yield df.create(rt)
        df.delete()

    def test_create_with_builder_pattern(self, df):
        assert df.compartment_id == SAMPLE_PAYLOAD["compartment_id"]
        assert df.driver_shape == SAMPLE_PAYLOAD["driver_shape"]
        assert df.executor_shape == SAMPLE_PAYLOAD["executor_shape"]
        assert df.logs_bucket_uri == SAMPLE_PAYLOAD["logs_bucket_uri"]
        assert df.num_executors == 1

    def test_runs(self, df):
        dfr1 = df.run(name="run-1")
        assert dfr1.display_name == "run-1"
        dfr1.watch()
        assert dfr1.status == oci.data_flow.models.Run.LIFECYCLE_STATE_FAILED

        dfr2 = df.run(args=["run-2"], wait=True)
        assert dfr2.status == oci.data_flow.models.Run.LIFECYCLE_STATE_SUCCEEDED

        dfr3 = df.run(
            args=["run-3"],
            freeform_tags={"tag": "test-tag"},
        )
        assert dfr3.freeform_tags["tag"] == "test-tag"

        dfrs = df.run_list()
        ids = [r.id for r in dfrs]
        assert dfr1.id in ids
        assert dfr2.id in ids
        assert dfr3.id in ids

    def test_create_from_id(self, df):
        df2 = DataFlow.from_id(df.id)
        dfr = df2.run(args=["run-4", "-v", "-l", "5"], wait=True)
        assert dfr.status == oci.data_flow.models.Run.LIFECYCLE_STATE_SUCCEEDED

    def test_create_with_jobs_api(self, df):
        job = (
            Job()
            .with_name("test-df-job")
            .with_infrastructure(df)
            .with_runtime(
                DataFlowRuntime()
                .with_script_uri(SAMPLE_PAYLOAD["file_uri"])
                .with_custom_conda(SAMPLE_CONDA_PAYLOAD["uri"])
            )
        )
        jr = job.create().run(args=["run-test", "-v", "-l", "5"], wait=True)
        assert job.infrastructure.df_app.display_name == "test-df-job"
        assert jr.status == oci.data_flow.models.Run.LIFECYCLE_STATE_SUCCEEDED

        job3 = Job.from_dataflow_job(job.id)
        assert job3.infrastructure.configuration == job.runtime.configuration

    def test_uploading_from_local(self):
        data_flow = (
            DataFlow()
            .with_compartment_id(SAMPLE_PAYLOAD["compartment_id"])
            .with_driver_shape(SAMPLE_PAYLOAD["driver_shape"])
            .with_executor_shape(SAMPLE_PAYLOAD["executor_shape"])
            .with_logs_bucket_uri(SAMPLE_PAYLOAD["logs_bucket_uri"])
        )
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        rt = (
            DataFlowRuntime()
            .with_script_uri(os.path.join(curr_dir, "../fixtures", "test-dataflow.py"))
            .with_custom_conda(SAMPLE_CONDA_PAYLOAD["uri"])
            .with_overwrite(False)
        )
        with pytest.raises(ValueError):
            data_flow.create(rt, overwrite=True)
        rt.with_script_bucket(secrets.other.BUCKET_2).with_archive_bucket(
            f"oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-df-upload"
        ).with_archive_uri(
            f"oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-df-upload/archive.zip"
        )
        with pytest.raises(FileExistsError):
            data_flow.create(rt, overwrite=False)
        job = Job(name="test-job-from-local")
        job.with_infrastructure(data_flow).with_runtime(rt)
        dfr = job.create(overwrite=True).run(args=["run-test-from-local"])
        assert (
            job.to_dict()["spec"]["runtime"]["spec"]["scriptPathURI"]
            == f"oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-dataflow.py"
        )
        assert (
            job.to_dict()["spec"]["runtime"]["spec"]["archiveUri"]
            == f"oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-df-upload/archive.zip"
        )
        assert job.to_dict()["spec"]["runtime"]["spec"]["overwrite"] == False

        dfr.wait()
        dfr.status == oci.data_flow.models.Run.LIFECYCLE_STATE_SUCCEEDED

        rt.with_script_uri(
            os.path.join(curr_dir, "../fixtures", "test-dataflow.py")
        ).with_script_bucket(
            f"oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-df-upload"
        ).with_archive_bucket(
            secrets.other.BUCKET_2
        )

        job.with_runtime(rt)
        job.create(overwrite=True)
        assert (
            job.to_dict()["spec"]["runtime"]["spec"]["scriptPathURI"]
            == f"oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-df-upload/test-dataflow.py"
        )
        assert (
            job.to_dict()["spec"]["runtime"]["spec"]["archiveUri"]
            == f"oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-df-upload/archive.zip"
        )
        assert job.to_dict()["spec"]["runtime"]["spec"]["overwrite"] == False
        job.delete()

    def test_uploading_from_local_large_file(self, tmpdir):
        os_client = oc.OCIClientFactory(**authutil.default_signer()).object_storage

        response = os_client.get_object(
            secrets.common.NAMESPACE,
            secrets.other.BUCKET_3,
            "dataflow/archive-pandas/archive.zip",
        )
        with open(os.path.join(tmpdir, "archive.zip"), "wb") as archive:
            archive.write(response.data.content)

        dst_path = DataFlow._upload_file(
            os.path.join(tmpdir, "archive.zip"),
            f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/dataflow/upload/",
            overwrite=True,
        )

        response = os_client.get_object(
            secrets.common.NAMESPACE,
            secrets.other.BUCKET_3,
            "dataflow/upload/archive.zip",
        )
        with open(os.path.join(tmpdir, "archive_dowloaded.zip"), "wb") as archive:
            archive.write(response.data.content)

        with open(os.path.join(tmpdir, "archive_dowloaded.zip"), "rb") as fdownloaded:
            with open(os.path.join(tmpdir, "archive.zip"), "rb") as foriginal:
                assert (
                    hashlib.md5(fdownloaded.read()).hexdigest()
                    == hashlib.md5(foriginal.read()).hexdigest()
                )

    def test_notebook_run(self):
        data_flow = (
            DataFlow()
            .with_compartment_id(SAMPLE_PAYLOAD["compartment_id"])
            .with_driver_shape(SAMPLE_PAYLOAD["driver_shape"])
            .with_executor_shape(SAMPLE_PAYLOAD["executor_shape"])
            .with_logs_bucket_uri(SAMPLE_PAYLOAD["logs_bucket_uri"])
        )
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as td:
            rt = (
                DataFlowNotebookRuntime()
                .with_notebook(
                    os.path.join(curr_dir, "../fixtures", "exclude_check.ipynb")
                )
                .with_output(td)
                .with_exclude_tag(["ignore", "remove"])
                .with_script_bucket(
                    f"oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-df-upload"
                )
                .with_custom_conda(SAMPLE_CONDA_PAYLOAD["uri"])
                .with_overwrite(True)
            )
            job = Job(
                name="test_notebook_run_1", infrastructure=data_flow, runtime=rt
            ).create()
            df_run = job.run(wait=True)
            assert df_run.status == oci.data_flow.models.Run.LIFECYCLE_STATE_SUCCEEDED
            job.delete()

        rt = (
            DataFlowNotebookRuntime()
            .with_notebook(os.path.join(curr_dir, "../fixtures", "exclude_check.ipynb"))
            .with_output(
                f"oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-df-upload/notebook"
            )
            .with_exclude_tag(["ignore", "remove"])
            .with_custom_conda(SAMPLE_CONDA_PAYLOAD["uri"])
            .with_overwrite(True)
        )
        job = Job(
            name="test_notebook_run_2", infrastructure=data_flow, runtime=rt
        ).create()
        df_run = job.run(wait=True)
        assert df_run.status == oci.data_flow.models.Run.LIFECYCLE_STATE_SUCCEEDED
        job.delete()

        rt = (
            DataFlowNotebookRuntime()
            .with_notebook(
                f"oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/exclude_check.ipynb"
            )
            .with_exclude_tag(["ignore", "remove"])
            .with_custom_conda(SAMPLE_CONDA_PAYLOAD["uri"])
        )
        job = Job(
            name="test_notebook_run_3", infrastructure=data_flow, runtime=rt
        ).create(overwrite=True)
        df_run = job.run(wait=True)
        assert df_run.status == oci.data_flow.models.Run.LIFECYCLE_STATE_SUCCEEDED
        job.delete()


class TestDataFlowFlexShape:
    @pytest.fixture(scope="class")
    def df(self):
        df = (
            DataFlow()
            .with_compartment_id(SAMPLE_PAYLOAD["compartment_id"])
            .with_driver_shape(SAMPLE_SHAPE_PAYLOAD["driver_shape"])
            .with_driver_shape_config(
                memory_in_gbs=SAMPLE_SHAPE_PAYLOAD["driver_shape_config"][
                    "memory_in_gbs"
                ],
                ocpus=SAMPLE_SHAPE_PAYLOAD["driver_shape_config"]["ocpus"],
            )
            .with_executor_shape(SAMPLE_SHAPE_PAYLOAD["executor_shape"])
            .with_executor_shape_config(
                memory_in_gbs=SAMPLE_SHAPE_PAYLOAD["executor_shape_config"][
                    "memory_in_gbs"
                ],
                ocpus=SAMPLE_SHAPE_PAYLOAD["executor_shape_config"]["ocpus"],
            )
            .with_logs_bucket_uri(SAMPLE_PAYLOAD["logs_bucket_uri"])
        )
        rt = (
            DataFlowRuntime()
            .with_script_uri(SAMPLE_PAYLOAD["file_uri"])
            .with_custom_conda(SAMPLE_CONDA_PAYLOAD["uri"])
        )
        yield df.create(rt)
        df.delete()

    def test_create_with_builder_pattern(self, df):
        assert df.compartment_id == SAMPLE_PAYLOAD["compartment_id"]
        assert df.driver_shape == SAMPLE_SHAPE_PAYLOAD["driver_shape"]
        assert df.driver_shape_config == SAMPLE_SHAPE_PAYLOAD["driver_shape_config"]
        assert df.executor_shape == SAMPLE_SHAPE_PAYLOAD["executor_shape"]
        assert df.executor_shape_config == SAMPLE_SHAPE_PAYLOAD["executor_shape_config"]
        assert df.logs_bucket_uri == SAMPLE_PAYLOAD["logs_bucket_uri"]
        assert df.num_executors == 1
