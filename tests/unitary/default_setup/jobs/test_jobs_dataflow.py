#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import os
import random
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import oci
import pytest
from oci.data_flow.models import Application, Run
from oci.response import Response
import logging

from ads.common import utils
from ads.jobs.builders.infrastructure.dataflow import (
    DataFlow,
    DataFlowApp,
    DataFlowRun,
    _Log,
    logger,
)
from ads.jobs.builders.runtimes.python_runtime import (
    DataFlowRuntime,
    DataFlowNotebookRuntime,
)

logger.setLevel(logging.DEBUG)


SAMPLE_PAYLOAD = dict(
    archive_uri="oci://test_bucket@test_namespace/test-dataflow/archive.zip",
    arguments=["test-df"],
    compartment_id="ocid1.compartment.oc1..<unique_ocid>",
    display_name="test-df",
    driver_shape="VM.Standard2.1",
    executor_shape="VM.Standard2.1",
    file_uri="oci://test_bucket@test_namespace/test-dataflow/test-dataflow.py",
    num_executors=1,
    spark_version="3.2.1",
    language="PYTHON",
    logs_bucket_uri="oci://test_bucket@test_namespace/",
)
random_seed = 42

# Current unittests running mock for "oci.config.from_file" and has specific requirement for test_config:
# "tenancy", "user", "fingerprint" must fit the ocid pattern.
# Add "# must be a real-like ocid" in the same line to pass pre-commit hook validation
test_config = {
    "tenancy": "ocid1.tenancy.oc1..xxx",  # must be a real-like ocid
    "user": "ocid1.user.oc1..xxx",  # must be a real-like ocid
    "fingerprint": "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
    "key_file": "<path>/<to>/<key_file>",
    "region": "<region>",
}


class TestDataFlowApp:
    @property
    def sample_create_application_response(self):
        payload = copy.deepcopy(SAMPLE_PAYLOAD)
        payload[
            "lifecycle_state"
        ] = oci.data_flow.models.Application.LIFECYCLE_STATE_ACTIVE
        payload["id"] = "ocid1.datasciencejob.oc1.iad.<unique_ocid>"
        return Response(
            data=Application(**payload), status=None, headers=None, request=None
        )

    @property
    def sample_create_application_response_with_default_display_name(self):
        payload = copy.deepcopy(SAMPLE_PAYLOAD)
        random.seed(random_seed)
        payload["display_name"] = utils.get_random_name_for_resource()
        return Response(
            data=Application(**payload), status=None, headers=None, request=None
        )

    @property
    def sample_delete_application_response(self):
        return Response(data=None, status=None, headers=None, request=None)

    @pytest.fixture(scope="class")
    def mock_to_dict(self):
        return MagicMock(return_value=SAMPLE_PAYLOAD)

    @pytest.fixture(scope="class")
    def mock_to_dict_with_default_display_name(self):
        payload = copy.deepcopy(SAMPLE_PAYLOAD)
        random.seed(random_seed)
        payload["display_name"] = utils.get_random_name_for_resource()
        return MagicMock(return_value=payload)

    @pytest.fixture(scope="class")
    def mock_client(self):
        mock_client = MagicMock()
        mock_client.create_application = Mock(
            return_value=self.sample_create_application_response
        )
        mock_client.delete_application = Mock(
            return_value=self.sample_delete_application_response
        )
        return mock_client

    @pytest.fixture(scope="class")
    def mock_client_with_default_display_name(self):
        mock_client = MagicMock()
        mock_client.create_application = Mock(
            return_value=self.sample_create_application_response_with_default_display_name
        )
        return mock_client

    @patch("oci.config.from_file", return_value=test_config)
    @patch("oci.signer.load_private_key_from_file")
    def test_create_delete(
        self, mock_load_key_file, mock_config_from_file, mock_to_dict, mock_client
    ):
        df = DataFlowApp(**SAMPLE_PAYLOAD)
        with patch.object(DataFlowApp, "client", mock_client):
            with patch.object(DataFlowApp, "to_dict", mock_to_dict):
                df.create()
                assert (
                    df.lifecycle_state
                    == oci.data_flow.models.Application.LIFECYCLE_STATE_ACTIVE
                )
                df.delete()
                assert (
                    df.lifecycle_state
                    == oci.data_flow.models.Application.LIFECYCLE_STATE_DELETED
                )
                assert len(df.to_yaml()) == 408

    def test_create_df_app_with_default_display_name(
        self,
        mock_to_dict_with_default_display_name,
        mock_client_with_default_display_name,
    ):
        df = DataFlowApp(**SAMPLE_PAYLOAD)
        with patch.object(DataFlowApp, "client", mock_client_with_default_display_name):
            with patch.object(
                DataFlowApp, "to_dict", mock_to_dict_with_default_display_name
            ):
                df.create()
                random.seed(random_seed)
                assert df.display_name[:-9] == utils.get_random_name_for_resource()[:-9]


class TestDataFlowRun:
    @property
    def sample_create_run_response(self):
        payload = copy.deepcopy(SAMPLE_PAYLOAD)
        payload["lifecycle_state"] = oci.data_flow.models.Run.LIFECYCLE_STATE_ACCEPTED
        payload["id"] = "ocid1.datasciencejobrun.oc1.iad.<unique_ocid>"
        return Response(data=Run(**payload), status=None, headers=None, request=None)

    @property
    def sample_create_run_response_with_default_display_name(self):
        payload = copy.deepcopy(SAMPLE_PAYLOAD)
        random.seed(random_seed)
        payload["display_name"] = utils.get_random_name_for_resource()
        return Response(data=Run(**payload), status=None, headers=None, request=None)

    @property
    def sample_get_run_response(self):
        payload = copy.deepcopy(SAMPLE_PAYLOAD)
        payload[
            "lifecycle_state"
        ] = oci.data_flow.models.Run.LIFECYCLE_STATE_IN_PROGRESS
        return Response(data=Run(**payload), status=None, headers=None, request=None)

    @property
    def sample_delete_run_response(self):
        return Response(data=None, status=None, headers=None, request=None)

    @pytest.fixture(scope="class")
    def mock_run_to_dict(self):
        payload = copy.deepcopy(SAMPLE_PAYLOAD)
        payload["application_id"] = "ocid1.dataflowapplication.oc1.iad.<unique_ocid>"
        mock = MagicMock(return_value=payload)
        return mock

    @pytest.fixture(scope="class")
    def mock_run_to_dict_with_default_display_name(self):
        payload = copy.deepcopy(SAMPLE_PAYLOAD)
        random.seed(random_seed)
        payload["display_name"] = utils.get_random_name_for_resource()
        mock = MagicMock(return_value=payload)
        return mock

    @pytest.fixture(scope="class")
    def mock_run_client(self):
        mock_client = MagicMock()
        mock_client.create_run = Mock(return_value=self.sample_create_run_response)
        mock_client.get_run = Mock(return_value=self.sample_get_run_response)
        mock_client.delete_run = Mock(return_value=self.sample_delete_run_response)
        return mock_client

    @pytest.fixture(scope="class")
    def mock_run_client_with_default_display_name(self):
        mock_client = MagicMock()
        mock_client.create_run = Mock(
            return_value=self.sample_create_run_response_with_default_display_name
        )
        return mock_client

    @patch("oci.config.from_file", return_value=test_config)
    @patch("oci.signer.load_private_key_from_file")
    def test_create_run_delete(
        self,
        mock_load_key_file,
        mock_config_from_file,
        mock_run_to_dict,
        mock_run_client,
    ):
        payload = copy.deepcopy(SAMPLE_PAYLOAD)
        payload["application_id"] = "ocid1.dataflowapplication.oc1.iad.<unique_ocid>"
        run = DataFlowRun(**payload)
        with patch.object(DataFlowRun, "client", mock_run_client):
            with patch.object(DataFlowRun, "to_dict", mock_run_to_dict):
                run.create()
                assert run.id == "ocid1.datasciencejobrun.oc1.iad.<unique_ocid>"
                assert (
                    run.lifecycle_state
                    == oci.data_flow.models.Run.LIFECYCLE_STATE_ACCEPTED
                )
                assert (
                    run.status == oci.data_flow.models.Run.LIFECYCLE_STATE_IN_PROGRESS
                )
                run.delete()
                assert (
                    run.lifecycle_state
                    == oci.data_flow.models.Run.LIFECYCLE_STATE_CANCELING
                )

    @patch("oci.config.from_file", return_value=test_config)
    @patch("oci.signer.load_private_key_from_file")
    def test_create_run_with_default_display_name(
        self,
        mock_load_key_file,
        mock_config_from_file,
        mock_run_to_dict_with_default_display_name,
        mock_run_client_with_default_display_name,
    ):
        payload = copy.deepcopy(SAMPLE_PAYLOAD)
        payload["display_name"] = None
        run = DataFlowRun(**payload)
        with patch.object(
            DataFlowRun, "client", mock_run_client_with_default_display_name
        ):
            with patch.object(
                DataFlowRun, "to_dict", mock_run_to_dict_with_default_display_name
            ):
                run.create()
                random.seed(random_seed)
                assert (
                    run.display_name[:-9] == utils.get_random_name_for_resource()[:-9]
                )

    @patch.object(DataFlowRun, "delete")
    @patch.object(DataFlowRun, "wait")
    def test_cancel(self, mock_wait, mock_delete):
        """Ensures that cancel method invoke the delete method."""
        payload = copy.deepcopy(SAMPLE_PAYLOAD)
        payload["display_name"] = None
        expected_result = DataFlowRun(**payload)

        assert expected_result == expected_result.cancel()
        mock_wait.assert_called()
        mock_delete.assert_called()


class TestDataFlowLog:
    @property
    def sample_list_run_logs_response(self):
        log = Mock()
        log.type = "STDOUT"
        log.name = "xxxx"
        log.source = "APPLICATION"
        return Response(data=[log] * 5, status=None, headers=None, request=None)

    def sample_get_run_log_response(self, text):
        mock = Mock()
        mock.text = f"{text}\n"
        return Response(data=mock, status=None, headers=None, request=None)

    @pytest.fixture(scope="class")
    def mock_client(self):
        mock_client = MagicMock()
        mock_client.list_run_logs = Mock(
            return_value=self.sample_list_run_logs_response
        )
        logs = [self.sample_get_run_log_response(f"sentence {i}") for i in range(1, 6)]
        mock_client.get_run_log = Mock(side_effect=logs)
        return mock_client

    def test_get_logs(self, mock_client, tmpdir):
        with patch.object(_Log, "client", mock_client):
            log_object = _Log("ocid", "APPLICATION", "STDOUT")
            assert log_object.head(n=1) == "sentence 1"
            assert log_object.tail(n=1) == "sentence 5"
            log_object.client.list_run_logs.assert_called_once()

            p = tmpdir.mkdir("test").join("hello.txt")
            log_object.download(p)
            assert p.read().splitlines()[0] == "sentence 1"


class TestDataFlow(TestDataFlowApp, TestDataFlowRun):
    @pytest.fixture(scope="class")
    def df(self):
        with patch.dict(
            os.environ,
            {k: v for k, v in os.environ.items() if k != "NB_SESSION_OCID"},
            clear=True,
        ):
            df = DataFlow()
            df.with_compartment_id(SAMPLE_PAYLOAD["compartment_id"]).with_configuration(
                {"spark.app.name": "My App Name", "spark.shuffle.io.maxRetries": "4"}
            ).with_driver_shape(SAMPLE_PAYLOAD["driver_shape"]).with_executor_shape(
                SAMPLE_PAYLOAD["executor_shape"]
            ).with_logs_bucket_uri(
                SAMPLE_PAYLOAD["logs_bucket_uri"]
            ).with_num_executors(
                2
            )
        return df

    def test_create_with_builder_pattern(self, mock_to_dict, mock_client, df):
        assert df.language == "PYTHON"
        assert df.spark_version == "3.2.1"
        assert df.num_executors == 2

        rt = (
            DataFlowRuntime()
            .with_script_uri(SAMPLE_PAYLOAD["file_uri"])
            .with_archive_uri(SAMPLE_PAYLOAD["archive_uri"])
            .with_custom_conda(
                "oci://allens_packs@ociodscdev/conda_environments/cpu/PySpark 3.0 and Data Flow/5.0/pyspark30_p37_cpu_v5"
            )
        )

        with patch.object(DataFlowApp, "client", mock_client):
            with patch.object(DataFlowApp, "to_dict", mock_to_dict):
                df.create(rt)
                assert (
                    df.id == df.job_id == "ocid1.datasciencejob.oc1.iad.<unique_ocid>"
                )
                df.name = "test"
                assert df.df_app.display_name == "test"

    def test_create_with_default_display_name(
        self,
        mock_to_dict_with_default_display_name,
        mock_client_with_default_display_name,
        df,
    ):
        rt = DataFlowRuntime().with_script_uri(SAMPLE_PAYLOAD["file_uri"])
        with patch.object(DataFlowApp, "client", mock_client_with_default_display_name):
            with patch.object(
                DataFlowApp, "to_dict", mock_to_dict_with_default_display_name
            ):
                df.create(rt)
                random.seed(random_seed)
                assert (
                    df.df_app.display_name[:-9]
                    == utils.get_random_name_for_resource()[:-9]
                )

    def test_run(self, df, mock_run_to_dict, mock_run_client):
        with patch.object(DataFlowRun, "client", mock_run_client):
            with patch.object(DataFlowRun, "to_dict", mock_run_to_dict):
                run = df.run()
                assert run.id == "ocid1.datasciencejobrun.oc1.iad.<unique_ocid>"
                assert (
                    run.status == oci.data_flow.models.Run.LIFECYCLE_STATE_IN_PROGRESS
                )

    def test_run_with_default_display_name(
        self,
        df,
        mock_run_to_dict_with_default_display_name,
        mock_run_client_with_default_display_name,
    ):
        with patch.object(
            DataFlowRun, "client", mock_run_client_with_default_display_name
        ):
            with patch.object(
                DataFlowRun, "to_dict", mock_run_to_dict_with_default_display_name
            ):
                run = df.run()
                random.seed(random_seed)
                assert (
                    run.display_name[:-9] == utils.get_random_name_for_resource()[:-9]
                )

    @patch.dict(
        os.environ,
        {k: v for k, v in os.environ.items() if k != "NB_SESSION_OCID"},
        clear=True,
    )
    @patch.object(DataFlowApp, "from_ocid")
    def test_create_from_id(self, mock_from_ocid):
        mock_from_ocid.return_value = Application(**SAMPLE_PAYLOAD)
        df = DataFlow.from_id("ocid1.datasciencejob.oc1.iad.<unique_ocid>")
        assert df.name == "test-df"
        assert df.driver_shape == "VM.Standard2.1"
        assert df.executor_shape == "VM.Standard2.1"
        assert (
            df.runtime.script_uri
            == "oci://test_bucket@test_namespace/test-dataflow/test-dataflow.py"
        )
        assert (
            df.runtime.archive_uri
            == "oci://test_bucket@test_namespace/test-dataflow/archive.zip"
        )

    @patch.dict(
        os.environ,
        {k: v for k, v in os.environ.items() if k != "NB_SESSION_OCID"},
        clear=True,
    )
    def test_to_and_from_dict(self, df):
        df_dict = df.to_dict()
        assert df_dict["spec"]["numExecutors"] == 2
        assert df_dict["spec"]["driverShape"] == "VM.Standard2.1"
        assert df_dict["spec"]["logsBucketUri"] == "oci://test_bucket@test_namespace/"

        df_dict["spec"].pop("language")
        df_dict["spec"].pop("numExecutors")
        df2 = DataFlow.from_dict(df_dict)
        df2_dict = df2.to_dict()
        assert df2_dict["spec"]["language"] == "PYTHON"
        assert df2_dict["spec"]["numExecutors"] == 1

        df_dict["spec"]["numExecutors"] = 2
        df_dict["spec"]["sparkVersion"] = "3.2.1"
        df3 = DataFlow.from_dict(df_dict)
        df3_dict = df3.to_dict()
        assert df3_dict["spec"]["sparkVersion"] == "3.2.1"
        assert df3_dict["spec"]["numExecutors"] == 2


class TestDataFlowNotebookRuntime:
    @pytest.mark.skipif(
        "NoDependency" in os.environ, reason="skip for dependency test: nbformat"
    )
    def test_notebook_conversion(self):
        with tempfile.TemporaryDirectory() as td:
            curr_folder = os.path.dirname(os.path.abspath(__file__))
            rt = (
                DataFlowNotebookRuntime()
                .with_notebook(
                    os.path.join(
                        curr_folder,
                        "../../..",
                        "integration",
                        "fixtures",
                        "exclude_check.ipynb",
                    )
                )
                .with_exclude_tag(["ignore", "remove"])
                .with_output(td)
            )
            rt.convert()
            assert rt.script_uri == os.path.join(td, "exclude_check.py")
            with open(os.path.join(td, "exclude_check.py")) as f:
                content = f.read()
            assert 'print("ignore")' not in content
            assert 'c = 4\n"ignore"' not in content
