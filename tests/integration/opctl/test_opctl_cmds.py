#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from io import StringIO
import os
import pytest
import sys

from ads.opctl.cmds import run, watch, cancel, delete
from ads.jobs import DataScienceJobRun, DataFlowRun
import oci
import oci.data_science
from tests.integration.config import secrets

TESTS_FILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "opctl_tests_files"
)
ADS_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def _assert_run_command(cmd_args, expected_outputs):
    # Capture the outputs from print()
    with Capturing() as captured_outputs:
        run({}, **cmd_args)
    print(f"Captured outputs of run command: {captured_outputs}")
    if expected_outputs:
        # In job runs, outputs may show up in different orders.
        for expected_output in expected_outputs:
            assert expected_output in captured_outputs
    # For local run, watch and delete will not work. There is no run_id.


def _test_command(cmd_args):
    ids = run({}, **cmd_args)
    watch(ocid=ids["run_id"])
    jr = DataScienceJobRun.from_ocid(ids["run_id"])
    assert jr.status == oci.data_science.models.JobRun.LIFECYCLE_STATE_SUCCEEDED, ids[
        "run_id"
    ]
    cancel(ocid=ids["run_id"])
    delete(ocid=ids["job_id"])


class TestCondaRun:
    def test_hello_world_with_local_backend(self):
        cmd_args = {
            "source_folder": os.path.join(TESTS_FILES_DIR, "hello_world_test"),
            "entrypoint": "main.py",
            "backend": "local",
            "kind": "job",
            "conda_slug": "dataexpl_p37_cpu_v2",
            "cmd_args": "-n ADS",
            "gpu": True,
            "ads_config": ADS_CONFIG_DIR,
        }
        expected_outputs = [
            "ADS 2.3.1",
            "Running user script...",
            "Hello World from ADS",
            "Printing environment variables...",
            '  "conda_slug": "dataexpl_p37_cpu_v2",',
            "This is an imported module.",
            "This is an imported module from nested folder.",
        ]
        _assert_run_command(cmd_args, expected_outputs)

        cmd_args = {
            "source_folder": os.path.join(TESTS_FILES_DIR, "hello_world_test"),
            "entrypoint": "main.py",
            "backend": "local",
            "kind": "job",
            "conda_uri": "oci://service_conda_packs@ociodscdev/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/2.0/dataexpl_p37_cpu_v2",
            "oci_profile": "DEFAULT",
            "ads_config": ADS_CONFIG_DIR,
        }
        _assert_run_command(cmd_args, expected_outputs)

    @pytest.mark.skip(
        reason="config file preprocessed for run has wrong format - more information here: "
        "https://confluence.oci.oraclecorp.com/display/~lrudenka/Issues+with+OPCTL+configs. "
        "Un-skip after this story completed: https://jira.oci.oraclecorp.com/browse/ODSC-39740."
    )
    def test_hello_world_with_job_backend(self):
        cmd_args = {
            "source_folder": os.path.join(TESTS_FILES_DIR, "hello_world_test"),
            "entrypoint": "main.py",
            "backend": "job",
            "conda_uri": "oci://service_conda_packs@ociodscdev/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/2.0/dataexpl_p37_cpu_v2",
            "oci_profile": "DEFAULT",
            "ads_config": ADS_CONFIG_DIR,
        }
        _test_command(cmd_args)

    def test_linear_reg_test_with_local_backend(self):
        cmd_args = {
            "source_folder": os.path.join(TESTS_FILES_DIR, "linear_reg_test"),
            "entrypoint": "main.py",
            "conda_slug": "dataexpl_p37_cpu_v2",
            "backend": "local",
            "kind": "job",
            "ads_config": ADS_CONFIG_DIR,
        }
        # The output numbers are non-deterministic
        expected_outputs = [
            "Coefficients: ",
        ]
        _assert_run_command(cmd_args, expected_outputs)

    @pytest.mark.skip(
        reason="config file preprocessed for run has wrong format - more information here: "
        "https://confluence.oci.oraclecorp.com/display/~lrudenka/Issues+with+OPCTL+configs. "
        "Un-skip after this story completed: https://jira.oci.oraclecorp.com/browse/ODSC-39740."
    )
    def test_linear_reg_test_with_job_backend(self):
        cmd_args = {
            "source_folder": os.path.join(TESTS_FILES_DIR, "linear_reg_test"),
            "entrypoint": "main.py",
            "conda_slug": "dataexpl_p37_cpu_v2",
            "backend": "job",
            "ads_config": ADS_CONFIG_DIR,
        }
        _test_command(cmd_args)

    def test_spark_run_with_local_backend(self):
        cmd_args = {
            "source_folder": os.path.join(TESTS_FILES_DIR, "spark_test"),
            "entrypoint": "example.py",
            "conda_slug": "pyspark30_p37_cpu_v3",
            "backend": "local",
            "kind": "job",
            "ads_config": ADS_CONFIG_DIR,
            "cmd_args": "app -l 10 -v",
        }
        expected_outputs = [f"record {i}" for i in range(5)]
        _assert_run_command(cmd_args, expected_outputs)

    @pytest.mark.skip(
        reason="config file preprocessed for run has wrong format - more information here: "
        "https://confluence.oci.oraclecorp.com/display/~lrudenka/Issues+with+OPCTL+configs. "
        "Un-skip after this story completed: https://jira.oci.oraclecorp.com/browse/ODSC-39740."
    )
    def test_notebook_run_with_job_backend(self):
        cmd_args = {
            "source_folder": os.path.join(TESTS_FILES_DIR, "linear_reg_test"),
            "entrypoint": "exclude_check.ipynb",
            "conda_slug": "dataexpl_p37_cpu_v2",
            "backend": "job",
            "ads_config": ADS_CONFIG_DIR,
            "exclude_tag": ["ignore", "remove"],
            "overwrite": True,
        }
        _test_command(cmd_args)


class TestContainerRun:
    @pytest.mark.skip(
        reason="config file preprocessed for run has wrong format - more information here: "
        "https://confluence.oci.oraclecorp.com/display/~lrudenka/Issues+with+OPCTL+configs. "
        "Un-skip after this story completed: https://jira.oci.oraclecorp.com/browse/ODSC-39740."
    )
    def test_user_image(self):
        cmd_args = {
            "image": "ghcr.io/oracle/oraclelinux:7-slim",
            "entrypoint": "ls",
            "command": "-a -t -l",
            "backend": "job",
            "job_name": "test-byod",
            "env_var": ["TEST=test"],
            "ads_config": ADS_CONFIG_DIR,
        }
        _test_command(cmd_args)


class TestDataFlowRun:
    @pytest.mark.skip(
        reason="config file preprocessed for run has wrong format - more information here: "
        "https://confluence.oci.oraclecorp.com/display/~lrudenka/Issues+with+OPCTL+configs. "
        "Un-skip after this story completed: https://jira.oci.oraclecorp.com/browse/ODSC-39740."
    )
    def test_dataflow_run(self):
        # The archive used here is prepared by following the instruction in https://docs.oracle.com/en-us/iaas/data-flow/using/third-party-libraries.htm.
        # The python version in archive should match with the DF runtime.
        # Currently, this archive is for python 3.8.x. If this test fail in the future, check the python version for DF runtime first.
        cmd_args = {
            "source_folder": os.path.join(TESTS_FILES_DIR, "../fixtures"),
            "entrypoint": "test-dataflow.py",
            "archive": f"oci://ADS_INT_TEST@{secrets.common.NAMESPACE}/dataflow/archives/archive.zip",
            "backend": "dataflow",
            "ads_config": ADS_CONFIG_DIR,
            "cmd_args": "app -v",
        }
        with pytest.raises(FileExistsError):
            _test_command(cmd_args)
        cmd_args["overwrite"] = True
        _test_command(cmd_args)

    @pytest.mark.skip(
        reason="config file preprocessed for run has wrong format - more information here: "
        "https://confluence.oci.oraclecorp.com/display/~lrudenka/Issues+with+OPCTL+configs. "
        "Un-skip after this story completed: https://jira.oci.oraclecorp.com/browse/ODSC-39740."
    )
    def test_notebook_run(self):
        cmd_args = {
            "source_folder": os.path.join(TESTS_FILES_DIR, "../fixtures"),
            "entrypoint": "exclude_check.ipynb",
            "backend": "dataflow",
            "ads_config": ADS_CONFIG_DIR,
            "exclude_tag": ["ignore", "remove"],
            "overwrite": True,
        }
        _test_command(cmd_args)
