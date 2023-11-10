#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from click.testing import CliRunner
from ads.opctl.cli import run, delete, watch
from shlex import split
import pytest
from tests.integration.config import secrets

TESTS_FILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "opctl_tests_files"
)
ADS_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

if "TEAMCITY_VERSION" in os.environ:
    # When running in TeamCity we specify dir, which is CHECKOUT_DIR="%teamcity.build.checkoutDir%"
    WORK_DIR = os.getenv("CHECKOUT_DIR", None)
    CONDA_PACK_FOLDER = f"{WORK_DIR}/conda"
else:
    CONDA_PACK_FOLDER = "~/conda"


def _assert_run_command(cmd_str, expected_outputs: list = None):
    runner = CliRunner()
    # Do not catch the exceptions so that we can see the traceback for debugging.
    # Test will fail with exception if there is any.
    res = runner.invoke(run, args=split(cmd_str), catch_exceptions=False)
    assert res.exit_code == 0, res.output
    # Exit code ==0 means the command finished without error.
    # We should also check the outputs to make sure the actual job run is finished as expected.
    if expected_outputs:
        # Check the outputs from print()
        actual_outputs = res.output.split("\n")
        # In job runs, outputs may show up in different orders.
        for expected_output in expected_outputs:
            assert expected_output in actual_outputs
    # For local run, watch and delete will not work. There is no run_id.


def _test_command(cmd):
    runner = CliRunner()
    res = runner.invoke(run, args=split(cmd), catch_exceptions=False)
    print(res.output)
    run_id = res.output.split("\n")[1]
    print(f"Run ID: {run_id}")
    res2 = runner.invoke(watch, args=[run_id])
    assert res2.exit_code == 0, res2.output
    runner.invoke(delete, args=[run_id])


class TestLocalRunsWithConda:
    # For tests, we can always run the command in debug mode (-d)
    # By default, pytest only print the logs if the test is failed,
    # in which case we would like to see the debug logs.
    CMD_OPTIONS = f"-d -b local --conda-pack-folder {CONDA_PACK_FOLDER} "

    def test_hello_world(self):
        test_folder = os.path.join(TESTS_FILES_DIR, "hello_world_test")
        cmd = (
            self.CMD_OPTIONS
            + f"--conda-slug dataexpl_p37_cpu_v2 -s {test_folder} "
            + "-e main.py --cmd-args '-n ADS' --env-var TEST_ENV=test"
        )

        expected_outputs = [
            "ADS 2.3.1",
            "Running user script...",
            "Hello World from ADS",
            "This is an imported module.",
            "This is an imported module from nested folder.",
        ]
        _assert_run_command(cmd, expected_outputs)

    def test_linear_reg_test(self):
        test_folder = os.path.join(TESTS_FILES_DIR, "linear_reg_test")
        cmd = (
            self.CMD_OPTIONS
            + f"--conda-slug dataexpl_p37_cpu_v2 -s {test_folder} -e main.py"
        )
        # The output numbers are non-deterministic
        expected_outputs = [
            "Coefficients: ",
        ]
        _assert_run_command(cmd, expected_outputs)

    @pytest.mark.skip(
        reason="spark do not support instance principal - this test candidate to remove"
    )
    def test_spark_run(self):
        test_folder = os.path.join(TESTS_FILES_DIR, "spark_test")
        cmd = (
            self.CMD_OPTIONS
            + f"--conda-slug pyspark30_p37_cpu_v3 -s {test_folder} -e example.py --cmd-args 'app -l 5 -v'"
        )
        expected_outputs = [f"record {i}" for i in range(5)]
        _assert_run_command(cmd, expected_outputs)

    def test_notebook_run(self):
        test_folder = os.path.join(TESTS_FILES_DIR, "notebook_test")
        cmd = (
            self.CMD_OPTIONS
            + f"--conda-slug dataexpl_p37_cpu_v2 -s {test_folder} -e exclude_check.ipynb --exclude-tag ignore -o"
        )
        expected_outputs = ["test", "hello", "remove", "hello world", "another line"]
        _assert_run_command(cmd, expected_outputs)


class TestLocalRunsWithContainer:
    # For tests, we can always run the command in debug mode (-d)
    CMD_OPTIONS = "-d -b local "

    def test_local_image(self):
        cmd = (
            self.CMD_OPTIONS
            + """-i ghcr.io/oracle/oraclelinux:7-slim -e bash -c '-c "echo $TEST"' --env-var TEST=test"""
        )
        expected_outputs = []
        _assert_run_command(cmd, expected_outputs)


class TestMLJobRunsWithConda:
    # For tests, we can always run the command in debug mode (-d)
    CMD_OPTIONS = f"-d -b job --ads-config {ADS_CONFIG_DIR} "

    @pytest.mark.skip(
        reason="config file preprocessed for run has wrong format - more information here: "
        "https://confluence.oci.oraclecorp.com/display/~lrudenka/Issues+with+OPCTL+configs. "
        "Un-skip after this story completed: https://jira.oci.oraclecorp.com/browse/ODSC-39740."
    )
    def test_hello_world(self):
        source_folder = os.path.join(TESTS_FILES_DIR, "hello_world_test")
        cmd = (
            self.CMD_OPTIONS
            + f"--conda-slug dataexpl_p37_cpu_v2 -s {source_folder} -e main.py"
        )
        _test_command(cmd)

    @pytest.mark.skip(
        reason="config file preprocessed for run has wrong format - more information here: "
        "https://confluence.oci.oraclecorp.com/display/~lrudenka/Issues+with+OPCTL+configs. "
        "Un-skip after this story completed: https://jira.oci.oraclecorp.com/browse/ODSC-39740."
    )
    def test_linear_reg_test(self):
        test_folder = os.path.join(ADS_CONFIG_DIR, "linear_reg_test")
        cmd = (
            self.CMD_OPTIONS + f"-s {test_folder} -e main.py --job-name test-linear-reg"
        )
        _test_command(cmd)


class TestMLJobRunsWithContainer:
    # For tests, we can always run the command in debug mode (-d)
    CMD_OPTIONS = f"-d -b job --ads-config {ADS_CONFIG_DIR} "

    @pytest.mark.skip(
        reason="config file preprocessed for run has wrong format - more information here: "
        "https://confluence.oci.oraclecorp.com/display/~lrudenka/Issues+with+OPCTL+configs. "
        "Un-skip after this story completed: https://jira.oci.oraclecorp.com/browse/ODSC-39740."
    )
    def test_remote_image(self):
        cmd = (
            self.CMD_OPTIONS
            + "-i ghcr.io/oracle/oraclelinux:7-slim --env-var TEST=test --job-name test-w-image "
            + """-e bash -c '-c "echo $TEST"' """
        )
        print(cmd)
        _test_command(cmd)


class TestDataFlowRun:
    # For tests, we can always run the command in debug mode (-d)
    CMD_OPTIONS = f"-d -b dataflow --ads-config {ADS_CONFIG_DIR} "

    @pytest.mark.skip(
        reason="config file preprocessed for run has wrong format - more information here: "
        "https://confluence.oci.oraclecorp.com/display/~lrudenka/Issues+with+OPCTL+configs. "
        "Un-skip after this story completed: https://jira.oci.oraclecorp.com/browse/ODSC-39740."
    )
    def test_dataflow_run(self):
        test_folder = os.path.join(TESTS_FILES_DIR, "../fixtures")
        cmd = (
            self.CMD_OPTIONS
            + f"-o -s {test_folder} -e test-dataflow.py "
            + f"--archive oci://{secrets.other.BUCKET_2}@{secrets.common.NAMESPACE}/test-df-upload/archive.zip "
            + "--cmd-args 'app -l 5 -v'"
        )
        _test_command(cmd)
