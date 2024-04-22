#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import time
import fsspec
import oci
import pytest
import random

from tests.integration.config import secrets
from tests.integration.jobs.test_dsc_job import DSCJobTestCaseWithCleanUp
from ads.common.auth import default_signer
from ads.jobs import (
    Job,
    DataScienceJob,
    ScriptRuntime,
    GitPythonRuntime,
    PythonRuntime,
    NotebookRuntime,
)

SKIP_TEST_FLAG = "TEST_JOB_RUN" not in os.environ
SKIP_TEST_REASON = "Set environment variable TEST_JOB_RUN to enable job run tests."


class DSCJobRunTestCase(DSCJobTestCaseWithCleanUp):
    """Contains tests for running OCI Data Science Jobs

    The tests are marked skip for pytest unless the environment variable TEST_JOB_RUN is set.
    Running all the tests here sequentially may take about 1.5 hours.
    It is recommended to run the tests in parallel using the pytest-xdist:
    https://pypi.org/project/pytest-xdist/

    For example, the following command will run 5 tests in parallel:
    TEST_JOB_RUN=1 pytest -n 5 tests/integration/test_jobs_runs.py

    """

    DIR_SOURCE_PATH = os.path.join(os.path.dirname(__file__), "../fixtures/job_archive")
    OCI_SOURCE_PATH = f"oci://{secrets.jobs.BUCKET_B}@{secrets.common.NAMESPACE}/job_artifact/job_archive.zip"
    SH_SOURCE_PATH = os.path.join(
        os.path.dirname(__file__), "../fixtures/job_archive", "script.sh"
    )
    ZIP_JOB_ENTRYPOINT = "job_archive/main.py"

    TEST_OUTPUT_DIR = "output"
    TEST_OUTPUT_URI = (
        f"oci://{secrets.jobs.BUCKET_B}@{secrets.common.NAMESPACE}/ads_int_test"
    )
    SHAPE_NAME = "VM.Standard2.1"
    CUSTOM_CONDA = f"oci://{secrets.jobs.BUCKET_B}@{secrets.common.NAMESPACE}/conda_environments/cpu/flaml/1.0/automl_flaml"

    TEST_LOGS_FUNCTION = [
        "This is a function in a package.",
        "This is a function in a module.",
        "This is the entrypoint inside a package.",
    ]

    TEST_LOGS_SCRIPT = [
        "This is the main script.",
        "This is a function in a module.",
        "This is a function in a package.",
    ]

    # With shortage of ip addresses in self.SUBNET_ID,
    # added pool of subnets with extra 8+8 ip addresses to run tests in parallel:
    SUBNET_POOL = {
        secrets.jobs.SUBNET_ID_1: 8,  # max 8 ip addresses available in SUBNET_ID_1
        secrets.jobs.SUBNET_ID_2: 8,
        secrets.jobs.SUBNET_ID: 32,
    }

    def setUp(self) -> None:
        self.maxDiff = None
        return super().setUp()

    @property
    def job_run_test_infra(self):
        """Data Science Job infrastructure with logging and managed egress for testing job runs"""

        # Pick subnet one of SUBNET_ID_1, SUBNET_ID_2, SUBNET_ID from self.SUBNET_POOL with available ip addresses.
        # Wait for 4 minutes if no ip addresses in any of 3 subnets, do 5 retries.
        max_retry_count = 5
        subnet_id = None
        interval = 4 * 60
        core_client = oci.core.VirtualNetworkClient(**default_signer())
        while max_retry_count > 0:
            for subnet, ips_limit in random.sample(list(self.SUBNET_POOL.items()), 2):
                allocated_ips = core_client.list_private_ips(subnet_id=subnet).data
                # Leave 4 extra ip address for later use by jobrun. Leave more extra ips in case tests will fail with
                # "All the available IP addresses in the subnet have been allocated."
                if len(allocated_ips) < ips_limit - 4:
                    subnet_id = subnet
                    break
            if subnet_id:
                break
            else:
                max_retry_count -= 1
                time.sleep(interval)
        # After all retries and no subnet_id with available ip addresses - using SUBNET_ID_1, subnet_id can't be None
        if not subnet_id:
            subnet_id = secrets.jobs.SUBNET_ID_1

        return DataScienceJob(
            compartment_id=self.COMPARTMENT_ID,
            project_id=self.PROJECT_ID,
            shape_name=self.SHAPE_NAME,
            block_storage_size=50,
            log_id=self.LOG_ID,
            job_infrastructure_type="STANDALONE",
            subnet_id=subnet_id,
        )

    @staticmethod
    def list_objects(uri: str) -> list:
        """Lists objects on OCI object storage."""
        oci_os = fsspec.filesystem("oci", **default_signer())
        if uri.startswith("oci://"):
            uri = uri[len("oci://") :]
        items = oci_os.ls(uri, detail=False, refresh=True)
        return [item[len(uri) :].lstrip("/") for item in items]

    @staticmethod
    def remove_objects(uri: str):
        """Removes objects from OCI object storage."""
        oci_os = fsspec.filesystem("oci", **default_signer())
        try:
            oci_os.rm(uri, recursive=True)
        except FileNotFoundError:
            pass

    def assert_job_run(self, run, expected_logs):
        """Asserts the job run status and logs."""
        actual_logs = [log["message"] for log in run.logs()] if run.log_id else []
        print(actual_logs)
        self.assertEqual(
            run.status, "SUCCEEDED", "Job Failed - Logs:\n" + "\n".join(actual_logs)
        )

        log_comparison = (
            "== Expected Logs ==:\n"
            + "\n".join(expected_logs)
            + "\n== Actual Logs ==:\n"
            + "\n".join(actual_logs)
        )

        for log in expected_logs:
            try:
                actual_logs.remove(log)
            except ValueError:
                self.fail(
                    "Expected logs not found in job run: " + log + "\n" + log_comparison
                )

    def create_and_assert_job_run(self, runtime, expected_logs=None, infra=None):
        """Runs a job with specific runtime and Data Science Job infrastructure,
        and checks the outputs in the logs.

        If infra is not specified, the default infra will be used.
        """
        if not expected_logs:
            expected_logs = []

        job = (
            Job()
            .with_infrastructure(infra or self.job_run_test_infra)
            .with_runtime(runtime)
            .create()
        )
        print("\n" + "=" * 20)
        print(job)
        run = job.run()
        print(run)
        run.watch()
        time.sleep(120)
        self.assert_job_run(run, expected_logs)


class ScriptRuntimeJobRunTest(DSCJobRunTestCase):
    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_python_using_script_runtime(self):
        """Tests running Python script using ScriptRuntime.
        This test also checks the passing of cmd arguments and environment variables.
        """
        runtime = (
            ScriptRuntime()
            .with_script(self.SCRIPT_URI)
            .with_argument("pos_arg1", "pos_arg2", key1="val1", key2="val2")
            .with_environment_variable(env1="env_v1", env2="env_v2")
        )
        self.create_and_assert_job_run(
            runtime,
            [
                "Positional Arguments: ['pos_arg1', 'pos_arg2']",
                "Keyword Arguments: {'--key1': 'val1', '--key2': 'val2'}",
                "Starting the job",
                "Sleeping for 60 seconds...",
                "After 60 seconds...",
                "Finishing the job",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_python_using_script_runtime_and_entrypoint(self):
        """Tests running a Python script using ScriptRuntime and specified the entrypoint
        The source code is a local directory containing python script, user defined package and module.
        The python script imports the user defined package and module.
        """
        entrypoint = "job_archive/main.py"
        runtime = (
            ScriptRuntime()
            .with_source(self.DIR_SOURCE_PATH, entrypoint=entrypoint)
            .with_service_conda("dbexp_p38_cpu_v1")
        )
        self.create_and_assert_job_run(
            runtime,
            self.TEST_LOGS_SCRIPT,
        )

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_shell_script(self):
        """Tests running a shell script using ScriptRuntime"""
        runtime = (
            ScriptRuntime()
            .with_source(self.SH_SOURCE_PATH)
            .with_service_conda("dbexp_p38_cpu_v1")
        )
        self.create_and_assert_job_run(
            runtime,
            ["Hello World"],
        )

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_python_in_zip_using_script_runtime(self):
        """Tests running Python script in a zip file stored in OCI object storage
        using ScriptRuntime.
        """
        runtime = (
            ScriptRuntime()
            .with_source(self.OCI_SOURCE_PATH, entrypoint=self.ZIP_JOB_ENTRYPOINT)
            .with_service_conda("dbexp_p38_cpu_v1")
        )
        self.create_and_assert_job_run(runtime, self.TEST_LOGS_SCRIPT)

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_script_with_many_logs(self):
        """Tests running a Python script generating many logs using ScriptRuntime."""
        runtime = ScriptRuntime().with_source(
            os.path.join(
                os.path.dirname(__file__), "../fixtures/script_with_many_logs.py"
            )
        )
        logs = [f"LOG: {i}" for i in range(2000)]
        self.create_and_assert_job_run(runtime, logs)


class GitRuntimeJobRunTest(DSCJobRunTestCase):
    PROXY_ENVS = dict(
        http_proxy=secrets.jobs.HTTP_PROXY,
        https_proxy=secrets.jobs.HTTPS_PROXY,
        ssh_proxy=secrets.jobs.SSH_PROXY,
        no_proxy=secrets.jobs.NO_PROXY,
    )

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_git_with_entry_function_and_arguments(self):
        """Tests running a Python function from Git repo and passing in the arguments."""
        envs = dict(OCI_LOG_LEVEL="DEBUG")
        envs.update(self.PROXY_ENVS)
        runtime = (
            GitPythonRuntime()
            .with_source(secrets.jobs.GITHUB_SOURCE)
            .with_entrypoint(path="src/main.py", func="entry_function")
            .with_python_path("src")
            .with_service_conda("dbexp_p38_cpu_v1")
            .with_argument(
                # Argument as string
                "arg1",
                # Keyword argument as a string
                key='{"key": ["val1", "val2"]}',
            )
            .with_environment_variable(**envs)
        )
        self.create_and_assert_job_run(
            runtime,
            [
                "This is the entry function.",
                "This is a function in a module.",
                'key={"key": ["val1", "val2"]} (<class \'str\'>)',
                "arg1 (<class 'str'>)",
                "2.6.8",
                # The following log will only show up if OCI_LOG_LEVEL is set to DEBUG
                "Job completed.",
                "Saving metadata to job run...",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_git_with_notebook_entrypoint_and_output_uri(self):
        """Tests running a notebook from Git repo and saving the outputs to object storage"""
        output_uri = os.path.join(self.TEST_OUTPUT_URI, "git_notebook")
        self.remove_objects(output_uri)
        envs = dict(OCI_LOG_LEVEL="DEBUG")
        envs.update(self.PROXY_ENVS)
        runtime = (
            GitPythonRuntime(skip_metadata_update=True)
            .with_source(secrets.jobs.GITHUB_SOURCE)
            .with_entrypoint(path="src/test_notebook.ipynb")
            .with_output("src", output_uri)
            .with_service_conda("dbexp_p38_cpu_v1")
            .with_environment_variable(**envs)
        )
        self.create_and_assert_job_run(
            runtime,
            [
                "2.6.8",
                "This is the test notebook.",
                "This is a function in a module.",
                # The following log will only show up if OCI_LOG_LEVEL is set to DEBUG
                "Job completed.",
            ],
        )
        objects = self.list_objects(output_uri)
        self.remove_objects(output_uri)
        self.assertIn("test_notebook.ipynb", objects)
        self.assertIn("output.txt", objects)
        self.assertIn("my_package", objects)
        self.assertIn("script.sh", objects)

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_git_with_shell_script_entrypoint(self):
        """Tests running a notebook from Git repo and saving the outputs to object storage"""
        output_uri = os.path.join(self.TEST_OUTPUT_URI, "git_notebook")
        self.remove_objects(output_uri)
        envs = dict(OCI_LOG_LEVEL="DEBUG")
        envs.update(self.PROXY_ENVS)
        runtime = (
            GitPythonRuntime(skip_metadata_update=True)
            .with_source(secrets.jobs.GITHUB_SOURCE)
            .with_entrypoint(path="src/conda_list.sh")
            .with_service_conda("dbexp_p38_cpu_v1")
            .with_argument("0.5", "+", 0.2, equals="0.7")
            .with_environment_variable(**envs)
        )
        self.create_and_assert_job_run(
            runtime,
            [
                "0.5 + 0.2 --equals 0.7",
                "# packages in environment at /home/datascience/conda/dbexp_p38_cpu_v1:",
                "Job completed.",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_git_with_http_proxy_and_entry_function(self):
        """Tests running a Python function from Git repo using HTTP proxy"""
        runtime = (
            GitPythonRuntime(skip_metadata_update=True)
            .with_source(secrets.jobs.GITHUB_SOURCE)
            .with_entrypoint(path="src/main.py", func="entry_function")
            .with_python_path("src")
            .with_service_conda("dbexp_p38_cpu_v1")
            .with_environment_variable(**self.PROXY_ENVS)
        )
        infra = self.job_run_test_infra.with_subnet_id(
            self.SUBNET_ID
        ).with_job_infrastructure_type("STANDALONE")
        self.create_and_assert_job_run(
            runtime,
            [
                "2.6.8",
                "This is the entry function.",
                "This is a function in a module.",
                "This is a function in a package.",
            ],
            infra=infra,
        )

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_git_with_ssh_key(self):
        envs = dict(OCI_LOG_LEVEL="DEBUG")
        envs.update(self.PROXY_ENVS)
        runtime = (
            GitPythonRuntime(skip_metadata_update=True)
            .with_source(
                url=secrets.jobs.GITHUB_SOURCE_URL,
                secret_ocid=secrets.jobs.SECRET_ID,
            )
            .with_entrypoint(path="src/main.py")
            .with_python_path("src")
            .with_custom_conda(self.CUSTOM_CONDA)
            .with_environment_variable(**envs)
        )
        self.create_and_assert_job_run(
            runtime,
            self.TEST_LOGS_SCRIPT,
        )

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_git_with_ssh_key_and_proxy(self):
        envs = dict(OCI_LOG_LEVEL="DEBUG")
        envs.update(self.PROXY_ENVS)
        runtime = (
            GitPythonRuntime(skip_metadata_update=True)
            .with_source(
                url=secrets.jobs.GITHUB_SOURCE_URL,
                branch="main",
                secret_ocid=secrets.jobs.SECRET_ID,
            )
            .with_entrypoint(path="src/main.py")
            .with_python_path("src")
            .with_custom_conda(self.CUSTOM_CONDA)
            .with_environment_variable(**envs)
        )
        infra = self.job_run_test_infra.with_subnet_id(
            self.SUBNET_ID
        ).with_job_infrastructure_type("STANDALONE")
        self.create_and_assert_job_run(
            runtime,
            self.TEST_LOGS_SCRIPT,
            infra=infra,
        )


class PythonRuntimeJobRunTest(DSCJobRunTestCase):
    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_python_with_dir_and_set_python_path(self):
        """Tests running a python job with a directory as artifact and additional python path."""
        entrypoint = "job_archive/my_package/entrypoint_ads.py"
        runtime = (
            PythonRuntime()
            .with_source(self.DIR_SOURCE_PATH, entrypoint=entrypoint)
            .with_python_path("job_archive")
            .with_service_conda("dbexp_p38_cpu_v1")
        )
        self.create_and_assert_job_run(
            runtime,
            [
                "2.6.8",
                "/home/datascience/decompressed_artifact/code",
            ]
            + self.TEST_LOGS_FUNCTION,
        )

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_python_job_with_dir_and_working_dir(self):
        """Tests running a python job with a directory as artifact and configuring working dir."""
        entrypoint = "my_package/entrypoint_ads.py"
        output_uri = os.path.join(self.TEST_OUTPUT_URI, "python_dir")
        self.remove_objects(output_uri)
        runtime = (
            PythonRuntime()
            # Change the working dir so that the entrypoint can be simplified.
            # Working directory is also added to PYTHONPATH by default.
            .with_working_dir("job_archive")
            .with_source(self.DIR_SOURCE_PATH, entrypoint=entrypoint)
            .with_service_conda("dbexp_p38_cpu_v1")
            .with_output(
                output_dir=self.TEST_OUTPUT_DIR,
                output_uri=output_uri,
            )
        )
        self.create_and_assert_job_run(
            runtime,
            [
                "2.6.8",
                "/home/datascience/decompressed_artifact/code/job_archive",
            ]
            + self.TEST_LOGS_FUNCTION,
        )
        objects = self.list_objects(output_uri)
        self.remove_objects(output_uri)
        self.assertEqual(len(objects), 1, objects)

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_python_with_zip(self):
        """Tests running a python job with a zip from OCI as job artifact."""
        entrypoint = "job_archive/my_package/entrypoint_ads.py"
        output_uri = os.path.join(self.TEST_OUTPUT_URI, "python_zip")
        self.remove_objects(output_uri)
        runtime = (
            PythonRuntime()
            .with_source(self.OCI_SOURCE_PATH, entrypoint=entrypoint)
            # Add path to PYTHONPATH for package import.
            .with_python_path("job_archive")
            .with_service_conda("dbexp_p38_cpu_v1")
            .with_output(
                output_dir=self.TEST_OUTPUT_DIR,
                output_uri=os.path.join(self.TEST_OUTPUT_URI, "python_zip"),
            )
        )
        self.create_and_assert_job_run(
            runtime,
            [
                "2.6.8",
                "/home/datascience/decompressed_artifact/code",
            ]
            + self.TEST_LOGS_FUNCTION,
        )

        objects = self.list_objects(output_uri)
        self.remove_objects(output_uri)
        self.assertEqual(len(objects), 1, objects)

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_python_with_notebook_entrypoint_and_output_uri(self):
        """Tests running a notebook as entrypoint using PythonRuntime."""
        output_uri = os.path.join(self.TEST_OUTPUT_URI, "python_notebook")
        self.remove_objects(output_uri)
        runtime = (
            PythonRuntime(skip_metadata_update=True)
            .with_source(self.DIR_SOURCE_PATH)
            # Change the working dir so that the entrypoint can be simplified
            .with_working_dir("job_archive")
            .with_entrypoint("test_notebook.ipynb")
            .with_output("outputs", output_uri)
            .with_service_conda("dbexp_p38_cpu_v1")
        )
        self.create_and_assert_job_run(
            runtime,
            [
                "'2.6.8'",
                "This is the test notebook.",
                "This is a function in a module.",
                "This is a function in a package.",
            ],
        )
        objects = self.list_objects(output_uri)
        self.remove_objects(output_uri)
        self.assertEqual(objects, ["python_test.txt"])


class NotebookRuntimeJobRunTest(DSCJobRunTestCase):
    NOTEBOOK_PATH = os.path.join(
        os.path.dirname(__file__), "../fixtures/ads_check.ipynb"
    )

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_notebook(self):
        """Tests running a notebook"""
        runtime = (
            NotebookRuntime()
            .with_notebook(self.NOTEBOOK_PATH)
            .with_service_conda("dbexp_p38_cpu_v1")
        )
        self.create_and_assert_job_run(
            runtime,
            ["2.6.8"],
        )

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_notebook_in_dir(self):
        """Tests running a notebook in a folder"""
        output_uri = os.path.join(self.TEST_OUTPUT_URI, "notebook_runtime")
        runtime = (
            NotebookRuntime()
            .with_source(self.DIR_SOURCE_PATH, notebook="test_notebook.ipynb")
            .with_service_conda("dbexp_p38_cpu_v1")
            .with_output(output_uri)
        )
        self.remove_objects(output_uri)
        self.create_and_assert_job_run(
            runtime,
            [
                "'2.6.8'",
                "This is the test notebook.",
                "This is a function in a module.",
                "This is a function in a package.",
            ],
        )
        objects = self.list_objects(output_uri)
        self.remove_objects(output_uri)
        for obj in [
            "main.py",
            "my_module.py",
            "my_package",
            "outputs",
            "script.sh",
            "test_notebook.ipynb",
        ]:
            self.assertIn(obj, objects)

    @pytest.mark.skipif(SKIP_TEST_FLAG, reason=SKIP_TEST_REASON)
    def test_run_notebook_in_dir_with_invalid_path(self):
        """Tests running a notebook in a folder but the notebook is not found."""
        runtime = (
            NotebookRuntime()
            .with_source(self.NOTEBOOK_PATH, notebook="test_notebook.ipynb")
            .with_service_conda("dbexp_p38_cpu_v1")
        )
        with self.assertRaises(ValueError):
            self.create_and_assert_job_run(
                runtime,
                [],
            )
