#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shlex
import sys
import subprocess
import tempfile
import unittest
from tests.unitary.default_setup.jobs.test_jobs_base import DataScienceJobPayloadTest

from ads.jobs import Job, DataScienceJob, GitPythonRuntime
from ads.jobs.templates.driver_utils import ArgumentParser


class GitDriverTestBase(unittest.TestCase):
    TEST_GIT_URL = "https://github.com/qiuosier/python_test.git"
    TEST_ENTRY_SCRIPT = "src/main.py"
    TEST_ENTRY_FUNCTION = ""
    TEST_PYTHON_PATH = "src"
    TEST_OUTPUT_DIR = "output"
    TEST_OUTPUT_URI = "oci://test_bucket@test_namespace/git_test"


class GitDriverTest(DataScienceJobPayloadTest, GitDriverTestBase):
    def test_create_job_from_git(self):
        runtime = (
            GitPythonRuntime()
            .with_source(self.TEST_GIT_URL)
            .with_entrypoint(path=self.TEST_ENTRY_SCRIPT)
            .with_python_path(self.TEST_PYTHON_PATH)
            .with_service_conda("slug")
        )

        expected_env_var = {
            "GIT_URL": self.TEST_GIT_URL,
            "PYTHON_PATH": self.TEST_PYTHON_PATH,
            "GIT_ENTRYPOINT": self.TEST_ENTRY_SCRIPT,
            "CONDA_ENV_SLUG": "slug",
            "CONDA_ENV_TYPE": "service",
            "JOB_RUN_ENTRYPOINT": "driver_oci.py",
        }

        self.assert_runtime_translation(runtime, expected_env_var)

    def test_create_job_from_git_with_output(self):
        runtime = (
            GitPythonRuntime()
            .with_source(self.TEST_GIT_URL)
            .with_entrypoint(path=self.TEST_ENTRY_SCRIPT)
            .with_python_path(self.TEST_PYTHON_PATH)
            .with_service_conda("slug")
            .with_output(
                output_dir=self.TEST_OUTPUT_DIR,
                output_uri=self.TEST_OUTPUT_URI,
            )
        )

        expected_env_var = {
            "GIT_ENTRYPOINT": self.TEST_ENTRY_SCRIPT,
            "GIT_URL": self.TEST_GIT_URL,
            "PYTHON_PATH": self.TEST_PYTHON_PATH,
            "JOB_RUN_ENTRYPOINT": "driver_oci.py",
            "CONDA_ENV_SLUG": "slug",
            "CONDA_ENV_TYPE": "service",
            "OUTPUT_DIR": self.TEST_OUTPUT_DIR,
            "OUTPUT_URI": self.TEST_OUTPUT_URI,
        }

        self.assert_runtime_translation(runtime, expected_env_var)


class EntryFunctionArgumentParsingTest(DataScienceJobPayloadTest, GitDriverTestBase):
    """Contains test cases for argument parsing

    In the test cases, args and kwargs will be used to create the runtime.
    When creating the job, it should contain the expected cmd_arguments.
    The ArgumentParser should be able to parse the cmd_arguments and return args and kwargs.

    """

    EXPECTED_ENV_VAR = {
        "JOB_RUN_ENTRYPOINT": "driver_oci.py",
        "GIT_URL": GitDriverTestBase.TEST_GIT_URL,
        "PYTHON_PATH": GitDriverTestBase.TEST_PYTHON_PATH,
        "GIT_ENTRYPOINT": GitDriverTestBase.TEST_ENTRY_SCRIPT,
        "CONDA_ENV_SLUG": "slug",
        "CONDA_ENV_TYPE": "service",
    }

    def assert_argument_parsing(self, args, kwargs, cmd_arguments):
        # Create runtime with arguments and check the OCI API payload
        runtime = (
            GitPythonRuntime()
            .with_source(self.TEST_GIT_URL)
            .with_entrypoint(path=self.TEST_ENTRY_SCRIPT)
            .with_python_path(self.TEST_PYTHON_PATH)
            .with_service_conda("slug")
            .with_argument(*args, **kwargs)
        )

        self.assert_runtime_translation(runtime, self.EXPECTED_ENV_VAR, cmd_arguments)
        # Simulate command line argument parsing with shlex
        parsed_argv = shlex.split(cmd_arguments)
        parsed_args, parsed_kwargs = ArgumentParser(parsed_argv).parse()
        self.assertEqual(parsed_args, args)
        self.assertEqual(parsed_kwargs, kwargs)

    def test_parsing_positional_arguments(self):
        test_case = {
            "args": ["arg1", "arg2"],
            "kwargs": {},
            "cmd_arguments": "arg1 arg2",
        }
        self.assert_argument_parsing(**test_case)

    def test_parsing_keyword_arguments(self):
        test_case = {
            "args": ["arg1", "arg2"],
            "kwargs": {"key": "val"},
            "cmd_arguments": "arg1 arg2 --key val",
        }
        self.assert_argument_parsing(**test_case)

    def test_parsing_empty_arguments(self):
        test_case = {
            "args": ["arg1", ""],
            "kwargs": {"key1": "", "key2": None},
            "cmd_arguments": "arg1 '' --key1 '' --key2",
        }
        self.assert_argument_parsing(**test_case)

    def test_parsing_arguments_with_space(self):
        test_case = {
            "args": ["arg1", "arg2"],
            "kwargs": {"key1": "val1a val1b", "key2": "val2"},
            "cmd_arguments": "arg1 arg2 --key1 'val1a val1b' --key2 val2",
        }
        self.assert_argument_parsing(**test_case)

    def test_parsing_arguments_with_quotes_around_string(self):
        test_case = {
            "args": ["'arg1'", '"arg2"'],
            "kwargs": {"key": 'A "Val"'},
            "cmd_arguments": "''\"'\"'arg1'\"'\"'' '\"arg2\"' --key 'A \"Val\"'",
        }
        self.assert_argument_parsing(**test_case)

    def test_parsing_multiple_values_in_keyword_argument(self):
        # Create runtime with arguments and check the OCI API payload
        runtime = (
            GitPythonRuntime()
            .with_source(self.TEST_GIT_URL)
            .with_entrypoint(path=self.TEST_ENTRY_SCRIPT)
            .with_python_path(self.TEST_PYTHON_PATH)
            .with_service_conda("slug")
            .with_argument("arg1", "arg2")
            .with_argument(key1="val1")
            .with_argument("arg3")
            .with_argument(key2="val2")
        )

        args = ["arg1", "arg2", "arg3"]
        kwargs = {"key1": "val1", "key2": "val2"}
        cmd_arguments = "arg1 arg2 --key1 val1 arg3 --key2 val2"

        self.assert_runtime_translation(runtime, self.EXPECTED_ENV_VAR, cmd_arguments)
        # Simulate command line argument parsing with shlex
        parsed_argv = shlex.split(cmd_arguments)
        parsed_args, parsed_kwargs = ArgumentParser(parsed_argv).parse()
        self.assertEqual(parsed_args, args)
        self.assertEqual(parsed_kwargs, kwargs)


class GitDriverRunTest(GitDriverTestBase):
    def assert_git_driver_locally(self, env_vars):
        with tempfile.TemporaryDirectory() as code_dir:
            env_vars["CODE_DIR"] = code_dir
            try:
                outputs = subprocess.check_output(
                    [
                        os.path.join(os.path.dirname(sys.executable), "coverage"),
                        "run",
                        "--concurrency=multiprocessing",
                        "ads/jobs/templates/driver_oci.py",
                        "arg",
                        "--key",
                        "val",
                    ],
                    env=env_vars,
                )
                lines = outputs.decode().split("\n")
                self.assertIn("This is a function in a module.", lines)
                self.assertIn("This is a function in a package.", lines)
                for line in lines:
                    print(line)
                return lines
            except subprocess.CalledProcessError as exc:
                print(exc.output)
                print(exc.stderr)
                self.fail("Error occurred when running Git driver.")


class GitDriverLocalRunTest(GitDriverRunTest):
    def test_run_script_from_git_branch(self):
        lines = self.assert_git_driver_locally(
            {
                "GIT_URL": self.TEST_GIT_URL,
                "GIT_BRANCH": "develop",
                "PYTHON_PATH": self.TEST_PYTHON_PATH,
                "CONDA_ENV_SLUG": "slug",
                "CONDA_ENV_TYPE": "service",
                "GIT_ENTRYPOINT": self.TEST_ENTRY_SCRIPT,
            }
        )

        self.assertIn("This is a function in a module.", lines)
        self.assertIn("This is a function in a package.", lines)
        self.assertIn("This is the main script in develop branch.", lines)

    def test_run_function_from_git_commit(self):
        self.assert_git_driver_locally(
            {
                "GIT_URL": self.TEST_GIT_URL,
                "GIT_COMMIT": "10461a65d5728a6620cca945853e04cab2976071",
                "PYTHON_PATH": self.TEST_PYTHON_PATH,
                "CONDA_ENV_SLUG": "slug",
                "CONDA_ENV_TYPE": "service",
                "GIT_ENTRYPOINT": self.TEST_ENTRY_SCRIPT,
                "ENTRY_FUNCTION": "entry_function",
            }
        )
