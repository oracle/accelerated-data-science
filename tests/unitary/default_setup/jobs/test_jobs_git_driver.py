#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shlex
import sys
import subprocess
import tempfile
import unittest
from tests.unitary.default_setup.jobs.test_jobs_base import DataScienceJobPayloadTest

from ads.jobs import GitPythonRuntime
from ads.jobs.templates.driver_oci import ArgumentParser


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
            .with_service_conda("mlcpuv1")
        )

        expected_env_var = {
            "GIT_URL": self.TEST_GIT_URL,
            "PYTHON_PATH": self.TEST_PYTHON_PATH,
            "GIT_ENTRYPOINT": self.TEST_ENTRY_SCRIPT,
            "CONDA_ENV_SLUG": "mlcpuv1",
            "CONDA_ENV_TYPE": "service",
        }

        self.assert_runtime_translation(runtime, expected_env_var)

    def test_create_job_from_git_with_output(self):
        runtime = (
            GitPythonRuntime()
            .with_source(self.TEST_GIT_URL)
            .with_entrypoint(path=self.TEST_ENTRY_SCRIPT)
            .with_python_path(self.TEST_PYTHON_PATH)
            .with_service_conda("mlcpuv1")
            .with_output(
                output_dir=self.TEST_OUTPUT_DIR,
                output_uri=self.TEST_OUTPUT_URI,
            )
        )

        expected_env_var = {
            "GIT_URL": self.TEST_GIT_URL,
            "PYTHON_PATH": self.TEST_PYTHON_PATH,
            "GIT_ENTRYPOINT": self.TEST_ENTRY_SCRIPT,
            "CONDA_ENV_SLUG": "mlcpuv1",
            "CONDA_ENV_TYPE": "service",
            "OUTPUT_DIR": self.TEST_OUTPUT_DIR,
            "OUTPUT_URI": self.TEST_OUTPUT_URI,
        }

        self.assert_runtime_translation(runtime, expected_env_var)

    # Test cases for argument parsing
    # "input" contains a list of value simulting sys.argv.
    # "arguments" contains a string simulating the command line arguments.
    # The translation of GitPythonRuntime() with "args" and "kwargs" should produce the "arguments" string.
    # The translation process involves JSON dump for strings, which adds extra quotes and escapes.
    # "input" list is a alternative way to specify the same "args" and "kwargs" without JSON dump and escapes.
    # "arguments" may not generate the same "input" after Python parsed it into sys.argv.
    # However, the "arguments" and "input" should results in the same args and kwargs after git driver parsing.
    def assert_argument_parsing(self, cmd_arguments, argv_input, args, kwargs):
        expected_env_var = {
            "GIT_URL": self.TEST_GIT_URL,
            "PYTHON_PATH": self.TEST_PYTHON_PATH,
            "GIT_ENTRYPOINT": self.TEST_ENTRY_SCRIPT,
            "CONDA_ENV_SLUG": "mlcpuv1",
            "CONDA_ENV_TYPE": "service",
        }

        # Create runtime with arguments and check the OCI API payload
        runtime = (
            GitPythonRuntime()
            .with_source(self.TEST_GIT_URL)
            .with_entrypoint(path=self.TEST_ENTRY_SCRIPT)
            .with_python_path(self.TEST_PYTHON_PATH)
            .with_service_conda("mlcpuv1")
            .with_argument(*args, **kwargs)
        )

        self.assert_runtime_translation(runtime, expected_env_var, cmd_arguments)
        # Assume command line has been parsed into sys.argv as the input of the driver script
        parsed_args, parsed_kwargs = ArgumentParser(argv_input).parse()
        self.assertEqual(parsed_args, args)
        self.assertEqual(parsed_kwargs, kwargs)
        # Simulate command line argument parsing with shlex
        parsed_argv = shlex.split(cmd_arguments)
        parsed_args, parsed_kwargs = ArgumentParser(parsed_argv).parse()
        self.assertEqual(parsed_args, args)
        self.assertEqual(parsed_kwargs, kwargs)

    def test_parsing_positional_arguments(self):
        test_case = {
            "argv_input": ["arg1", "arg2"],
            "args": ["arg1", "arg2"],
            "kwargs": {},
            "cmd_arguments": "arg1 arg2",
        }
        self.assert_argument_parsing(**test_case)

    def test_parsing_keyword_arguments(self):
        test_case = {
            "argv_input": ["arg1", "arg2", "--key", "val"],
            "args": ["arg1", "arg2"],
            "kwargs": {"key": "val"},
            "cmd_arguments": "arg1 arg2 --key val",
        }
        self.assert_argument_parsing(**test_case)

    def test_parsing_empty_arguments(self):
        test_case = {
            "argv_input": ["arg1", "", "--key1", "", "--key2"],
            "args": ["arg1", ""],
            "kwargs": {"key1": "", "key2": None},
            "cmd_arguments": "arg1 '' --key1 '' --key2",
        }
        self.assert_argument_parsing(**test_case)

    def test_parsing_list_or_dict_arguments(self):
        test_case = {
            "argv_input": [
                '["arg1a", "arg1b"]',
                '"arg2"',
                "--key1",
                '["val1a", "val1b"]',
                "--key2",
                '{"key": "val2"}',
            ],
            "args": [["arg1a", "arg1b"], "arg2"],
            "kwargs": {"key1": ["val1a", "val1b"], "key2": {"key": "val2"}},
            "cmd_arguments": '\'["arg1a", "arg1b"]\' arg2 --key1 \'["val1a", "val1b"]\' --key2 \'{"key": "val2"}\'',
        }
        self.assert_argument_parsing(**test_case)

    def test_parsing_arguments_with_space(self):
        test_case = {
            "argv_input": ["arg1", "arg2", "--key1", "val1a val1b", "--key2", "val2"],
            "args": ["arg1", "arg2"],
            "kwargs": {"key1": "val1a val1b", "key2": "val2"},
            "cmd_arguments": "arg1 arg2 --key1 'val1a val1b' --key2 val2",
        }
        self.assert_argument_parsing(**test_case)

    def test_parsing_arguments_with_quotes_in_string(self):
        test_case = {
            "argv_input": ["--key1", '["val1a", "val1b"]', "--key2", 'Not "JSON"'],
            "args": [],
            "kwargs": {"key1": ["val1a", "val1b"], "key2": 'Not "JSON"'},
            "cmd_arguments": '--key1 \'["val1a", "val1b"]\' --key2 \'Not "JSON"\'',
        }
        self.assert_argument_parsing(**test_case)

    def test_parsing_arguments_with_quotes_around_string(self):
        test_case = {
            "argv_input": ['["arg1a", "arg1b"]', '"\\"arg2\\""'],
            "args": [["arg1a", "arg1b"], '"arg2"'],
            "kwargs": {},
            "cmd_arguments": '\'["arg1a", "arg1b"]\' \'"\\"arg2\\""\'',
        }
        self.assert_argument_parsing(**test_case)

    def test_parsing_json_payload_arguments(self):
        test_case = {
            "argv_input": [
                "--key1",
                '"[\\"val1a\\", \\"val1b\\"]"',
                "--key2",
                '"{\\"key\\": \\"val2\\"}"',
            ],
            "args": [],
            "kwargs": {"key1": '["val1a", "val1b"]', "key2": '{"key": "val2"}'},
            "cmd_arguments": '--key1 \'"[\\"val1a\\", \\"val1b\\"]"\' --key2 \'"{\\"key\\": \\"val2\\"}"\'',
        }
        self.assert_argument_parsing(**test_case)

    # It is not possible to generate the following input with a single call of with_argument
    # Multiple values for keyword arguments
    # {
    #     "argv_input": ["arg1", "arg2", "--key1", "val1a", "val1b", "--key2", "val2"],
    # },


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
                self.assertIn("Job completed.", lines)
                self.assertIn("This is a function in a module.", lines)
                self.assertIn("This is a function in a package.", lines)
                for line in lines:
                    print(line)
                return lines
            except subprocess.CalledProcessError as e:
                print(e.output)
                print(e.stderr)
                self.fail("Error occurred when running Git driver.")


class GitDriverLocalRunTest(GitDriverRunTest):
    def test_run_script_from_git_branch(self):
        lines = self.assert_git_driver_locally(
            {
                "GIT_URL": self.TEST_GIT_URL,
                "GIT_BRANCH": "develop",
                "PYTHON_PATH": self.TEST_PYTHON_PATH,
                "CONDA_ENV_SLUG": "mlcpuv1",
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
                "CONDA_ENV_SLUG": "mlcpuv1",
                "CONDA_ENV_TYPE": "service",
                "GIT_ENTRYPOINT": self.TEST_ENTRY_SCRIPT,
                "ENTRY_FUNCTION": "entry_function",
            }
        )
