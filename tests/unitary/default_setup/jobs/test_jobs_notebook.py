#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
import tempfile

import fsspec
import pytest
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    NotebookArtifact,
    NotebookRuntimeHandler,
)
from tests.unitary.default_setup.jobs.test_jobs_base import DriverRunTest
from ads.jobs.templates.driver_notebook import substitute_output_uri


class NotebookDriverRunTest(DriverRunTest):
    DRIVER_PATH = os.path.join(
        "ads/jobs/templates", NotebookArtifact.CONST_DRIVER_SCRIPT
    )

    def run_notebook(self, notebook_path, output_uri=None, env_vars=None):
        """Asserts running a notebook with notebook driver locally

        Parameters
        ----------
        notebook_path : str
            Path of the notebook
        output_uri : str, optional
            URI for storing the outputs, by default None
        env_vars : dict, optional
            Environment variables, by default None
        """
        if not env_vars:
            env_vars = {}
        # Use a temporary directory as working directory
        with tempfile.TemporaryDirectory() as working_dir:
            # Copy driver and notebook to temporary directory
            test_driver = os.path.join(working_dir, os.path.basename(self.DRIVER_PATH))
            shutil.copy(self.DRIVER_PATH, test_driver)
            shutil.copy(
                notebook_path,
                os.path.join(working_dir, os.path.basename(notebook_path)),
            )
            # Set envs for the driver
            env_vars["JOB_RUN_NOTEBOOK"] = os.path.basename(notebook_path)
            if output_uri:
                # Clear the files in output URI
                fs = fsspec.filesystem(
                    "oci", config=os.path.expanduser("~/.oci/config")
                )
                if fs.find(output_uri):
                    fs.rm(output_uri, recursive=True)
                env_vars["OUTPUT_URI"] = output_uri
            return super().run_driver(test_driver, env_vars=env_vars)


class NotebookDriverLocalTest(NotebookDriverRunTest):
    @pytest.mark.skipif(
        "NoDependency" in os.environ,
        reason="skip for dependency test: nbformat, nbconvert",
    )
    def test_run_notebook_with_exclude_tags(self):
        """Tests running a notebook and excluding some cells."""
        notebook_path = os.path.join(
            os.path.dirname(__file__),
            "../../../integration/fixtures/exclude_check.ipynb",
        )
        outputs = self.run_notebook(
            notebook_path,
            env_vars={
                NotebookRuntimeHandler.CONST_EXCLUDE_TAGS: '["ignore", "remove"]'
            },
        )
        self.assertIn("8", outputs)
        self.assertIn("test", outputs)
        self.assertIn("hello", outputs)
        self.assertIn("hello world", outputs)
        self.assertIn("another line", outputs)
        self.assertNotIn("ignore", outputs)
        self.assertNotIn('"ignore"', outputs)
        self.assertNotIn("'ignore'", outputs)

    def test_substitute_output_uri(self):
        envs = dict(A="foo", B="bar")
        test_cases = [
            ("oci://path/to/my_$B", "oci://path/to/my_bar"),
            ("oci://path/to/my_$", "oci://path/to/my_$"),
            ("oci://path/to/my_$C", "oci://path/to/my_$C"),
            ("oci://path/to/my_$$A", "oci://path/to/my_$foo"),
            ("oci://path/$A/$B", "oci://path/foo/bar"),
        ]
        os.environ.update(envs)
        for input_uri, expected_uri in test_cases:
            self.assertEqual(
                substitute_output_uri(input_uri),
                expected_uri,
                f"Expect {input_uri} to be converted to {expected_uri}.",
            )
