#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
import tempfile

import fsspec
from ads.common.auth import default_signer, AuthType
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    NotebookRuntimeHandler,
)
from tests.unitary.default_setup.jobs.test_jobs_base import DriverRunTest
from ads.jobs.templates.driver_utils import OCIHelper
from ads.jobs.builders.runtimes.artifact import PythonArtifact


class NotebookDriverRunTest(DriverRunTest):
    def run_notebook(
        self, notebook_path, output_uri=None, env_vars=None, suppress_error=False
    ):
        """Asserts running a notebook with notebook driver locally

        Parameters
        ----------
        notebook_path : str
            Path of the notebook
        output_uri : str, optional
            URI for storing the outputs, by default None
        env_vars : dict, optional
            Environment variables, by default None
        suppress_error : bool
            Whether to suppress the exception when there is an error running the driver.
            When there is an error running the notebook:
                If this is set to False, an exception will be raised and no output will be returned.
                If this is set to True, no exception will be raised and the outputs will be returned.

        Returns
        -------
        list
            output messages.
        """
        if not env_vars:
            env_vars = {}
        # Use a temporary directory as working directory
        with tempfile.TemporaryDirectory() as working_dir:
            # Copy driver and notebook to temporary directory
            for driver in ["driver_notebook.py", "driver_utils.py"]:
                driver_src = os.path.join("ads/jobs/templates", driver)
                driver_dst = os.path.join(working_dir, driver)
                shutil.copy(driver_src, driver_dst)

            test_driver = os.path.join(working_dir, "driver_notebook.py")
            code_dir = os.path.join(working_dir, PythonArtifact.USER_CODE_DIR)
            os.mkdir(code_dir)
            shutil.copy(
                notebook_path,
                os.path.join(code_dir, os.path.basename(notebook_path)),
            )
            # Set envs for the driver
            env_vars["JOB_RUN_NOTEBOOK"] = os.path.basename(notebook_path)
            # TeamCity will use Instance Principal, when running locally - set OCI_IAM_TYPE to security_token
            env_vars["OCI_IAM_TYPE"] = os.getenv(
                "OCI_IAM_TYPE", AuthType.INSTANCE_PRINCIPAL
            )
            if output_uri:
                # Clear the files in output URI
                try:
                    # Ignore the error for unit tests.
                    fs = fsspec.filesystem("oci", **default_signer())
                    if fs.find(output_uri):
                        fs.rm(output_uri, recursive=True)
                except:
                    pass
                env_vars["OUTPUT_URI"] = output_uri
            return super().run_driver(
                test_driver, env_vars=env_vars, suppress_error=suppress_error
            )


class NotebookDriverLocalTest(NotebookDriverRunTest):
    def test_run_notebook_with_exclude_tags(self):
        """Tests running a notebook and excluding some cells."""
        notebook_path = os.path.join(
            os.path.dirname(__file__),
            "../fixtures/exclude_check.ipynb",
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
                OCIHelper.substitute_output_uri(input_uri),
                expected_uri,
                f"Expect {input_uri} to be converted to {expected_uri}.",
            )

    def test_copy_output_calls(self):
        """Tests if the copy output function is called when there is an error running NotebookRuntime."""
        notebook_path = os.path.join(
            os.path.dirname(__file__),
            "../fixtures/notebook_with_error.ipynb",
        )
        # Here we set an invalid value for the output uri
        # The bucket and namespace info is missing
        outputs = self.run_notebook(
            notebook_path, output_uri="oci://path/to/output", suppress_error=True
        )
        # The driver should print out an error message about the invalid output uri
        # if the copy output function is called.
        self.assertIn(
            "Output URI should have the format of oci://bucket@namespace/path/to/dir",
            outputs,
        )
