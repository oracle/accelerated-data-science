#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import pytest
import os
import tempfile
from zipfile import ZipFile

import fsspec
from ads.common.auth import default_signer
from tests.integration.config import secrets
from tests.integration.jobs.test_dsc_job import DSCJobTestCaseWithCleanUp
from tests.integration.jobs.test_jobs_notebook import NotebookDriverRunTest
from ads.jobs import Job, NotebookRuntime
from ads.jobs.builders.runtimes.artifact import NotebookArtifact
from ads.jobs.builders.infrastructure.dsc_job_runtime import NotebookRuntimeHandler


class NotebookRuntimeTest(DSCJobTestCaseWithCleanUp):
    NOTEBOOK_PATH = os.path.join(
        os.path.dirname(__file__), "../fixtures/ads_check.ipynb"
    )
    NOTEBOOK_PATH_EXCLUDE = os.path.join(
        os.path.dirname(__file__), "../fixtures/exclude_check.ipynb"
    )

    EXPECTED_JOB_CONFIG = {
        "command_line_arguments": None,
        "environment_variables": {
            "CONDA_ENV_SLUG": "mlcpuv1",
            "CONDA_ENV_TYPE": "service",
            NotebookRuntimeHandler.CONST_NOTEBOOK_ENCODING: "utf-8",
            NotebookRuntimeHandler.CONST_ENTRYPOINT: NotebookArtifact.CONST_DRIVER_SCRIPT,
            NotebookRuntimeHandler.CONST_NOTEBOOK_NAME: os.path.basename(NOTEBOOK_PATH),
        },
        # "hyperparameter_values": None,
        "job_type": "DEFAULT",
        "maximum_runtime_in_minutes": None,
    }

    def test_create_job_with_notebook(self):
        """Tests creating a job from notebook"""
        # Creates a job with service conda and notebook
        job = (
            Job()
            .with_infrastructure(self.default_datascience_job.with_log_id(self.LOG_ID))
            .with_runtime(
                NotebookRuntime()
                .with_notebook(self.NOTEBOOK_PATH)
                .with_service_conda("mlcpuv1")
            )
            .create()
        )
        # Once we call create(), job.infrastructure.dsc_job.job_configuration_details will be loaded from OCI
        # We can check this to see if the job is created with the correct conda pack
        self.assertEqual(
            json.loads(str(job.infrastructure.dsc_job.job_configuration_details)),
            self.EXPECTED_JOB_CONFIG,
        )
        # Load the job from OCI, this will make sure the job object is coming from OCI.
        job = Job.from_datascience_job(job.id)
        self.assertIsInstance(job.runtime, NotebookRuntime)
        self.assertEqual(job.runtime.notebook_uri, os.path.basename(self.NOTEBOOK_PATH))
        # Download the job artifact and see if it is converted correctly.
        try:
            # Show all diff when there is an error.
            self.maxDiff = None
            # Set delete=False when creating NamedTemporaryFile,
            # Otherwise, the file will be delete when download_artifact() close the file.
            artifact = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            # Close the file since NamedTemporaryFile() opens the file by default.
            artifact.close()
            # Download job artifact from OCI
            job.infrastructure.dsc_job.download_artifact(artifact.name)
            # Read the file and check the content
            with ZipFile(artifact.name, "r") as zip_file:
                files = zip_file.namelist()
                self.assertEqual(len(files), 4, str(files))
                self.assertIn(NotebookArtifact.CONST_DRIVER_SCRIPT, files)
                self.assertIn(NotebookArtifact.CONST_DRIVER_UTILS, files)
                self.assertIn("code/" + os.path.basename(self.NOTEBOOK_PATH), files)

        finally:
            # Clean up the file
            os.unlink(artifact.name)


class NotebookDriverIntegrationTest(NotebookDriverRunTest):
    @pytest.mark.skip(
        reason="api_keys not an option anymore, this test is candidate to be removed"
    )
    def test_notebook_driver_with_outputs(self):
        """Tests run the notebook driver with a notebook plotting and saving data."""
        # Notebook to be executed
        notebook_path = os.path.join(
            os.path.dirname(__file__), "../fixtures/plot.ipynb"
        )
        # Object storage output location
        output_uri = f"oci://{secrets.jobs.BUCKET_B}@{secrets.common.NAMESPACE}/notebook_driver_int_test/plot/"
        # Run the notebook with driver and check the logs
        outputs = self.run_notebook(notebook_path, output_uri=output_uri)
        self.assertIn("Import Finished.", outputs)
        self.assertIn("Data saved to JSON.", outputs)

        # Check the notebook saved to object storage.
        with fsspec.open(
            os.path.join(output_uri, os.path.basename(notebook_path)),
            **default_signer(),
        ) as f:
            outputs = [cell.get("outputs") for cell in json.load(f).get("cells")]
            # There should be 7 cells in the notebook
            self.assertEqual(len(outputs), 7)
            self.assertEqual(outputs[0][0].get("output_type"), "stream")
            # Cell 5 contains the image
            self.assertEqual(outputs[5][0].get("output_type"), "execute_result")
            self.assertEqual(outputs[5][1].get("output_type"), "display_data")
            self.assertIsNotNone(outputs[5][1].get("data").get("image/png"))
        # Check the JSON output file from the notebook
        with fsspec.open(
            os.path.join(output_uri, "data.json"),
            **default_signer(),
        ) as f:
            data = json.load(f)
            # There should be 10 data points
            self.assertIsInstance(data, dict)
            self.assertIn("x", data)
            self.assertIn("y", data)
            self.assertEqual(len(data.get("x")), 10)

        # Try to download the outputs
        with tempfile.TemporaryDirectory() as working_dir:
            job = Job().with_runtime(NotebookRuntime().with_output(output_uri))
            job.download(working_dir)
            files = os.listdir(working_dir)
            # There should be 3 files
            self.assertEqual(len(files), 3)
            self.assertIn("new_file.txt", files)
            self.assertIn("plot.ipynb", files)
            self.assertIn("data.json", files)
