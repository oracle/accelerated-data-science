#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/up
"""Contains tests for DataScienceJobRun class."""

from unittest import TestCase, mock

import oci

from ads.jobs import DataScienceJobRun


class JobRunTestCase(TestCase):
    """Contains test cases for Job runs."""

    @mock.patch("ads.jobs.builders.infrastructure.dsc_job.DataScienceJobRun.sync")
    def test_job_run_wait(self, *args):
        """Test waiting for job run."""
        run = DataScienceJobRun()
        run.lifecycle_state = oci.data_science.models.JobRun.LIFECYCLE_STATE_IN_PROGRESS

        # Mock time.sleep to change the job run lifecycle state.
        def mock_sleep(*args, **kwargs):
            run.lifecycle_state = (
                oci.data_science.models.JobRun.LIFECYCLE_STATE_SUCCEEDED
            )

        with mock.patch("time.sleep", wraps=mock_sleep):
            run.wait()

        self.assertEqual(
            run.lifecycle_state,
            oci.data_science.models.JobRun.LIFECYCLE_STATE_SUCCEEDED,
        )

    def test_job_run_exit_code(self):
        """Tests job run exit code."""
        run = DataScienceJobRun()
        self.assertEqual(run.exit_code, None)
        run.lifecycle_state = run.LIFECYCLE_STATE_SUCCEEDED
        self.assertEqual(run.exit_code, 0)
        run.lifecycle_state = run.LIFECYCLE_STATE_IN_PROGRESS
        run.lifecycle_details = "Job run in progress."
        self.assertEqual(run.exit_code, None)
        run.lifecycle_state = run.LIFECYCLE_STATE_FAILED
        run.lifecycle_details = "Job run artifact execution failed with exit code 21."
        self.assertEqual(run.exit_code, 21)

    @mock.patch("ads.jobs.builders.infrastructure.dsc_job.DataScienceJobRun.cancel")
    @mock.patch("ads.common.oci_datascience.OCIDataScienceMixin.delete")
    def test_job_run_delete(self, mock_delete, mock_cancel):
        """Tests deleting job run."""
        run = DataScienceJobRun()
        # Cancel will not be called if job run is succeeded.
        run.lifecycle_state = run.LIFECYCLE_STATE_SUCCEEDED
        run.delete()
        mock_delete.assert_called_once()
        mock_cancel.assert_not_called()
        # Cancel will be called if job run is in progress and force_delete is set.
        run.lifecycle_state = run.LIFECYCLE_STATE_IN_PROGRESS
        run.delete(force_delete=True)
        mock_cancel.assert_called_once()
