#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import json
import os
from tests.integration.jobs.test_dsc_job import DSCJobTestCaseWithCleanUp
from ads.jobs import Job, PythonRuntime
from ads.jobs.builders.runtimes.artifact import PythonArtifact
from ads.jobs.builders.infrastructure.dsc_job_runtime import PythonRuntimeHandler


class PythonRuntimTest(DSCJobTestCaseWithCleanUp):
    DRIVER_PATH = os.path.join("ads/jobs/templates", "driver_python.py")

    DIR_SOURCE_PATH = os.path.join(
        os.path.dirname(__file__), "../fixtures/job_archive"
    )
    DIR_ENTRYPOINT = "job_archive/my_package/entrypoint.py"
    CONST_SCRIPT_PATH = "job_archive"

    def test_jobs_python_runtime_with_dir(self):
        job = (
            Job()
            .with_infrastructure(self.default_datascience_job)
            .with_runtime(
                PythonRuntime()
                .with_source(self.DIR_SOURCE_PATH)
                .with_entrypoint(self.DIR_ENTRYPOINT)
                .with_service_conda("mlcpuv1")
            )
            .create()
        )

        self.assertEqual(
            json.loads(str(job.infrastructure.dsc_job.job_configuration_details)),
            {
                "command_line_arguments": None,
                "environment_variables": {
                    "CONDA_ENV_SLUG": "mlcpuv1",
                    "CONDA_ENV_TYPE": "service",
                    PythonRuntimeHandler.CONST_JOB_ENTRYPOINT: PythonArtifact.CONST_DRIVER_SCRIPT,
                    PythonRuntimeHandler.CONST_CODE_ENTRYPOINT: self.DIR_ENTRYPOINT,
                },
                # "hyperparameter_values": None,
                "job_type": "DEFAULT",
                "maximum_runtime_in_minutes": None,
            },
        )

        expected_infra_spec = copy.deepcopy(self.DEFAULT_INFRA_SPEC)
        # expect display name include name of artifact
        expected_infra_spec["displayName"] = "job_archive"
        # Load the job from OCI and check the configurations
        # However, it is not possible to recover the source path (scriptPathURI)
        # scriptPathURI is the local path of the artifact when the user first upload them.
        job = Job.from_datascience_job(job.id)
        self.assertIsInstance(job.runtime, PythonRuntime)
        self.assert_job_creation(
            job,
            expected_infra_spec,
            {
                PythonRuntime.CONST_CONDA: {
                    PythonRuntime.CONST_CONDA_TYPE: "service",
                    PythonRuntime.CONST_CONDA_SLUG: "mlcpuv1",
                },
                PythonRuntime.CONST_ENTRYPOINT: self.DIR_ENTRYPOINT,
                PythonRuntime.CONST_SCRIPT_PATH: self.CONST_SCRIPT_PATH,
            },
        )
