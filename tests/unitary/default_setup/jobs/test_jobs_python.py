#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
import tempfile
from zipfile import ZipFile

from ads.jobs import PythonRuntime
from ads.jobs.builders.runtimes.artifact import PythonArtifact
from ads.jobs.builders.infrastructure.dsc_job_runtime import PythonRuntimeHandler
from ads.jobs.templates.driver_python import USER_CODE_DIR
from tests.unitary.default_setup.jobs.test_jobs_base import (
    DriverRunTest,
    DataScienceJobPayloadTest,
)


class PythonRuntimeTest(DataScienceJobPayloadTest):
    """Contains tests from PythonRuntime in ADS Jobs API."""

    DIR_SOURCE_PATH = os.path.join(
        os.path.dirname(__file__), "../../../integration/fixtures/job_archive"
    )
    SCRIPT_SOUCE_PATH = os.path.join(
        os.path.dirname(__file__), "../../../integration/fixtures/job_archive/main.py"
    )

    def test_prepare_artifact_with_dir(self):
        """Tests preparing a directory as job artifact."""
        with PythonArtifact(self.DIR_SOURCE_PATH) as artifact:
            with ZipFile(artifact.path, "r") as zip_file:
                files = zip_file.namelist()
                files = [f for f in files if "__pycache__" not in f]
                files.sort()
                expected_files = [
                    "code/",
                    "code/job_archive/",
                    "driver_python.py",
                    "code/job_archive/my_package/",
                    "code/job_archive/my_module.py",
                    "code/job_archive/script.sh",
                    "code/job_archive/main.py",
                    "code/job_archive/my_package/__init__.py",
                    "code/job_archive/my_package/entrypoint.py",
                    "code/job_archive/my_package/entrypoint_ads.py",
                    "code/job_archive/my_package/utils.py",
                ]
                expected_files.sort()

                self.assertEqual(files, expected_files)

    def test_prepare_artifact_with_script(self):
        """Tests preparing a python script as job artifact."""
        with PythonArtifact(self.SCRIPT_SOUCE_PATH) as artifact:
            with ZipFile(artifact.path, "r") as zip_file:
                files = zip_file.namelist()
                files.sort()
                expected_files = ["driver_python.py", "code/", "code/main.py"]
                expected_files.sort()

                self.assertEqual(files, expected_files)

    def test_create_job_with_python_runtime(self):
        """Tests the translation of PythonRuntime to OCI API Payload."""
        entrypoint = "my_package/entrypoint.py"
        expected_env_var = {
            PythonRuntimeHandler.CONST_JOB_ENTRYPOINT: "driver_python.py",
            PythonRuntimeHandler.CONST_CODE_ENTRYPOINT: "my_package/entrypoint.py",
            PythonRuntimeHandler.CONST_CONDA_TYPE: "service",
            PythonRuntimeHandler.CONST_CONDA_SLUG: "mlcpuv1",
        }

        runtime = (
            PythonRuntime()
            .with_source(self.DIR_SOURCE_PATH, entrypoint=entrypoint)
            .with_service_conda("mlcpuv1")
        )
        self.assert_runtime_translation(runtime, expected_env_var)


class PythonDriverTest(DriverRunTest):
    DRIVER_PATH = os.path.join("ads/jobs/templates", "driver_python.py")

    def test_run_python_driver(self):
        """Tests running the PythonRuntime driver script locally."""
        env_vars = {}

        # Use a temporary directory as working directory
        with tempfile.TemporaryDirectory() as working_dir:
            # Copy driver and notebook to temporary directory
            test_driver = os.path.join(working_dir, os.path.basename(self.DRIVER_PATH))
            shutil.copy(self.DRIVER_PATH, test_driver)

            code_dir = os.path.join(working_dir, USER_CODE_DIR)
            dir_path = "./test_files/job_archive"
            src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_path)
            shutil.copytree(src_dir, code_dir)
            # Set envs for the driver
            env_vars["CODE_ENTRYPOINT"] = "my_package/entrypoint.py"

            outputs = super().run_driver(test_driver, env_vars=env_vars)

        # Checks the outputs
        outputs = [output for output in outputs if output]
        self.assertEqual(
            len(outputs),
            5,
            f"Unexpected or missing messages.\nActual outputs:\n{outputs}",
        )
        self.assertIn("This is the entrypoint inside a package.", outputs)
        self.assertIn("This is a function in a module.", outputs)
        self.assertIn("This is a function in a package.", outputs)
        self.assertTrue(
            outputs[-2].endswith(os.path.abspath(os.path.join(working_dir, code_dir)))
        )
