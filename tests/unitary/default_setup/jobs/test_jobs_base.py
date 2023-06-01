#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import os
import subprocess
import sys
import unittest
from unittest import mock
from zipfile import ZipFile
from unittest.mock import patch
import pytest


from ads.jobs import (
    Job,
    infrastructure,
    ScriptRuntime,
    NotebookRuntime,
)
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    CondaRuntimeHandler,
    ScriptRuntimeHandler,
    NotebookRuntimeHandler,
)
from ads.jobs.builders.runtimes.artifact import NotebookArtifact, ScriptArtifact
from ads.jobs.builders.infrastructure.base import RunInstance
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    DataScienceJobRuntimeManager,
    ContainerRuntimeHandler,
)


class DataScienceJobPayloadTest(unittest.TestCase):
    """Base class for testing data science job creation"""

    PAYLOAD_TEMPLATE = {
        "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
        "projectId": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
        "jobConfigurationDetails": {"jobType": "DEFAULT", "environmentVariables": {}},
        "jobInfrastructureConfigurationDetails": {
            "jobInfrastructureType": "ME_STANDALONE",
            "blockStorageSizeInGBs": 50,
        },
    }

    SCRIPT_URI = "my_script.py"

    def setUp(self) -> None:
        self.maxDiff = None
        return super().setUp()

    def mock_create_job(self, job):
        with mock.patch(
            "ads.jobs.builders.infrastructure.dsc_job.DSCJob._create_with_oci_api"
        ):
            return job.create()

    def assert_payload(self, job, expected_env_var=None, expected_arguments=None):
        """Checks the payload for OCI data science job.

        Parameters
        ----------
        job : Job
            An ADS job instance containing data science job infrastructure.
        expected_env_var : dict, optional
            Expected environment variables, by default None
        expected_arguments : str, optional
            Expected command line arguments, by default None
        """
        expected_payload = copy.deepcopy(self.PAYLOAD_TEMPLATE)
        # Do not check job names here as they are randomly generated
        expected_payload["displayName"] = job.name
        if expected_env_var:
            expected_payload["jobConfigurationDetails"][
                "environmentVariables"
            ] = expected_env_var
        if expected_arguments:
            expected_payload["jobConfigurationDetails"][
                "commandLineArguments"
            ] = expected_arguments
        actual_payload = job.infrastructure.dsc_job.to_dict()
        self.assertEqual(actual_payload, expected_payload)

    def assert_runtime_translation(
        self,
        runtime,
        expected_env_var=None,
        expected_arguments=None,
        assert_extraction=True,
    ):
        """Checks the runtime translation and extraction by mocking the API calls to create a data science job.
        This method also checks the job serialization by converting the job to a string and load it back.

        Parameters
        ----------
        runtime : Runtime
            Runtime of a job
        expected_env_var : dict, optional
            Expected environment variables, by default None
        expected_arguments : str, optional
            Expected command line arguments, by default None
        """
        job = (
            Job(name=self.__class__.__name__)
            .with_infrastructure(
                infrastructure.DataScienceJob()
                .with_compartment_id("ocid1.compartment.oc1..<unique_ocid>")
                .with_project_id("ocid1.datascienceproject.oc1.iad.<unique_ocid>")
            )
            .with_runtime(runtime)
        )
        self.mock_create_job(job)
        self.assert_payload(job, expected_env_var, expected_arguments)
        # Serialize to string and de-serialize
        job_from_string = Job.from_string(str(job))
        # Test the payload again
        self.mock_create_job(job_from_string)
        self.assert_payload(job, expected_env_var, expected_arguments)
        if assert_extraction:
            self.assert_runtime_extraction(job.infrastructure, job.runtime)

    def assert_runtime_extraction(self, data_science_job, expected_runtime):
        """Checks the runtime extraction from a data science job returned by OCI.

        Parameters
        ----------
        data_science_job : DataScienceJob
            A DataScienceJob loaded from OCI, which also the job.infrastructure.
        expected_runtime : Runtime
            An ADS job runtime
        """
        runtime = DataScienceJobRuntimeManager(data_science_job).extract(
            data_science_job.dsc_job
        )
        runtime = runtime.to_dict()
        expected_runtime = expected_runtime.to_dict()
        # Remove fields with local URIs as they could be different.
        # Local paths are not preserved when the artifact is uploaded.
        uri_attrs = ["scriptPathURI", "notebookPathURI"]
        for attr in uri_attrs:
            runtime["spec"].pop(attr, None)
            expected_runtime["spec"].pop(attr, None)

        if runtime["type"] == "container":
            # Container cmd/entrypoint are converted to argument list.
            container_attrs = ["entrypoint", "cmd"]
            for attr in container_attrs:
                if attr in expected_runtime["spec"] and isinstance(
                    expected_runtime["spec"][attr], str
                ):
                    expected_runtime["spec"][attr] = ContainerRuntimeHandler.split_args(
                        expected_runtime["spec"][attr]
                    )
        self.assertEqual(runtime, expected_runtime, f"\n{runtime}\n{expected_runtime}")


class DriverRunTest(unittest.TestCase):
    DRIVER_PATH = os.path.join(
        "ads/jobs/templates", NotebookArtifact.CONST_DRIVER_SCRIPT
    )

    def run_driver(
        self, driver_path, env_vars=None, args=None, suppress_error=False
    ) -> list:
        """Runs the driver script

        Parameters
        ----------
        driver_path : str
            Path of the driver script.
        env_vars : dict, optional
            Environment variables for running the driver, by default None
        args : list, optional
            Command line arguments, by default None
        suppress_error : bool
            Whether to suppress the exception when there is an error running the driver.
            When the is an error running the notebook:
                If this is set to False, an exception will be raised and no output will be returned.
                If this is set to True, no exception will be raised and the outputs will be returned.

        Returns
        -------
        list
            output messages.
        """
        if not args:
            args = []

        cmd = [
            os.path.join(os.path.dirname(sys.executable), "coverage"),
            "run",
            "--concurrency=multiprocessing",
            driver_path,
        ] + args
        lines = []
        try:
            outputs = subprocess.check_output(
                cmd, env=env_vars, stderr=subprocess.STDOUT
            )
            lines += outputs.decode().split("\n")
        except subprocess.CalledProcessError as ex:
            if ex.stdout:
                lines += ex.stdout.decode().split("\n")
            if ex.stderr:
                lines += ex.stderr.decode().split("\n")
            if not suppress_error:
                self.fail(f"Error occurred when running {driver_path}")
        finally:
            # print out log message for debugging purpose
            for line in lines:
                print(line)

        return lines


class DataScienceJobRuntimeTest(DataScienceJobPayloadTest):
    def test_create_runtime_with_dict(self):
        """Test creating a job with a dict as runtime specification.
        A runtime can be created with a dict in two ways:
        1. Pass in the dict directly.
        2. Pass in the kwargs by unpacking the dict.
        """
        runtime_spec = {
            ScriptRuntime.CONST_SCRIPT_PATH: "my_script.py",
            ScriptRuntime.CONST_CONDA: {
                ScriptRuntime.CONST_CONDA_TYPE: "service",
                ScriptRuntime.CONST_CONDA_SLUG: "mlcpuv1",
            },
            ScriptRuntime.CONST_ENV_VAR: {"ENV": "VAR"},
            ScriptRuntime.CONST_ARGS: ["arg", "--key", "val"],
        }
        expected_env_vars = {
            CondaRuntimeHandler.CONST_CONDA_TYPE: "service",
            CondaRuntimeHandler.CONST_CONDA_SLUG: "mlcpuv1",
            "ENV": "VAR",
        }
        expected_cmd_args = "arg --key val"
        runtime = ScriptRuntime(copy.deepcopy(runtime_spec))
        self.assert_runtime_translation(runtime, expected_env_vars, expected_cmd_args)
        runtime = ScriptRuntime(**runtime_spec)
        self.assert_runtime_translation(runtime, expected_env_vars, expected_cmd_args)

    def test_create_env_var_with_non_str(self):
        """Tests when creating a job with non-string environment variables."""
        runtime_spec = {
            ScriptRuntime.CONST_SCRIPT_PATH: "my_script.py",
            ScriptRuntime.CONST_ENV_VAR: {"ENV1": {"key": "val"}, "ENV2": None},
        }
        # ENV1 should be JSON serialized
        # ENV2 has None value and should be converted to empty string.
        expected_env_vars = {"ENV1": '{"key": "val"}', "ENV2": ""}
        runtime = ScriptRuntime(**runtime_spec)
        # For this case the runtime extracted from the OCI model will be
        # different from the one we used to create the job due to env var serialization.
        self.assert_runtime_translation(
            runtime, expected_env_vars, assert_extraction=False
        )
        runtime = ScriptRuntime({ScriptRuntime.CONST_SCRIPT_PATH: "my_script.py"})
        runtime.with_environment_variable(**runtime_spec[ScriptRuntime.CONST_ENV_VAR])
        self.assert_runtime_translation(
            runtime, expected_env_vars, assert_extraction=False
        )

    def test_runtime_with_custom_conda_pack(self):
        expected_env_var = {
            CondaRuntimeHandler.CONST_CONDA_TYPE: "published",
            CondaRuntimeHandler.CONST_CONDA_REGION: "custom_region",
            CondaRuntimeHandler.CONST_CONDA_BUCKET: "bucket",
            CondaRuntimeHandler.CONST_CONDA_NAMESPACE: "namespace",
            CondaRuntimeHandler.CONST_CONDA_OBJ_NAME: "conda_pack/pack_name",
        }
        runtime = (
            ScriptRuntime()
            .with_script("my_script")
            .with_custom_conda(
                "oci://bucket@namespace/conda_pack/pack_name", region="custom_region"
            )
        )
        self.assert_runtime_translation(runtime, expected_env_var=expected_env_var)

    def test_runtime_with_zip_entrypoint(self):
        ZIP_JOB_ENRTYPOINT = "job_archive/main.py"
        DIR_SOURCE_PATH = os.path.join(
            os.path.dirname(__file__), "../../../integration/fixtures/job_archive.zip"
        )

        expected_env_var = {
            ScriptRuntimeHandler.CONST_ENTRYPOINT: ZIP_JOB_ENRTYPOINT,
            CondaRuntimeHandler.CONST_CONDA_TYPE: "service",
            CondaRuntimeHandler.CONST_CONDA_SLUG: "mlcpuv1",
        }

        runtime = (
            ScriptRuntime()
            .with_source(DIR_SOURCE_PATH, entrypoint=ZIP_JOB_ENRTYPOINT)
            .with_service_conda("mlcpuv1")
        )
        self.assert_runtime_translation(runtime, expected_env_var)

    def test_notebook_runtime_with_exclude_tag_as_single_list(self):
        NOTEBOOK_PATH = "path/to/notebook.ipynb"
        OUTPUT_URI = "oci://bucket@namespace/path/to/dir"
        CONDA_SLUG = "tensorflow26_p37_cpu_v1"

        expected_env_var = {
            CondaRuntimeHandler.CONST_CONDA_TYPE: "service",
            CondaRuntimeHandler.CONST_CONDA_SLUG: CONDA_SLUG,
            NotebookRuntimeHandler.CONST_NOTEBOOK_ENCODING: "utf-8",
            NotebookRuntimeHandler.CONST_ENTRYPOINT: NotebookArtifact.CONST_DRIVER_SCRIPT,
            NotebookRuntimeHandler.CONST_NOTEBOOK_NAME: os.path.basename(NOTEBOOK_PATH),
            NotebookRuntimeHandler.CONST_OUTPUT_URI: OUTPUT_URI,
            NotebookRuntimeHandler.CONST_EXCLUDE_TAGS: '["ignore", "remove"]',
        }

        runtime = (
            NotebookRuntime()
            .with_notebook(NOTEBOOK_PATH)
            .with_service_conda(CONDA_SLUG)
            .with_output(OUTPUT_URI)
            .with_exclude_tag(["ignore", "remove"])
        )

        self.assert_runtime_translation(runtime, expected_env_var)

    def test_notebook_runtime_with_exclude_tags_as_string_args(self):
        NOTEBOOK_PATH = "path/to/notebook.ipynb"
        OUTPUT_URI = "oci://bucket@namespace/path/to/dir"
        CONDA_SLUG = "tensorflow26_p37_cpu_v1"

        expected_env_var = {
            CondaRuntimeHandler.CONST_CONDA_TYPE: "service",
            CondaRuntimeHandler.CONST_CONDA_SLUG: CONDA_SLUG,
            NotebookRuntimeHandler.CONST_NOTEBOOK_ENCODING: "utf-8",
            NotebookRuntimeHandler.CONST_ENTRYPOINT: NotebookArtifact.CONST_DRIVER_SCRIPT,
            NotebookRuntimeHandler.CONST_NOTEBOOK_NAME: os.path.basename(NOTEBOOK_PATH),
            NotebookRuntimeHandler.CONST_OUTPUT_URI: OUTPUT_URI,
            NotebookRuntimeHandler.CONST_EXCLUDE_TAGS: '["ignore", "remove"]',
        }

        runtime = (
            NotebookRuntime()
            .with_notebook(NOTEBOOK_PATH)
            .with_service_conda(CONDA_SLUG)
            .with_output(OUTPUT_URI)
            .with_exclude_tag("ignore", "remove")
        )

        self.assert_runtime_translation(runtime, expected_env_var)

    def test_notebook_runtime_with_alternative_output_uri(self):
        NOTEBOOK_PATH = "path/to/notebook.ipynb"
        OUTPUT_URI = "oci://bucket@namespace/path/to/dir"
        CONDA_SLUG = "tensorflow26_p37_cpu_v1"

        expected_env_var = {
            CondaRuntimeHandler.CONST_CONDA_TYPE: "service",
            CondaRuntimeHandler.CONST_CONDA_SLUG: CONDA_SLUG,
            NotebookRuntimeHandler.CONST_ENTRYPOINT: NotebookArtifact.CONST_DRIVER_SCRIPT,
            NotebookRuntimeHandler.CONST_NOTEBOOK_NAME: os.path.basename(NOTEBOOK_PATH),
            NotebookRuntimeHandler.CONST_OUTPUT_URI: OUTPUT_URI,
        }

        runtime = NotebookRuntime(
            {
                NotebookRuntime.CONST_NOTEBOOK_PATH: NOTEBOOK_PATH,
                NotebookRuntime.CONST_CONDA: {
                    NotebookRuntime.CONST_CONDA_TYPE: "service",
                    NotebookRuntime.CONST_CONDA_SLUG: CONDA_SLUG,
                },
                NotebookRuntime.CONST_OUTPUT_URI_ALT: OUTPUT_URI,
            }
        )
        # For this case, the runtime extracted from the OCI model will
        # have the key outputUri instead of outputURI.
        self.assert_runtime_translation(
            runtime, expected_env_var, assert_extraction=False
        )

    def test_notebook_runtime_with_folder(self):
        runtime = (
            NotebookRuntime()
            .with_source("source", notebook="relative/path/to/notebook.ipynb")
            .with_service_conda("slug")
        )
        expected_env_var = {
            CondaRuntimeHandler.CONST_CONDA_TYPE: "service",
            CondaRuntimeHandler.CONST_CONDA_SLUG: "slug",
            NotebookRuntimeHandler.CONST_ENTRYPOINT: NotebookArtifact.CONST_DRIVER_SCRIPT,
            NotebookRuntimeHandler.CONST_NOTEBOOK_NAME: "source/relative/path/to/notebook.ipynb",
            NotebookRuntimeHandler.CONST_NOTEBOOK_ENCODING: "utf-8",
        }
        self.assert_runtime_translation(
            runtime, expected_env_var, assert_extraction=True
        )


class DataScienceJobCreationErrorTest(DataScienceJobPayloadTest):
    existing_compartment_ocid = None
    existing_nb_session_ocid = None

    def setUp(self) -> None:
        self.existing_compartment_ocid = os.environ.pop(
            "NB_SESSION_COMPARTMENT_OCID", None
        )
        self.existing_nb_session_ocid = os.environ.pop("NB_SESSION_OCID", None)
        return super().setUp()

    def tearDown(self) -> None:
        if isinstance(self.existing_compartment_ocid, str):
            os.environ["NB_SESSION_COMPARTMENT_OCID"] = self.existing_compartment_ocid
        if isinstance(self.existing_nb_session_ocid, str):
            os.environ["NB_SESSION_OCID"] = self.existing_nb_session_ocid
        return super().tearDown()

    def test_create_job_without_compartment_id(self):
        job = (
            Job(name="test")
            .with_infrastructure(infrastructure.DataScienceJob())
            .with_runtime(ScriptRuntime().with_script(self.SCRIPT_URI))
        )
        with self.assertRaises(ValueError):
            with mock.patch(
                "ads.jobs.builders.infrastructure.dsc_job.DSCJob._create_with_oci_api"
            ):
                job.create()

    def test_create_job_with_invalid_runtime_args(self):
        with self.assertRaises(ValueError):
            ScriptRuntime().with_script(self.SCRIPT_URI).with_argument(
                **{"a key": "val"}
            )

    def test_run_job_without_create(self):
        job = (
            Job(name="test")
            .with_infrastructure(
                infrastructure.DataScienceJob()
                .with_log_group_id("ocid1.loggroup.oc1.iad.<unique_ocid>")
                .with_log_id("ocid1.log.oc1.iad.<unique_ocid>")
                .with_compartment_id("ocid1.compartment.oc1..<unique_ocid>")
                .with_project_id("ocid1.datascienceproject.oc1.iad.<unique_ocid>")
                .with_shape_name("VM.Standard2.1")
                .with_subnet_id("ocid1.subnet.oc1.iad.<unique_ocid>")
                .with_block_storage_size(50)
            )
            .with_runtime(ScriptRuntime().with_script(self.SCRIPT_URI))
        )
        with self.assertRaises(RuntimeError):
            job.run()

    def test_create_job_with_invalid_notebook_env(self):
        os.environ["NB_SESSION_OCID"] = "invalid"
        job = (
            Job(name="test")
            .with_infrastructure(
                infrastructure.DataScienceJob()
                .with_log_group_id("ocid1.loggroup.oc1.iad.<unique_ocid>")
                .with_log_id("ocid1.log.oc1.iad.<unique_ocid>")
                .with_compartment_id("ocid1.compartment.oc1..<unique_ocid>")
                .with_project_id("ocid1.datascienceproject.oc1.iad.<unique_ocid>")
                .with_shape_name("VM.Standard2.1")
                .with_subnet_id("ocid1.subnet.oc1.iad.<unique_ocid>")
                .with_block_storage_size(50)
            )
            .with_runtime(ScriptRuntime().with_script(self.SCRIPT_URI))
        )
        with mock.patch(
            "ads.jobs.builders.infrastructure.dsc_job.DSCJob._create_with_oci_api"
        ):
            job.create()
        del os.environ["NB_SESSION_OCID"]

    def test_job_with_non_flex_shape_and_shape_details(self):
        job = (
            Job(name="test")
            .with_infrastructure(
                infrastructure.DataScienceJob()
                .with_compartment_id("ocid1.compartment.oc1..<unique_ocid>")
                .with_project_id("ocid1.datascienceproject.oc1.iad.<unique_ocid>")
                .with_shape_name("VM.Standard2.1")
                .with_shape_config_details(memory_in_gbs=16, ocpus=1)
                .with_block_storage_size(50)
            )
            .with_runtime(ScriptRuntime().with_script(self.SCRIPT_URI))
        )

        with pytest.raises(
            ValueError,
            match="Shape config is not required for non flex shape from user end."
        ):
            job.create()


class ScriptRuntimeArtifactTest(unittest.TestCase):
    DIR_SOURCE_PATH = os.path.join(os.path.dirname(__file__), "test_files/job_archive")
    SCRIPT_SOURCE_PATH = os.path.join(
        os.path.dirname(__file__), "test_files/job_archive/main.py"
    )

    def test_prepare_artifact_with_dir(self):
        """Tests preparing a directory as job artifact."""
        with ScriptArtifact(self.DIR_SOURCE_PATH) as artifact:
            with ZipFile(artifact.path, "r") as zip_file:
                files = zip_file.namelist()
                files = [f for f in files if "__pycache__" not in f]
                files.sort()
                expected_files = [
                    "job_archive/",
                    "job_archive/my_package/",
                    "job_archive/my_module.py",
                    "job_archive/script.sh",
                    "job_archive/main.py",
                    "job_archive/my_package/__init__.py",
                    "job_archive/my_package/entrypoint.py",
                    "job_archive/my_package/entrypoint_ads.py",
                    "job_archive/my_package/utils.py",
                ]
                expected_files.sort()

                self.assertEqual(files, expected_files)

    def test_prepare_artifact_with_script(self):
        """Tests preparing a python script as job artifact."""
        with ScriptArtifact(self.SCRIPT_SOURCE_PATH) as artifact:
            self.assertEqual(
                os.path.basename(artifact.path),
                os.path.basename(self.SCRIPT_SOURCE_PATH),
            )


class DataScienceJobNameSubstitutionTest(DataScienceJobPayloadTest):
    @mock.patch("ads.jobs.builders.infrastructure.dsc_job.DSCJob.run")
    def test_substitute_job_name(self, mocked_run):
        envs = dict(A="foo", B="bar")
        test_cases = [
            ("my_$B", "my_bar"),
            ("my_$", "my_$"),
            ("my_$C", "my_$C"),
            ("my_$$$A", "my_$foo"),
        ]
        for input_name, expected_name in test_cases:
            job = (
                Job(name=input_name)
                .with_infrastructure(
                    infrastructure.DataScienceJob()
                    .with_compartment_id("ocid1.compartment.oc1..<unique_ocid>")
                    .with_project_id("ocid1.datascienceproject.oc1.iad.<unique_ocid>")
                )
                .with_runtime(
                    ScriptRuntime()
                    .with_source("my_script.py")
                    .with_environment_variable(**envs)
                )
            )
            job = self.mock_create_job(job)
            self.assertEqual(
                job.name,
                expected_name,
                f"Job Name: Expect {input_name} to be converted to {expected_name}.",
            )
            job.run(name=input_name)
            _, kwargs = mocked_run.call_args
            self.assertEqual(
                kwargs["display_name"],
                expected_name,
                f"Job Run Name: Expect {input_name} to be converted to {expected_name}.",
            )


class TestRunInstance:
    """Tests for the `ads.jobs.builders.infrastructure.base.RunInstance` class."""

    @pytest.mark.parametrize(
        "test_template, expected_result",
        [
            ("https://{region}/details/{id}", "https://test_region/details/test_id"),
            ("https://{region}/details/", "https://test_region/details/"),
            ("https://details/{id}", "https://details/test_id"),
            ("", ""),
        ],
    )
    @patch("ads.common.utils.extract_region", return_value="test_region")
    def test_run_details_link(
        self, mock_extract_region, test_template, expected_result
    ):
        """Ensures that details link can be generated based on the provided template."""
        test_run_instance = RunInstance()
        test_run_instance.id = "test_id"

        test_run_instance._DETAILS_LINK = test_template

        test_result = test_run_instance.run_details_link
        if test_template:
            mock_extract_region.assert_called()
        assert test_result == expected_result

    @patch("ads.common.utils.extract_region", side_effect=Exception("some error"))
    def test_run_details_link_fail(self, mock_extract_region):
        """Ensures that details link returns an empty string in case of any errors in the method."""
        test_run_instance = RunInstance()
        test_result = test_run_instance.run_details_link
        assert test_result == ""
