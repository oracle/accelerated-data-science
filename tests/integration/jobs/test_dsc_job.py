#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import datetime
import importlib
import json
import logging
import os
import random
import string
import time
import traceback
import threading
import unittest
from unittest import mock

import oci
import yaml
from ads import config
from ads.common.auth import default_signer
from ads.jobs import (
    Job,
    DataScienceJob,
    ScriptRuntime,
)
from ads.jobs.builders.runtimes.base import Runtime
from tests.integration.config import secrets


# logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


class DSCJobTestCase(unittest.TestCase):
    """Base class for Data Science Job integration tests."""

    INT_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPT_URI = os.path.join(INT_TEST_DIR, "../fixtures", "my_script.py")

    TENANCY_ID = secrets.common.TENANCY_ID
    COMPARTMENT_ID = secrets.common.COMPARTMENT_ID
    # This is an inactive notebook that will never be deleted
    NOTEBOOK_ID = secrets.jobs.NOTEBOOK_ID
    # This is the compartment for creating the project only.
    # Jobs are not created in this compartment.
    PROJECT_COMPARTMENT_ID = secrets.jobs.PROJECT_COMPARTMENT_ID
    PROJECT_ID = ""
    LOG_GROUP_ID = secrets.jobs.LOG_GROUP_ID
    LOG_ID = secrets.jobs.LOG_ID
    SUBNET_ID = secrets.jobs.SUBNET_ID

    DEFAULT_INFRA_SPEC = {
        "compartmentId": COMPARTMENT_ID,
        "jobType": "DEFAULT",
        "jobInfrastructureType": "STANDALONE",
        "shapeName": "VM.Standard.E3.Flex",
        "shapeConfigDetails": {"memoryInGBs": 16, "ocpus": 1},
        "blockStorageSize": 100,
        "subnetId": SUBNET_ID,
    }

    DEFAULT_RUNTIME_SPEC = {ScriptRuntime.CONST_SCRIPT_PATH: SCRIPT_URI}

    BUCKET = secrets.jobs.BUCKET_A
    NAMESPACE = secrets.common.NAMESPACE

    EXISTING_ENV = {}

    random_seed = 42

    @property
    def default_datascience_job(self):
        random.seed(self.random_seed)
        return DataScienceJob().with_project_id(self.PROJECT_ID)

    @classmethod
    def setUpClass(cls) -> None:
        """Sets the environment variables to simulate a notebook environment."""
        # Simulate the testing in a notebook environment
        cls.EXISTING_ENV = os.environ.copy()
        os.environ["NB_SESSION_COMPARTMENT_OCID"] = cls.COMPARTMENT_ID
        os.environ["NB_SESSION_OCID"] = cls.NOTEBOOK_ID
        project = (
            oci.data_science.DataScienceClient(**default_signer())
            .create_project(
                {
                    "displayName": "int-test-ads-"
                    + datetime.datetime.now().strftime("%Y%m%d-%H%M"),
                    "description": "ADS Integration Test",
                    "compartmentId": cls.PROJECT_COMPARTMENT_ID,
                }
            )
            .data
        )
        importlib.reload(config)
        cls.PROJECT_ID = project.id
        print(f"Project OCID: {cls.PROJECT_ID}")
        cls.DEFAULT_INFRA_SPEC["projectId"] = cls.PROJECT_ID
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the environment variables."""
        if cls.PROJECT_ID:
            # Wait for the jobs to be deleted
            time.sleep(10)
            oci.data_science.DataScienceClient(**default_signer()).delete_project(
                project_id=cls.PROJECT_ID
            )

        # When the tests are finished, some job runs might be in transient states.
        # The tests might not be able to delete all the jobs due to unexpected reason.
        # Here we will try to clean up the projects again.
        # If the test project that are created more than 36 hours ago,
        # we assume the tests have been finished.
        # All tests should be run in the projects in the test_ads_jobs compartment.
        projects = (
            oci.data_science.DataScienceClient(**default_signer())
            .list_projects(
                compartment_id=cls.PROJECT_COMPARTMENT_ID,
                lifecycle_state="ACTIVE",
            )
            .data
        )
        time_to_delete = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(hours=36)

        projects = [
            project
            for project in projects
            if project.description == "ADS Integration Test"
            and project.time_created < time_to_delete
        ]
        for project in projects:
            cls.clean_up_project(project.id)

        # Unset the env vars
        for env in ["NB_SESSION_COMPARTMENT_OCID", "NB_SESSION_OCID"]:
            if env in cls.EXISTING_ENV:
                os.environ[env] = cls.EXISTING_ENV[env]
            else:
                os.environ.pop(env, None)
        importlib.reload(config)
        return super().tearDownClass()

    @classmethod
    def clean_up_project(cls, project_ocid):
        jobs = Job.datascience_job(
            compartment_id=cls.COMPARTMENT_ID,
            project_id=project_ocid,
            lifecycle_state="ACTIVE",
        )

        if jobs:
            for job in jobs:
                job.delete()
            time.sleep(20)

        oci.data_science.DataScienceClient(**default_signer()).delete_project(
            project_id=project_ocid
        )

    def assert_job_spec(self, job, expected_infra, expected_runtime):
        self.assertIsInstance(job, Job)
        # Convert the job object as a string and load it as dict
        job_dict = yaml.safe_load(str(job))
        self.assertEqual(job_dict.get("kind"), "job")
        job_spec = job_dict.get("spec")
        # Check the expected display name, the job name may have a timestamp appended.
        # Script name should be in job name.
        self.assertIn(expected_infra["displayName"], job_spec.get("name"))
        self.assertEqual(job_spec.get("infrastructure").get("spec"), expected_infra)

        actual_runtime_spec = job_spec.get("runtime").get("spec")
        # Check script path for script runtime
        # It is not possible to get the full script path back from OCI infra
        if ScriptRuntime.CONST_SCRIPT_PATH in expected_runtime:
            actual_script_uri = actual_runtime_spec.pop(
                ScriptRuntime.CONST_SCRIPT_PATH, None
            )
            self.assertIsNotNone(actual_script_uri)
            expected_runtime = copy.deepcopy(expected_runtime)
            expected_script_uri = expected_runtime.pop(ScriptRuntime.CONST_SCRIPT_PATH)
            self.assertIn(
                actual_script_uri,
                [expected_script_uri, os.path.basename(expected_script_uri)],
            )
        # Check the runtime spec
        self.assertEqual(actual_runtime_spec, expected_runtime)

    def assert_job_creation(self, job, expected_infra_spec, expected_runtime_spec):
        """Asserts the job creation by comparing the specifications of infrastructure and runtime.
        The assertion checks the specifications for:
        1. The job object passing into this function.
        2. Serializing and Deserializing the job to a YAML file on OCI object storage.
        3. Loading the job from OCI using OCID.
        It will also check if the job is listed as ACTIVE when calling the list API.
        """
        # Set maxDiff to show all differences
        self.maxDiff = None
        # The expected display name should be a sub-string of the actual name
        # since a timestamp may be added to the job.
        self.assertIn(expected_infra_spec["displayName"], job.name)
        # Update the job name in infra spec for comparison purpose.
        expected_infra_spec["displayName"] = job.name
        self.assert_job_spec(job, expected_infra_spec, expected_runtime_spec)
        # Job ID should not be none once created
        self.assertIsNotNone(job.id)
        # Save the job ID
        job_id = job.id

        # Test serialization and de-serialization
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        random.seed(threading.get_ident() + os.getpid())
        random_suffix = "".join(random.choices(string.ascii_uppercase, k=6))
        yaml_uri = f"oci://{self.BUCKET}@{self.NAMESPACE}/tests/{timestamp}/example_job_{random_suffix}.yaml"
        config_path = "~/.oci/config"
        job.to_yaml(uri=yaml_uri, config=config_path)
        print(f"Job YAML saved to {yaml_uri}")
        try:
            job = Job.from_yaml(uri=yaml_uri, config=config_path)
        except Exception:
            self.fail(f"Failed to load job from YAML\n{traceback.format_exc()}")

        # Retrieve the job from infrastructure
        job = Job.from_datascience_job(job_id)
        self.assertIsNotNone(job.infrastructure.dsc_job.artifact)
        self.assert_job_spec(job, expected_infra_spec, expected_runtime_spec)

        # List all active jobs in the project
        jobs = Job.datascience_job(
            project_id=self.PROJECT_ID,
            compartment_id=job.infrastructure.compartment_id,
            lifecycle_state="ACTIVE",
        )
        job_id_list = [j.id for j in jobs]
        self.assertIn(job.id, job_id_list)


class DSCJobTestCaseWithCleanUp(DSCJobTestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        # Delete all jobs once tests are completed.
        cls.delete_jobs_in_project()
        return super().tearDownClass()

    @classmethod
    def delete_jobs_in_project(cls, **kwargs):
        """Deletes all jobs and the corresponding runs in the project."""
        jobs = Job.datascience_job(
            compartment_id=cls.COMPARTMENT_ID,
            project_id=cls.PROJECT_ID,
            lifecycle_state="ACTIVE",
            **kwargs,
        )
        for job in jobs:
            try:
                job.delete()
            except oci.exceptions.ServiceError as ex:
                logger.error("Job ID: %s, %s", job.id, str(ex))


class DSCJobCreationTestCase(DSCJobTestCaseWithCleanUp):
    def test_create_job_with_default_config(self):
        """Tests creating a job with default infrastructure configuration from notebook."""
        expected_infra_spec = copy.deepcopy(self.DEFAULT_INFRA_SPEC)
        # expect display name include name of artifact
        expected_infra_spec["displayName"] = "my_script"
        expected_runtime_spec = copy.deepcopy(self.DEFAULT_RUNTIME_SPEC)

        # Create a job
        job = (
            Job()
            .with_infrastructure(self.default_datascience_job)
            .with_runtime(ScriptRuntime().with_script(DSCJobTestCase.SCRIPT_URI))
            .create()
        )

        self.assert_job_creation(job, expected_infra_spec, expected_runtime_spec)

    def test_create_job_with_dsc_infra_config(self):
        """Tests creating a job with various combinations of infrastructure configurations."""
        configs = {
            "shape_name": "VM.Standard.E3.Flex",
            "block_storage_size": 100,
            "project_id": self.PROJECT_ID,
            "shape_config_details": {"memory_in_gbs": 16, "ocpus": 1},
        }
        attr_map = {
            "shape_name": "shapeName",
            "block_storage_size": "blockStorageSize",
            "project_id": "projectId",
            "shape_config_details": "shapeConfigDetails",
            "memory_in_gbs": "memoryInGBs",
            "ocpus": "ocpus",
        }

        expected_runtime_spec = copy.deepcopy(self.DEFAULT_RUNTIME_SPEC)
        # Single infra config
        for k, v in configs.items():
            expected_infra_spec = copy.deepcopy(self.DEFAULT_INFRA_SPEC)
            # expect display name include name of artifact
            expected_infra_spec["displayName"] = "my_script"

            infra = self.default_datascience_job
            builder_method = getattr(infra, f"with_{k}")

            if isinstance(v, dict):
                expected_infra_spec[attr_map[k]] = {}
                for sk, sv in v.items():
                    expected_infra_spec[attr_map[k]][attr_map[sk]] = sv
                infra = builder_method(**v)
            else:
                expected_infra_spec[attr_map[k]] = v
                infra = builder_method(v)

            self.assertIsInstance(infra, DataScienceJob)

            job = (
                Job()
                .with_infrastructure(infra)
                .with_runtime(ScriptRuntime().with_script(DSCJobTestCase.SCRIPT_URI))
                .create()
            )

            self.assert_job_creation(job, expected_infra_spec, expected_runtime_spec)

        # All infra config
        expected_infra_spec = copy.deepcopy(self.DEFAULT_INFRA_SPEC)
        # expect display name include name of artifact
        expected_infra_spec["displayName"] = "my_script"
        infra = self.default_datascience_job

        for k, v in configs.items():
            builder_method = getattr(infra, f"with_{k}")
            if isinstance(v, dict):
                expected_infra_spec[attr_map[k]] = {}
                for sk, sv in v.items():
                    expected_infra_spec[attr_map[k]][attr_map[sk]] = sv
                infra = builder_method(**v)
            else:
                expected_infra_spec[attr_map[k]] = v
                infra = builder_method(v)

            self.assertIsInstance(infra, DataScienceJob)

        job = (
            Job()
            .with_infrastructure(infra)
            .with_runtime(ScriptRuntime().with_script(DSCJobTestCase.SCRIPT_URI))
            .create()
        )

        self.assert_job_creation(job, expected_infra_spec, expected_runtime_spec)

    def assert_runtime_test_cases(self, test_cases, spec_key):
        for func in test_cases:
            expected_infra_spec = copy.deepcopy(self.DEFAULT_INFRA_SPEC)
            # expect display name include name of artifact
            expected_infra_spec["displayName"] = "my_script"
            expected_runtime_spec = copy.deepcopy(self.DEFAULT_RUNTIME_SPEC)
            runtime = ScriptRuntime().with_script(DSCJobTestCase.SCRIPT_URI)
            expected_runtime_spec[spec_key] = func(runtime)
            job = (
                Job()
                .with_infrastructure(self.default_datascience_job)
                .with_runtime(runtime)
                .create()
            )
            self.assert_job_creation(job, expected_infra_spec, expected_runtime_spec)

    @staticmethod
    def add_one_runtime_arg(runtime: ScriptRuntime):
        runtime.with_argument("pos1")
        spec_args = ["pos1"]
        return spec_args

    @staticmethod
    def add_two_runtime_args(runtime: ScriptRuntime):
        runtime.with_argument("pos1", "pos2")
        spec_args = ["pos1", "pos2"]
        return spec_args

    @staticmethod
    def add_one_runtime_kwarg(runtime: ScriptRuntime):
        runtime.with_argument(key1="val1")
        spec_args = ["--key1", "val1"]
        return spec_args

    @staticmethod
    def add_two_runtime_kwargs(runtime: ScriptRuntime):
        runtime.with_argument(key1="val1", key2="val2")
        spec_args = ["--key1", "val1", "--key2", "val2"]
        return spec_args

    @staticmethod
    def add_one_runtime_arg_and_two_runtime_kwargs(runtime: ScriptRuntime):
        runtime.with_argument("pos1", key1="val1", key2="val2")
        spec_args = ["pos1", "--key1", "val1", "--key2", "val2"]
        return spec_args

    @staticmethod
    def add_two_runtime_kwargs_and_one_runtime_arg(runtime: ScriptRuntime):
        runtime.with_argument(key1="val1", key2="val2")
        runtime.with_argument("pos1")
        spec_args = ["--key1", "val1", "--key2", "val2", "pos1"]
        return spec_args

    @staticmethod
    def add_mixed_runtime_args_and_kwargs(runtime: ScriptRuntime):
        runtime.with_argument("pos1")
        runtime.with_argument(key1="val1", key2="val2.1 val2.2")
        runtime.with_argument("pos2")
        spec_args = ["pos1", "--key1", "val1", "--key2", "val2.1 val2.2", "pos2"]
        return spec_args

    @staticmethod
    def add_runtime_kwargs_with_none(runtime: ScriptRuntime):
        runtime.with_argument("pos1")
        runtime.with_argument(key1=None, key2="val2")
        runtime.with_argument("pos2")
        spec_args = ["pos1", "--key1", "--key2", "val2", "pos2"]
        return spec_args

    def test_create_job_with_python_runtime_args(self):
        test_cases = [
            self.add_one_runtime_arg,
            self.add_two_runtime_args,
            self.add_one_runtime_kwarg,
            self.add_two_runtime_kwargs,
            self.add_one_runtime_arg_and_two_runtime_kwargs,
            self.add_two_runtime_kwargs_and_one_runtime_arg,
            self.add_mixed_runtime_args_and_kwargs,
            self.add_runtime_kwargs_with_none,
        ]
        self.assert_runtime_test_cases(test_cases, "args")

    @staticmethod
    def add_one_runtime_env(runtime: ScriptRuntime):
        runtime.with_environment_variable(key1="val1")
        spec_env = [
            {"name": "key1", "value": "val1"},
        ]
        return spec_env

    @staticmethod
    def add_two_runtime_envs(runtime: ScriptRuntime):
        runtime.with_environment_variable(key1="val1", key2="val2")
        spec_env = [
            {"name": "key1", "value": "val1"},
            {"name": "key2", "value": "val2"},
        ]
        return spec_env

    @staticmethod
    def add_runtime_env_twice(runtime: ScriptRuntime):
        runtime.with_environment_variable(key3="val3").with_environment_variable(
            key1="val1", key2="val2"
        )
        spec_env = [
            {"name": "key1", "value": "val1"},
            {"name": "key2", "value": "val2"},
        ]
        return spec_env

    def test_create_job_with_python_runtime_env(self):
        test_cases = [
            self.add_one_runtime_env,
            self.add_two_runtime_envs,
            self.add_runtime_env_twice,
        ]
        self.assert_runtime_test_cases(test_cases, Runtime.CONST_ENV_VAR)

    @staticmethod
    def add_runtime_tags(runtime: ScriptRuntime):
        runtime.with_freeform_tag(key3="val3").with_freeform_tag(
            key1="val1", key2="val2"
        )
        spec_tag = {"key1": "val1", "key2": "val2"}
        return spec_tag

    def test_create_job_with_tags(self):
        self.assert_runtime_test_cases(
            [self.add_runtime_tags], Runtime.CONST_FREEFORM_TAGS
        )

    def test_create_job_with_mixed_runtime_config(self):
        expected_infra_spec = copy.deepcopy(self.DEFAULT_INFRA_SPEC)
        # expect display name include name of artifact
        expected_infra_spec["displayName"] = "my_script"
        expected_runtime_spec = copy.deepcopy(self.DEFAULT_RUNTIME_SPEC)

        runtime = (
            ScriptRuntime()
            .with_script(DSCJobTestCase.SCRIPT_URI)
            .with_environment_variable(key1="val1", key2="val2")
            .with_argument("pos", key1="val1")
            .with_freeform_tag(key1="val1")
        )
        expected_runtime_spec[Runtime.CONST_ARGS] = ["pos", "--key1", "val1"]
        expected_runtime_spec[Runtime.CONST_FREEFORM_TAGS] = {"key1": "val1"}
        expected_runtime_spec[Runtime.CONST_ENV_VAR] = [
            {"name": "key1", "value": "val1"},
            {"name": "key2", "value": "val2"},
        ]
        job = (
            Job()
            .with_infrastructure(self.default_datascience_job)
            .with_runtime(runtime)
            .create()
        )
        self.assert_job_creation(job, expected_infra_spec, expected_runtime_spec)

    def test_create_job_with_service_conda(self):
        expected_infra_spec = copy.deepcopy(self.DEFAULT_INFRA_SPEC)
        # expect display name include name of artifact
        expected_infra_spec["displayName"] = "my_script"
        expected_infra_spec["logGroupId"] = self.LOG_GROUP_ID
        expected_infra_spec["logId"] = self.LOG_ID
        expected_runtime_spec = copy.deepcopy(self.DEFAULT_RUNTIME_SPEC)

        runtime = (
            ScriptRuntime()
            .with_script(DSCJobTestCase.SCRIPT_URI)
            .with_environment_variable(key1="val1", key2="val2")
            .with_service_conda("mlcpuv1")
            .with_argument("pos", key1="val1")
            .with_freeform_tag(key1="val1")
        )
        expected_runtime_spec[Runtime.CONST_ARGS] = ["pos", "--key1", "val1"]
        expected_runtime_spec[Runtime.CONST_FREEFORM_TAGS] = {"key1": "val1"}
        expected_runtime_spec[ScriptRuntime.CONST_CONDA] = {
            "type": "service",
            "slug": "mlcpuv1",
        }
        expected_runtime_spec[Runtime.CONST_ENV_VAR] = [
            {"name": "key1", "value": "val1"},
            {"name": "key2", "value": "val2"},
        ]
        job = (
            Job()
            .with_infrastructure(
                # Default shape will be used if with_shape_name(None) is called.
                self.default_datascience_job.with_log_id(self.LOG_ID).with_shape_name(
                    os.environ.get("MY_SHAPE")
                )
            )
            .with_runtime(runtime)
            .create()
        )
        self.assertIsNotNone(job.infrastructure.dsc_job)
        print(job)
        self.assertEqual(
            json.loads(str(job.infrastructure.dsc_job.job_configuration_details)),
            {
                "command_line_arguments": "pos --key1 val1",
                "environment_variables": {
                    "CONDA_ENV_SLUG": "mlcpuv1",
                    "CONDA_ENV_TYPE": "service",
                    "key1": "val1",
                    "key2": "val2",
                },
                # "hyperparameter_values": None,
                "job_type": "DEFAULT",
                "maximum_runtime_in_minutes": None,
            },
        )
        self.assert_job_creation(job, expected_infra_spec, expected_runtime_spec)

    def test_create_job_with_directory(self):
        SOURCE_PATH = os.path.join(
            os.path.dirname(__file__), "../fixtures/job_archive/"
        )
        JOB_ENRTYPOINT = "job_archive/main.py"
        job = (
            Job()
            .with_infrastructure(self.default_datascience_job.with_log_id(self.LOG_ID))
            .with_runtime(
                ScriptRuntime()
                .with_source(SOURCE_PATH, entrypoint=JOB_ENRTYPOINT)
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
                    "JOB_RUN_ENTRYPOINT": JOB_ENRTYPOINT,
                },
                # "hyperparameter_values": None,
                "job_type": "DEFAULT",
                "maximum_runtime_in_minutes": None,
            },
        )


class DSCJobTestCaseWithoutCleanUp(DSCJobTestCase):
    def test_list_shapes(self):
        shapes = DataScienceJob.instance_shapes(self.COMPARTMENT_ID)
        self.assertIsInstance(shapes, list)
        self.assertGreater(len(shapes), 5)
        shape_names = [shape.name for shape in shapes]
        self.assertIn("VM.Standard2.1", shape_names)

    def test_list_fast_launch_shapes(self):
        shapes = DataScienceJob.fast_launch_shapes(self.COMPARTMENT_ID)
        self.assertIsInstance(shapes, list)
        shape_names = [shape.shape_name for shape in shapes]
        self.assertIn("VM.Standard2.1", shape_names)

    def assert_infra_before_build(self, infra):
        print(infra)
        self.assertEqual(
            infra.job_infrastructure_type,
            None,
            "Job infrastructure type should be None before calling build()",
        )
        # Project ID is specified in the self.default_datascience_job
        # This project ID is not the same as the one for the notebook session
        self.assertEqual(infra.project_id, self.PROJECT_ID)
        self.assertEqual(infra.compartment_id, None)
        self.assertEqual(infra.block_storage_size, None)
        self.assertEqual(infra.subnet_id, None)

    def test_build_job_within_notebook(self):
        job = (
            Job()
            .with_infrastructure(self.default_datascience_job)
            .with_runtime(ScriptRuntime().with_script(DSCJobTestCase.SCRIPT_URI))
        )
        self.assert_infra_before_build(job.infrastructure)
        job.build()
        print(job.infrastructure)
        self.assertEqual(job.infrastructure.job_infrastructure_type, "STANDALONE")
        # Project ID should remain the same after build.
        self.assertEqual(job.infrastructure.project_id, self.PROJECT_ID)
        self.assertEqual(job.infrastructure.compartment_id, self.COMPARTMENT_ID)
        # The notebook is using a block storage size of 100GB
        self.assertEqual(job.infrastructure.block_storage_size, 100)
        self.assertEqual(job.infrastructure.subnet_id, self.SUBNET_ID)

    @mock.patch.dict(os.environ)
    def test_build_job_outside_notebook(self):
        os.environ.pop("NB_SESSION_OCID", None)
        job = (
            Job()
            .with_infrastructure(self.default_datascience_job)
            .with_runtime(ScriptRuntime().with_script(DSCJobTestCase.SCRIPT_URI))
        )
        self.assert_infra_before_build(job.infrastructure)
        job.build()
        print(job.infrastructure)
        self.assertEqual(job.infrastructure.job_infrastructure_type, "ME_STANDALONE")
        self.assertEqual(job.infrastructure.project_id, self.PROJECT_ID)
        self.assertEqual(job.infrastructure.compartment_id, self.COMPARTMENT_ID)
        self.assertEqual(job.infrastructure.block_storage_size, 50)
        self.assertEqual(job.infrastructure.subnet_id, None)
