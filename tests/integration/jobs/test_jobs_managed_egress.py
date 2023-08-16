#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import os
from unittest import mock
from ads.jobs import Job, ScriptRuntime
from tests.integration.config import secrets
from tests.integration.jobs.test_dsc_job import DSCJobTestCaseWithCleanUp


# This notebook is configured with default networking (managed egress)
NOTEBOOK_WITH_ME = secrets.jobs.NOTEBOOK_WITH_ME
# This notebook is using a subnet
NOTEBOOK_WITH_SUBNET = secrets.jobs.NOTEBOOK_ID


class DSCJobManagedEgressTestCase(DSCJobTestCaseWithCleanUp):
    @mock.patch.dict(os.environ, NB_SESSION_OCID=NOTEBOOK_WITH_ME)
    def test_create_managed_egress_job_within_managed_egress_nb_session(self):
        """Tests creating a job using default configurations from notebook with managed egress."""
        expected_infra_spec = {
            "displayName": "my_script",
            "compartmentId": self.COMPARTMENT_ID,
            "jobType": "DEFAULT",
            "jobInfrastructureType": "STANDALONE",
            "shapeName": "VM.Standard.E3.Flex",
            "shapeConfigDetails": {"memoryInGBs": 16.0, "ocpus": 1.0},
            "blockStorageSize": 100,
            "projectId": self.PROJECT_ID,
            "subnetId": self.SUBNET_ID,
        }

        expected_runtime_spec = copy.deepcopy(self.DEFAULT_RUNTIME_SPEC)

        # Create a job
        job = (
            Job()
            .with_infrastructure(self.default_datascience_job)
            .with_runtime(ScriptRuntime().with_script(self.SCRIPT_URI))
            .create()
        )

        self.assert_job_creation(job, expected_infra_spec, expected_runtime_spec)

    @mock.patch.dict(os.environ, NB_SESSION_OCID=NOTEBOOK_WITH_SUBNET)
    def test_create_managed_egress_job_within_nb_session_using_subnet(self):
        """Tests creating a job using managed egress from notebook with a subnet."""
        expected_infra_spec = {
            "displayName": "my_script",
            "compartmentId": self.COMPARTMENT_ID,
            "jobType": "DEFAULT",
            "jobInfrastructureType": "STANDALONE",
            "shapeName": "VM.Standard.E3.Flex",
            "shapeConfigDetails": {"memoryInGBs": 16.0, "ocpus": 1.0},
            "blockStorageSize": 100,
            "projectId": self.PROJECT_ID,
            "subnetId": self.SUBNET_ID,
        }

        expected_runtime_spec = copy.deepcopy(self.DEFAULT_RUNTIME_SPEC)

        # Create a job
        job = (
            Job()
            .with_infrastructure(
                self.default_datascience_job.with_job_infrastructure_type(
                    "ME_STANDALONE"
                )
            )
            .with_runtime(ScriptRuntime().with_script(self.SCRIPT_URI))
            .create()
        )

        self.assert_job_creation(job, expected_infra_spec, expected_runtime_spec)

    @mock.patch.dict(os.environ, NB_SESSION_OCID=NOTEBOOK_WITH_SUBNET)
    def test_create_job_using_same_subnet_within_nb_session_using_subnet(self):
        """Tests creating a job using managed egress from notebook with a subnet."""
        expected_infra_spec = {
            "displayName": "my_script",
            "compartmentId": self.COMPARTMENT_ID,
            "jobType": "DEFAULT",
            "jobInfrastructureType": "STANDALONE",
            "shapeName": "VM.Standard.E3.Flex",
            "shapeConfigDetails": {"memoryInGBs": 16, "ocpus": 1},
            "blockStorageSize": 100,
            "projectId": self.PROJECT_ID,
            "subnetId": self.SUBNET_ID,
        }

        expected_runtime_spec = copy.deepcopy(self.DEFAULT_RUNTIME_SPEC)

        # Create a job
        job = (
            Job()
            .with_infrastructure(self.default_datascience_job)
            .with_runtime(ScriptRuntime().with_script(self.SCRIPT_URI))
            .create()
        )

        self.assert_job_creation(job, expected_infra_spec, expected_runtime_spec)

    @mock.patch.dict(os.environ, NB_SESSION_OCID=NOTEBOOK_WITH_SUBNET)
    def test_create_job_using_different_subnet_within_nb_session_using_subnet(self):
        """Tests creating a job using managed egress from notebook with a subnet."""
        expected_infra_spec = {
            "displayName": "my_script",
            "compartmentId": self.COMPARTMENT_ID,
            "jobType": "DEFAULT",
            "jobInfrastructureType": "STANDALONE",
            "shapeName": "VM.Standard.E3.Flex",
            "shapeConfigDetails": {"memoryInGBs": 16, "ocpus": 1},
            "blockStorageSize": 100,
            "projectId": self.PROJECT_ID,
            "subnetId": secrets.jobs.SUBNET_ID_DIFF,
        }

        expected_runtime_spec = copy.deepcopy(self.DEFAULT_RUNTIME_SPEC)

        # Create a job
        job = (
            Job()
            .with_infrastructure(
                self.default_datascience_job.with_subnet_id(secrets.jobs.SUBNET_ID_DIFF)
            )
            .with_runtime(ScriptRuntime().with_script(self.SCRIPT_URI))
            .create()
        )

        self.assert_job_creation(job, expected_infra_spec, expected_runtime_spec)
