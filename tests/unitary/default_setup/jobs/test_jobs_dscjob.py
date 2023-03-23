#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import logging
import os
import random
from unittest.mock import MagicMock, Mock, patch

import oci
import pytest
from ads.common import utils
from ads.common.oci_datascience import DSCNotebookSession
from ads.common.oci_mixin import OCIModelMixin
from ads.jobs.builders.infrastructure.dsc_job import DSCJob, logger
from oci.data_science.models import (
    Job,
    JobConfigurationDetails,
    JobInfrastructureConfigurationDetails,
)
from oci.response import Response

logger.setLevel(logging.DEBUG)


random_seed = 42


class TestDSCJob:
    def setup_method(self):
        self.payload = dict(
            compartment_id="test_compartment_id",
            display_name="test_display_name",
            job_configuration_details=JobConfigurationDetails(),
            job_infrastructure_configuration_details=JobInfrastructureConfigurationDetails(),
            project_id="test_project_id",
            time_created=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        )

    @property
    def sample_create_job_response(self):
        self.payload[
            "lifecycle_state"
        ] = oci.data_science.models.Job.LIFECYCLE_STATE_ACTIVE
        self.payload["id"] = "ocid1.datasciencejob.oc1.iad.<unique_ocid>"
        return Response(
            data=Job(**self.payload), status=None, headers=None, request=None
        )

    @property
    def sample_create_job_response_with_default_display_name(self):
        self.payload[
            "lifecycle_state"
        ] = oci.data_science.models.Job.LIFECYCLE_STATE_ACTIVE
        random.seed(random_seed)
        self.payload["display_name"] = utils.get_random_name_for_resource()
        return Response(
            data=Job(**self.payload), status=None, headers=None, request=None
        )

    @property
    def sample_create_job_response_with_default_display_name_with_artifact(self):
        self.payload[
            "lifecycle_state"
        ] = oci.data_science.models.Job.LIFECYCLE_STATE_ACTIVE
        random.seed(random_seed)
        self.payload["display_name"] = "my_script_name"
        return Response(
            data=Job(**self.payload), status=None, headers=None, request=None
        )

    @property
    def sample_delete_job_response(self):
        return Response(data=None, status=None, headers=None, request=None)

    @pytest.fixture(scope="function")
    def mock_details(self):
        return MagicMock(return_value=self.payload)

    @pytest.fixture(scope="function")
    def mock_details_with_default_display_name(self):
        random.seed(random_seed)
        self.payload["display_name"] = utils.get_random_name_for_resource()
        return MagicMock(return_value=self.payload)

    @pytest.fixture(scope="function")
    def mock_details_with_default_display_name_with_artifact(self):
        self.payload["display_name"] = "my_script_name"
        return MagicMock(return_value=self.payload)

    @pytest.fixture(scope="function")
    def mock_client(self):
        mock_client = MagicMock()
        mock_client.create_job = Mock(return_value=self.sample_create_job_response)
        mock_client.delete_application = Mock(
            return_value=self.sample_delete_job_response
        )
        return mock_client

    @pytest.fixture(scope="function")
    def mock_client_with_default_display_name(self):
        mock_client = MagicMock()
        mock_client.create_job = Mock(
            return_value=self.sample_create_job_response_with_default_display_name
        )
        return mock_client

    @pytest.fixture(scope="function")
    def mock_client_with_default_display_name_with_artifact(self):
        mock_client = MagicMock()
        mock_client.create_job = Mock(
            return_value=self.sample_create_job_response_with_default_display_name_with_artifact
        )
        return mock_client

    def test_create_delete(self, mock_details, mock_client):
        job = DSCJob(**self.payload)
        with patch.object(DSCJob, "client", mock_client):
            with patch.object(OCIModelMixin, "to_oci_model", mock_details):
                job.create()
                assert (
                    job.lifecycle_state
                    == oci.data_science.models.Job.LIFECYCLE_STATE_ACTIVE
                )
                assert job.display_name == self.payload["display_name"]

    def test_create_job_with_default_display_name(
        self,
        mock_details_with_default_display_name,
        mock_client_with_default_display_name,
    ):
        # create job with no artifact and no display_name
        self.payload["display_name"] = None
        job = DSCJob(**self.payload)
        with patch.object(DSCJob, "client", mock_client_with_default_display_name):
            with patch.object(
                OCIModelMixin, "to_oci_model", mock_details_with_default_display_name
            ):
                job.create()
                random.seed(random_seed)
                assert (
                    job.display_name[:-9] == utils.get_random_name_for_resource()[:-9]
                )

    def test_create_job_with_default_display_name_with_artifact(
        self,
        mock_details_with_default_display_name_with_artifact,
        mock_client_with_default_display_name_with_artifact,
    ):
        # create job with artifact and no display_name
        self.payload["display_name"] = None
        artifact = "/tmp/my_script_name.py"
        job = DSCJob(artifact=artifact, **self.payload)
        with patch.object(
            DSCJob, "client", mock_client_with_default_display_name_with_artifact
        ):
            with patch.object(
                OCIModelMixin,
                "to_oci_model",
                mock_details_with_default_display_name_with_artifact,
            ):
                job.create()
                assert job.display_name == os.path.basename(str(artifact)).split(".")[0]

    @pytest.mark.parametrize(
        "test_config_details, expected_result",
        [
            (
                {
                    "jobInfrastructureType": "STANDALONE",
                    "shapeName": "VM.Standard.E3.Flex",
                    "jobShapeConfigDetails": {
                        "memoryInGBs": 22.0,
                        "ocpus": 2.0,
                    },
                    "blockStorageSizeInGBs": 100,
                    "subnetId": "test_subnet_id",
                },
                {
                    "jobInfrastructureType": "STANDALONE",
                    "shapeName": "VM.Standard.E3.Flex",
                    "jobShapeConfigDetails": {
                        "memoryInGBs": 22.0,
                        "ocpus": 2.0,
                    },
                    "blockStorageSizeInGBs": 100,
                    "subnetId": "test_subnet_id",
                },
            ),
            (
                {
                    "jobInfrastructureType": "STANDALONE",
                    "shapeName": "VM.Standard.E3.Flex",
                    "blockStorageSizeInGBs": 100,
                    "subnetId": "test_subnet_id",
                },
                {
                    "jobInfrastructureType": "STANDALONE",
                    "shapeName": "VM.Standard.E3.Flex",
                    "jobShapeConfigDetails": {
                        "memoryInGBs": 16.0,
                        "ocpus": 1.0,
                    },
                    "blockStorageSizeInGBs": 100,
                    "subnetId": "test_subnet_id",
                },
            ),
            (
                {
                    "jobInfrastructureType": "STANDALONE",
                },
                {
                    "jobInfrastructureType": "STANDALONE",
                    "shapeName": "VM.Standard.E3.Flex",
                    "jobShapeConfigDetails": {
                        "memoryInGBs": 16.0,
                        "ocpus": 1.0,
                    },
                    "blockStorageSizeInGBs": 100,
                    "subnetId": "test_subnet_id",
                },
            ),
            (
                {
                    "jobInfrastructureType": "STANDALONE",
                    "shapeName": "VM.Standard2.24",
                },
                {
                    "jobInfrastructureType": "STANDALONE",
                    "shapeName": "VM.Standard2.24",
                    "blockStorageSizeInGBs": 100,
                    "subnetId": "test_subnet_id",
                },
            ),
        ],
    )
    def test__load_infra_from_notebook(
        self,
        test_config_details,
        expected_result,
    ):
        payload = {
            "projectId": "test_project_id",
            "compartmentId": "<compartment_id>",
            "displayName": "<job_name>",
            "jobConfigurationDetails": {
                "jobType": "DEFAULT",
            },
        }

        nb_config = DSCNotebookSession(
            **{
                "notebook_session_configuration_details": {
                    "shape": "VM.Standard.E3.Flex",
                    "block_storage_size_in_gbs": 100,
                    "subnet_id": "test_subnet_id",
                    "notebook_session_shape_config_details": {
                        "ocpus": 1.0,
                        "memory_in_gbs": 16.0,
                    },
                }
            }
        ).notebook_session_configuration_details

        payload.update({"jobInfrastructureConfigurationDetails": test_config_details})
        job = DSCJob(**payload)
        job._load_infra_from_notebook(nb_config)
        assert job.job_infrastructure_configuration_details == expected_result
