#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import copy
from unittest.mock import MagicMock, patch
import oci
import unittest
import pytest

from ads.common.dsc_file_system import OCIFileStorage
from ads.jobs.ads_job import Job
from ads.jobs.builders.infrastructure import DataScienceJob
from ads.jobs.builders.runtimes.python_runtime import PythonRuntime

try:
    from oci.data_science.models import JobStorageMountConfigurationDetails
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest(
        "Support for mounting file systems to OCI Job is not available. Skipping the Job tests."
    )

dsc_job_payload = oci.data_science.models.Job(
    compartment_id="test_compartment_id",
    created_by="test_created_by",
    description="test_description",
    display_name="test_display_name",
    freeform_tags={"test_key": "test_value"},
    id="test_id",
    job_configuration_details=oci.data_science.models.DefaultJobConfigurationDetails(
        **{
            "command_line_arguments": [],
            "environment_variables": {"key": "value"},
            "job_type": "DEFAULT",
            "maximum_runtime_in_minutes": 10,
        }
    ),
    job_log_configuration_details=oci.data_science.models.JobLogConfigurationDetails(
        **{
            "enable_auto_log_creation": False,
            "enable_logging": True,
            "log_group_id": "test_log_group_id",
            "log_id": "test_log_id",
        },
    ),
    job_storage_mount_configuration_details_list=[
        oci.data_science.models.FileStorageMountConfigurationDetails(
            **{
                "destination_directory_name": "test_destination_directory_name_from_dsc",
                "export_id": "export_id_from_dsc",
                "mount_target_id": "mount_target_id_from_dsc",
                "storage_type": "FILE_STORAGE",
            },
        ),
        oci.data_science.models.FileStorageMountConfigurationDetails(
            **{
                "destination_directory_name": "test_destination_directory_name_from_dsc",
                "export_id": "export_id_from_dsc",
                "mount_target_id": "mount_target_id_from_dsc",
                "storage_type": "FILE_STORAGE",
            }
        )
    ],
    lifecycle_details="ACTIVE",
    lifecycle_state="STATE",
    project_id="test_project_id",
)

job = (
    Job(name="My Job")
    .with_infrastructure(
        DataScienceJob()
        .with_subnet_id("ocid1.subnet.oc1.iad.xxxx")
        .with_shape_name("VM.Standard.E3.Flex")
        .with_shape_config_details(memory_in_gbs=16, ocpus=1)
        .with_block_storage_size(50)
        .with_storage_mount(
            {
                "src" : "1.1.1.1:test_export_path_one",
                "dest" : "test_mount_one",
            },
            {
                "src" : "2.2.2.2:test_export_path_two",
                "dest" : "test_mount_two",
            },  
        )
    )
    .with_runtime(
        PythonRuntime()
        .with_service_conda("pytorch110_p38_cpu_v1")
        .with_source("custom_script.py")
        .with_environment_variable(NAME="Welcome to OCI Data Science.")
    )
)

job_yaml_string = """
kind: job
spec:
  infrastructure:
    kind: infrastructure
    spec:
      blockStorageSize: 50
      jobType: DEFAULT
      shapeConfigDetails:
        memoryInGBs: 16
        ocpus: 1
      shapeName: VM.Standard.E3.Flex
      storageMount:
      - src: 1.1.1.1:test_export_path_one
        dest: test_mount_one
      - src: 2.2.2.2:test_export_path_two
        dest: test_mount_two
      subnetId: ocid1.subnet.oc1.iad.xxxx
    type: dataScienceJob
  name: My Job
  runtime:
    kind: runtime
    spec:
      conda:
        slug: pytorch110_p38_cpu_v1
        type: service
      env:
      - name: NAME
        value: Welcome to OCI Data Science.
      scriptPathURI: custom_script.py
    type: python
"""


class TestDataScienceJobMountFileSystem(unittest.TestCase):
    def test_data_science_job_initialize(self):
        assert isinstance(job.infrastructure.storage_mount, list)
        dsc_file_storage_one = job.infrastructure.storage_mount[0]
        assert isinstance(dsc_file_storage_one, dict)
        assert dsc_file_storage_one["src"] == "1.1.1.1:test_export_path_one"
        assert dsc_file_storage_one["dest"] == "test_mount_one"

        dsc_file_storage_two = job.infrastructure.storage_mount[1]
        assert isinstance(dsc_file_storage_two, dict)
        assert dsc_file_storage_two["src"] == "2.2.2.2:test_export_path_two"
        assert dsc_file_storage_two["dest"] == "test_mount_two"

    def test_data_science_job_from_yaml(self):
        job_from_yaml = Job.from_yaml(job_yaml_string)

        assert isinstance(job_from_yaml.infrastructure.storage_mount, list)
        dsc_file_storage_one = job_from_yaml.infrastructure.storage_mount[0]
        assert isinstance(dsc_file_storage_one, dict)
        assert dsc_file_storage_one["src"] == "1.1.1.1:test_export_path_one"
        assert dsc_file_storage_one["dest"] == "test_mount_one"

        dsc_file_storage_two = job.infrastructure.storage_mount[1]
        assert isinstance(dsc_file_storage_two, dict)
        assert dsc_file_storage_two["src"] == "2.2.2.2:test_export_path_two"
        assert dsc_file_storage_two["dest"] == "test_mount_two"

    def test_data_science_job_to_dict(self):
        assert job.to_dict() == {
            "kind": "job",
            "spec": {
                "name": "My Job",
                "runtime": {
                    "kind": "runtime",
                    "type": "python",
                    "spec": {
                        "conda": {"type": "service", "slug": "pytorch110_p38_cpu_v1"},
                        "scriptPathURI": "custom_script.py",
                        "env": [
                            {"name": "NAME", "value": "Welcome to OCI Data Science."}
                        ],
                    },
                },
                "infrastructure": {
                    "kind": "infrastructure",
                    "type": "dataScienceJob",
                    "spec": {
                        "jobType": "DEFAULT",
                        "subnetId": "ocid1.subnet.oc1.iad.xxxx",
                        "shapeName": "VM.Standard.E3.Flex",
                        "shapeConfigDetails": {"ocpus": 1, "memoryInGBs": 16},
                        "blockStorageSize": 50,
                        "storageMount": [
                            {
                                "src" : "1.1.1.1:test_export_path_one",
                                "dest" : "test_mount_one",
                            },
                            {
                                "src" : "2.2.2.2:test_export_path_two",
                                "dest" : "test_mount_two",
                            },
                        ],
                    },
                },
            },
        }

    def test_mount_file_system_failed(self):
        job_copy = copy.deepcopy(job)
        dsc_file_storage = {
            "src" : "1.1.1.1:test_export_path",
            "dest" : "test_mount",
        }
        storage_mount_list = [dsc_file_storage] * 6
        with pytest.raises(
            ValueError,
            match="A maximum number of 5 file systems are allowed to be mounted at this time for a job.",
        ):
            job_copy.infrastructure.with_storage_mount(*storage_mount_list)

        job_copy = copy.deepcopy(job)
        with pytest.raises(
            ValueError,
            match="Parameter `storage_mount` should be a list of dictionaries.",
        ):
            job_copy.infrastructure.with_storage_mount(dsc_file_storage, [1, 2, 3])

    @patch.object(oci.file_storage.FileStorageClient, "get_export")
    @patch.object(oci.file_storage.FileStorageClient, "get_mount_target")
    def test_update_storage_mount_from_dsc_model(
        self, mock_get_mount_target, mock_get_export
    ):
        mount_target_mock = MagicMock()
        mount_target_mock.data = MagicMock()
        mount_target_mock.data.display_name = "mount_target_from_dsc"
        mock_get_mount_target.return_value = mount_target_mock

        export_mock = MagicMock()
        export_mock.data = MagicMock()
        export_mock.data.path = "export_path_from_dsc"
        mock_get_export.return_value = export_mock
        job_copy = copy.deepcopy(job)
        infrastructure = job_copy.infrastructure
        infrastructure._update_from_dsc_model(dsc_job_payload)

        assert len(infrastructure.storage_mount) == 2
        assert isinstance(infrastructure.storage_mount[0], dict)
        assert isinstance(infrastructure.storage_mount[1], dict)
        assert infrastructure.storage_mount[0] == {
            "src" : "mount_target_id_from_dsc:export_id_from_dsc",
            "dest" : "test_destination_directory_name_from_dsc"
        }
        assert infrastructure.storage_mount[1] == {
            "src" : "mount_target_id_from_dsc:export_id_from_dsc",
            "dest" : "test_destination_directory_name_from_dsc"
        }

    @patch.object(OCIFileStorage, "update_to_dsc_model")
    def test_update_job_infra(
        self, mock_update_to_dsc_model
    ):
        job_copy = copy.deepcopy(job)
        dsc_job_payload_copy = copy.deepcopy(dsc_job_payload)

        mock_update_to_dsc_model.return_value = {
            "destinationDirectoryName": "test_destination_directory_name_from_dsc",
            "exportId": "test_export_id_one",
            "mountTargetId": "test_mount_target_id_one",
            "storageType": "FILE_STORAGE",
        }

        dsc_job_payload_copy.job_storage_mount_configuration_details_list = []
        infrastructure = job_copy.infrastructure
        infrastructure._update_job_infra(dsc_job_payload_copy)

        assert (
            len(dsc_job_payload_copy.job_storage_mount_configuration_details_list)
            == 2
        )
        assert dsc_job_payload_copy.job_storage_mount_configuration_details_list[
            0
        ] == {
            "destinationDirectoryName": "test_destination_directory_name_from_dsc",
            "exportId": "test_export_id_one",
            "mountTargetId": "test_mount_target_id_one",
            "storageType": "FILE_STORAGE",
        }
