#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import copy
from unittest.mock import MagicMock, patch
import oci
import unittest
import pytest

from ads.jobs.ads_job import Job
from ads.jobs.builders.infrastructure import DSCFileStorage, DataScienceJob
from ads.jobs.builders.runtimes.python_runtime import PythonRuntime

try:
    from oci.data_science.models import FileStorageMountConfigurationDetails
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
        {
            "destinationDirectoryName": "test_destination_directory_name_from_dsc",
            "exportId": "export_id_from_dsc",
            "mountTargetId": "mount_target_id_from_dsc",
            "storageType": "FILE_STORAGE",
        },
        {
            "destinationDirectoryName": "test_destination_directory_name_from_dsc",
            "exportId": "export_id_from_dsc",
            "mountTargetId": "mount_target_id_from_dsc",
            "storageType": "FILE_STORAGE",
        },
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
            DSCFileStorage(
                destination_directory_name="test_mount_one",
                mount_target="test_mount_target_one",
                export_path="test_export_path_one",
            ),
            {
                "destination_directory_name": "test_mount_two",
                "mount_target": "test_mount_target_two",
                "export_path": "test_export_path_two",
                "storage_type": "FILE_STORAGE",
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
      - destinationDirectoryName: test_mount_one
        mountTarget: test_mount_target_one
        exportPath: test_export_path_one
        storageType: FILE_STORAGE
      - destinationDirectoryName: test_mount_two
        mountTarget: test_mount_target_two
        exportPath: test_export_path_two
        storageType: FILE_STORAGE
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
        assert isinstance(dsc_file_storage_one, DSCFileStorage)
        assert dsc_file_storage_one.storage_type == "FILE_STORAGE"
        assert dsc_file_storage_one.destination_directory_name == "test_mount_one"
        assert dsc_file_storage_one.mount_target == "test_mount_target_one"
        assert dsc_file_storage_one.export_path == "test_export_path_one"

        dsc_file_storage_two = job.infrastructure.storage_mount[1]
        assert isinstance(dsc_file_storage_two, DSCFileStorage)
        assert dsc_file_storage_two.storage_type == "FILE_STORAGE"
        assert dsc_file_storage_two.destination_directory_name == "test_mount_two"
        assert dsc_file_storage_two.mount_target == "test_mount_target_two"
        assert dsc_file_storage_two.export_path == "test_export_path_two"

    def test_data_science_job_from_yaml(self):
        job_from_yaml = Job.from_yaml(job_yaml_string)

        assert isinstance(job_from_yaml.infrastructure.storage_mount, list)
        dsc_file_storage_one = job_from_yaml.infrastructure.storage_mount[0]
        assert isinstance(dsc_file_storage_one, DSCFileStorage)
        assert dsc_file_storage_one.storage_type == "FILE_STORAGE"
        assert dsc_file_storage_one.destination_directory_name == "test_mount_one"
        assert dsc_file_storage_one.mount_target == "test_mount_target_one"
        assert dsc_file_storage_one.export_path == "test_export_path_one"

        dsc_file_storage_two = job.infrastructure.storage_mount[1]
        assert isinstance(dsc_file_storage_two, DSCFileStorage)
        assert dsc_file_storage_two.storage_type == "FILE_STORAGE"
        assert dsc_file_storage_two.destination_directory_name == "test_mount_two"
        assert dsc_file_storage_two.mount_target == "test_mount_target_two"
        assert dsc_file_storage_two.export_path == "test_export_path_two"

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
                                "destinationDirectoryName": "test_mount_one",
                                "mountTarget": "test_mount_target_one",
                                "exportPath": "test_export_path_one",
                                "storageType": "FILE_STORAGE",
                            },
                            {
                                "destinationDirectoryName": "test_mount_two",
                                "mountTarget": "test_mount_target_two",
                                "exportPath": "test_export_path_two",
                                "storageType": "FILE_STORAGE",
                            },
                        ],
                    },
                },
            },
        }

    def test_mount_file_system_failed(self):
        with pytest.raises(
            ValueError,
            match="Either parameter `export_path` or `export_id` must be provided to mount file system.",
        ):
            DSCFileStorage(
                destination_directory_name="test_mount",
                mount_target_id="ocid1.mounttarget.oc1.iad.xxxx",
            )

        with pytest.raises(
            ValueError,
            match="Either parameter `mount_target` or `mount_target_id` must be provided to mount file system.",
        ):
            DSCFileStorage(
                destination_directory_name="test_mount",
                export_id="ocid1.export.oc1.iad.xxxx",
            )

        with pytest.raises(
            ValueError,
            match="Parameter `destination_directory_name` must be provided to mount file system.",
        ):
            DSCFileStorage(
                mount_target_id="ocid1.mounttarget.oc1.iad.xxxx",
                export_id="ocid1.export.oc1.iad.xxxx",
            )

        job_copy = copy.deepcopy(job)
        dsc_file_storage = DSCFileStorage(
            destination_directory_name="test_mount",
            mount_target="test_mount_target",
            export_id="ocid1.export.oc1.iad.xxxx",
        )
        storage_mount_list = [dsc_file_storage] * 6
        with pytest.raises(
            ValueError,
            match="A maximum number of 5 file systems are allowed to be mounted at this time for a job.",
        ):
            job_copy.infrastructure.with_storage_mount(*storage_mount_list)

        job_copy = copy.deepcopy(job)
        with pytest.raises(
            ValueError,
            match="Parameter `storage_mount` should be a list of either DSCFileSystem instances or dictionaries.",
        ):
            job_copy.infrastructure.with_storage_mount(dsc_file_storage, [1, 2, 3])

        job_copy = copy.deepcopy(job)
        with pytest.raises(
            ValueError,
            match="Parameter `storage_type` must be provided for each file system to be mounted.",
        ):
            job_copy.infrastructure.with_storage_mount(
                dsc_file_storage,
                {
                    "destination_directory_name": "test_mount",
                    "mount_target_id": "ocid1.mounttarget.oc1.iad.xxxx",
                    "export_id": "ocid1.export.oc1.iad.xxxx",
                },
            )

        job_copy = copy.deepcopy(job)
        wrong_type = "WRONG_TYPE"
        wrong_file_system_dict = {
            "destination_directory_name": "test_mount",
            "mount_target_id": "ocid1.mounttarget.oc1.iad.xxxx",
            "export_id": "ocid1.export.oc1.iad.xxxx",
            "storage_type": wrong_type,
        }
        with pytest.raises(
            ValueError, match=f"Storage type {wrong_type} is not supprted."
        ):
            job_copy.infrastructure.with_storage_mount(
                dsc_file_storage, wrong_file_system_dict
            )

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
        assert isinstance(infrastructure.storage_mount[0], DSCFileStorage)
        assert isinstance(infrastructure.storage_mount[1], DSCFileStorage)
        assert infrastructure.storage_mount[0].to_dict() == {
            "destinationDirectoryName": "test_destination_directory_name_from_dsc",
            "exportId": "export_id_from_dsc",
            "exportPath": "export_path_from_dsc",
            "mountTarget": "mount_target_from_dsc",
            "mountTargetId": "mount_target_id_from_dsc",
            "storageType": "FILE_STORAGE",
        }
        assert infrastructure.storage_mount[1].to_dict() == {
            "destinationDirectoryName": "test_destination_directory_name_from_dsc",
            "exportId": "export_id_from_dsc",
            "exportPath": "export_path_from_dsc",
            "mountTarget": "mount_target_from_dsc",
            "mountTargetId": "mount_target_id_from_dsc",
            "storageType": "FILE_STORAGE",
        }

    @patch.object(oci.file_storage.FileStorageClient, "list_exports")
    @patch.object(oci.file_storage.FileStorageClient, "list_mount_targets")
    @patch.object(oci.identity.IdentityClient, "list_availability_domains")
    def test_update_job_infra(
        self, mock_list_availability_domains, mock_list_mount_targets, mock_list_exports
    ):
        job_copy = copy.deepcopy(job)
        dsc_job_payload_copy = copy.deepcopy(dsc_job_payload)

        list_availability_domains_mock = MagicMock()
        list_availability_domains_mock.data = [
            oci.identity.models.availability_domain.AvailabilityDomain(
                compartment_id=job_copy.infrastructure.compartment_id,
                name="NNFR:US-ASHBURN-AD-1",
                id="test_id_one",
            )
        ]

        mock_list_availability_domains.return_value = list_availability_domains_mock

        list_mount_targets_mock = MagicMock()
        list_mount_targets_mock.data = [
            oci.file_storage.models.mount_target_summary.MountTargetSummary(
                **{
                    "availability_domain": "NNFR:US-ASHBURN-AD-1",
                    "compartment_id": job_copy.infrastructure.compartment_id,
                    "display_name": "test_mount_target_one",
                    "id": "test_mount_target_id_one",
                }
            ),
        ]
        mock_list_mount_targets.return_value = list_mount_targets_mock

        list_exports_mock = MagicMock()
        list_exports_mock.data = [
            oci.file_storage.models.export.Export(
                **{
                    "id": "test_export_id_one",
                    "path": "test_export_path_one",
                }
            ),
            oci.file_storage.models.export.Export(
                **{
                    "id": "test_export_id_two",
                    "path": "test_export_path_two",
                }
            ),
        ]
        mock_list_exports.return_value = list_exports_mock

        dsc_job_payload_copy.job_storage_mount_configuration_details_list = []
        infrastructure = job_copy.infrastructure
        with pytest.raises(
            ValueError,
            match="No `mount_target` with value test_mount_target_two found under compartment test_compartment_id.",
        ):
            infrastructure._update_job_infra(dsc_job_payload_copy)

            assert (
                len(dsc_job_payload_copy.job_storage_mount_configuration_details_list)
                == 1
            )
            assert dsc_job_payload_copy.job_storage_mount_configuration_details_list[
                0
            ] == {
                "destinationDirectoryName": "test_destination_directory_name_from_dsc",
                "exportId": "test_export_id_one",
                "mountTargetId": "test_mount_target_id_one",
                "storageType": "FILE_STORAGE",
            }
