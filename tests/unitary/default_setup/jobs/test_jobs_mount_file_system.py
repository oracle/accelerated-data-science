#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import copy
from unittest.mock import MagicMock, patch
import oci
import unittest
import pytest

from ads.common.dsc_file_system import DSCFileSystemManager, OCIFileStorage, OCIObjectStorage
from ads.jobs.ads_job import Job
from ads.jobs.builders.infrastructure import DataScienceJob
from ads.jobs.builders.runtimes.python_runtime import PythonRuntime

try:
    from oci.data_science.models import JobStorageMountConfigurationDetails
    from oci.data_science.models import FileStorageMountConfigurationDetails
    from oci.data_science.models import ObjectStorageMountConfigurationDetails
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
        FileStorageMountConfigurationDetails(
            **{
                "destination_directory_name": "test_destination_directory_name_from_dsc",
                "export_id": "export_id_from_dsc",
                "mount_target_id": "mount_target_id_from_dsc",
                "storage_type": "FILE_STORAGE",
            },
        ),
        FileStorageMountConfigurationDetails(
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
            {
                "src" : "oci://bucket_name@namespace/synthetic/",
                "dest" : "test_mount_three",
            } 
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
      - src: oci://bucket_name@namespace/synthetic/
        dest: test_mount_three
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

        dsc_object_storage = job.infrastructure.storage_mount[2]
        assert isinstance(dsc_object_storage, dict)
        assert dsc_object_storage["src"] == "oci://bucket_name@namespace/synthetic/"
        assert dsc_object_storage["dest"] == "test_mount_three"

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

        dsc_object_storage = job.infrastructure.storage_mount[2]
        assert isinstance(dsc_object_storage, dict)
        assert dsc_object_storage["src"] == "oci://bucket_name@namespace/synthetic/"
        assert dsc_object_storage["dest"] == "test_mount_three"

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
                            {
                                "src" : "oci://bucket_name@namespace/synthetic/",
                                "dest" : "test_mount_three",
                            } 
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
            == 3
        )
        assert dsc_job_payload_copy.job_storage_mount_configuration_details_list[
            0
        ] == {
            "destinationDirectoryName": "test_destination_directory_name_from_dsc",
            "exportId": "test_export_id_one",
            "mountTargetId": "test_mount_target_id_one",
            "storageType": "FILE_STORAGE",
        }

    @patch.object(OCIObjectStorage, "update_to_dsc_model")
    @patch.object(OCIFileStorage, "update_to_dsc_model")
    def test_file_manager_process_data(self, mock_fss_update_to_dsc_model, mock_oss_update_to_dsc_model):
        test_mount_file_system = {
            "src" : "1.1.1.1:/test_export",
            "dest" : "test_dest_one"
        }

        DSCFileSystemManager.initialize(test_mount_file_system)
        mock_fss_update_to_dsc_model.assert_called()   

        test_mount_file_system = {
            "src" : "ocid1.mounttarget.xxx:ocid1.export.xxx",
            "dest" : "test_dest_two"
        }

        DSCFileSystemManager.initialize(test_mount_file_system) 
        mock_fss_update_to_dsc_model.assert_called() 

        test_mount_file_system = {
            "src" : "oci://bucket@namespace/prefix",
            "dest" : "test_dest_three"
        }

        DSCFileSystemManager.initialize(test_mount_file_system)
        mock_oss_update_to_dsc_model.assert_called()

    def test_file_manager_process_data_error(self):
        test_mount_file_system = {}
        with pytest.raises(
            ValueError,
            match="Parameter `src` is required for mounting file storage system."
        ):
            DSCFileSystemManager.initialize(test_mount_file_system)

        test_mount_file_system["src"] = "test_src"
        with pytest.raises(
            ValueError,
            match="Parameter `dest` is required for mounting file storage system."
        ):
            DSCFileSystemManager.initialize(test_mount_file_system)

        test_mount_file_system["dest"] = "test_dest_four"
        with pytest.raises(
            ValueError,
            match="Invalid dict for mounting file systems. Specify a valid one."
        ):
            DSCFileSystemManager.initialize(test_mount_file_system)

        test_mount_file_system_list = [test_mount_file_system] * 2
        with pytest.raises(
            ValueError,
            match="Duplicate `dest` found. Please specify different `dest` for each file system to be mounted."
        ):
            for mount_file_system in test_mount_file_system_list:
                DSCFileSystemManager.initialize(mount_file_system)

    def test_dsc_object_storage(self):
        object_storage = OCIObjectStorage(
            src="oci://bucket@namespace/prefix",
            dest="test_dest",
        )

        result = object_storage.update_to_dsc_model()
        assert result["bucket"] == "bucket"
        assert result["namespace"] == "namespace"
        assert result["prefix"] == "prefix"
        assert result["storageType"] == "OBJECT_STORAGE"
        assert result["destinationDirectoryName"] == "test_dest"

        dsc_model = ObjectStorageMountConfigurationDetails(
            **{
                "destination_directory_name": "test_destination_directory_name_from_dsc",
                "storage_type": "OBJECT_STORAGE",
                "bucket": "bucket",
                "namespace": "namespace",
                "prefix": "prefix"
            }
        )
        
        result = OCIObjectStorage.update_from_dsc_model(dsc_model)
        assert result["src"] == "oci://bucket@namespace/prefix"
        assert result["dest"] == "test_destination_directory_name_from_dsc"

    def test_dsc_object_storage_error(self):
        error_messages = {
            "namespace" : "Missing parameter `namespace` from service. Check service log to see the error.",
            "bucket" : "Missing parameter `bucket` from service. Check service log to see the error.",
            "destination_directory_name" : "Missing parameter `destination_directory_name` from service. Check service log to see the error."
        }

        dsc_model_dict = {
            "destination_directory_name": "test_destination_directory_name_from_dsc",
            "storage_type": "OBJECT_STORAGE",
            "bucket": "bucket",
            "namespace": "namespace",
            "prefix": "prefix"
        }

        for error in error_messages:
            with pytest.raises(
                ValueError,
                match=error_messages[error]
            ):
                dsc_model_copy = copy.deepcopy(dsc_model_dict)
                dsc_model_copy.pop(error)
                OCIObjectStorage.update_from_dsc_model(
                    ObjectStorageMountConfigurationDetails(**dsc_model_copy)
                )

    @patch.object(oci.resource_search.ResourceSearchClient, "search_resources")
    def test_dsc_file_storage(self, mock_search_resources):
        file_storage = OCIFileStorage(
            src="ocid1.mounttarget.oc1.iad.xxxx:ocid1.export.oc1.iad.xxxx",
            dest="test_dest",
        )
        file_storage = file_storage.update_to_dsc_model()
        assert file_storage == {
            "destinationDirectoryName" : "test_dest",
            "exportId" : "ocid1.export.oc1.iad.xxxx",
            "mountTargetId" : "ocid1.mounttarget.oc1.iad.xxxx",
            "storageType" : "FILE_STORAGE"
        }

        file_storage = OCIFileStorage(
            src="1.1.1.1:/test_export",
            dest="test_dest",
        )

        items = [
            oci.resource_search.models.resource_summary.ResourceSummary(
                **{
                    "additional_details": {},
                    "availability_domain": "null",
                    "compartment_id": "ocid1.compartment.oc1..aaaaaaaapvb3hearqum6wjvlcpzm5ptfxqa7xfftpth4h72xx46ygavkqteq",
                    "defined_tags": {},
                    "display_name": "test_name",
                    "freeform_tags": {
                        "oci:compute:instanceconfiguration": "ocid1.instanceconfiguration.oc1.iad.xxxx"
                    },
                    "identifier": "ocid1.mounttarget.oc1.iad.xxxx",
                    "identity_context": {},
                    "lifecycle_state": "AVAILABLE",
                    "resource_type": "MountTarget",
                    "search_context": "null",
                    "system_tags": {},
                    "time_created": "2020-09-25T22:43:48.301000+00:00"
                }
            ),
            oci.resource_search.models.resource_summary.ResourceSummary(
                **{
                    "additional_details": {},
                    "availability_domain": "null",
                    "compartment_id": "ocid1.compartment.oc1..aaaaaaaapvb3hearqum6wjvlcpzm5ptfxqa7xfftpth4h72xx46ygavkqteq",
                    "defined_tags": {},
                    "display_name": "test_name",
                    "freeform_tags": {
                        "oci:compute:instanceconfiguration": "ocid1.instanceconfiguration.oc1.iad.xxxx"
                    },
                    "identifier": "ocid1.export.oc1.iad.xxxx",
                    "identity_context": {},
                    "lifecycle_state": "AVAILABLE",
                    "resource_type": "Export",
                    "search_context": "null",
                    "system_tags": {},
                    "time_created": "2020-09-25T22:43:48.301000+00:00"
                }
            )
        ]

        data = MagicMock()
        data.items = items
        return_value = MagicMock()
        return_value.data = data
        mock_search_resources.return_value = return_value

        file_storage = file_storage.update_to_dsc_model()
        assert file_storage == {
            "destinationDirectoryName" : "test_dest",
            "exportId" : "ocid1.export.oc1.iad.xxxx",
            "mountTargetId" : "ocid1.mounttarget.oc1.iad.xxxx",
            "storageType" : "FILE_STORAGE"
        }

        dsc_model = FileStorageMountConfigurationDetails(
            **{
                "destination_directory_name": "test_dest",
                "storage_type": "FILE_STORAGE",
                "export_id": "ocid1.export.oc1.iad.xxxx",
                "mount_target_id": "ocid1.mounttarget.oc1.iad.xxxx"
            }
        )
        result = OCIFileStorage.update_from_dsc_model(dsc_model)
        assert result["src"] == "ocid1.mounttarget.oc1.iad.xxxx:ocid1.export.oc1.iad.xxxx"
        assert result["dest"] == "test_dest"

    def test_dsc_file_storage_error(self):
        error_messages = {
            "mount_target_id" : "Missing parameter `mount_target_id` from service. Check service log to see the error.",
            "export_id" : "Missing parameter `export_id` from service. Check service log to see the error.",
            "destination_directory_name" : "Missing parameter `destination_directory_name` from service. Check service log to see the error."
        }

        dsc_model_dict = {
            "destination_directory_name": "test_destination_directory_name_from_dsc",
            "storage_type": "FILE_STORAGE",
            "mount_target_id": "ocid1.mounttarget.oc1.iad.xxxx",
            "export_id": "ocid1.export.oc1.iad.xxxx",
        }

        for error in error_messages:
            with pytest.raises(
                ValueError,
                match=error_messages[error]
            ):
                dsc_model_copy = copy.deepcopy(dsc_model_dict)
                dsc_model_copy.pop(error)
                OCIFileStorage.update_from_dsc_model(
                    FileStorageMountConfigurationDetails(**dsc_model_copy)
                )