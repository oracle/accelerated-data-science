#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import ads
import oci
import ipaddress

from dataclasses import dataclass

FILE_STORAGE_TYPE = "FILE_STORAGE"
OBJECT_STORAGE_TYPE = "OBJECT_STORAGE"


@dataclass
class DSCFileSystem:

    src: str = None
    dest: str = None
    storage_type: str = None
    destination_directory_name: str = None

    def update_to_dsc_model(self) -> dict:
        """Updates arguments to dsc model.

        Returns
        -------
        dict:
            A dictionary of arguments.
        """
        pass

    @classmethod
    def update_from_dsc_model(cls, dsc_model) -> dict:
        """Updates arguments from dsc model.

        Parameters
        ----------
        dsc_model: oci.data_science.models.JobStorageMountConfigurationDetails
            An instance of oci.data_science.models.JobStorageMountConfigurationDetails.

        Returns
        -------
        dict
            A dictionary of arguments.
        """
        pass


@dataclass
class OCIFileStorage(DSCFileSystem):

    mount_target_id: str = None
    mount_target: str = None
    export_id: str = None
    export_path: str = None
    storage_type: str = FILE_STORAGE_TYPE        

    def update_to_dsc_model(self) -> dict:
        """Updates arguments to dsc model.

        Returns
        -------
        dict:
            A dictionary of arguments.
        """
        arguments = {
            "destinationDirectoryName" : self.dest,
            "storageType" : self.storage_type
        }
        
        self._get_mount_target_and_export_ids(arguments)

        return arguments

    def _get_mount_target_and_export_ids(self, arguments: dict):
        """Gets the mount target id and export id from src.

        Parameters
        ----------
        arguments: dict
            A dictionary of arguments.
        """
        resource_client = oci.resource_search.ResourceSearchClient(**ads.auth.default_signer())
        src_list = self.src.split(":")
        first_segment = src_list[0]
        second_segment = src_list[1]

        if first_segment.startswith("ocid") and "mounttarget" in first_segment:
            arguments["mountTargetId"] = first_segment
        else:
            ip_resource = resource_client.search_resources(
                search_details=oci.resource_search.models.FreeTextSearchDetails(
                    text=first_segment,
                    matching_context_type="NONE"
                )
            ).data.items

            ip_resource = sorted(ip_resource, key=lambda resource_summary: resource_summary.time_created)

            if not ip_resource or not hasattr(ip_resource[-1], "identifier"):
                raise ValueError(f"Can't find the identifier from ip {first_segment}. Specify a valid `src`.")

            mount_target_resource = resource_client.search_resources(
                search_details=oci.resource_search.models.FreeTextSearchDetails(
                    text=ip_resource[-1].identifier,
                    matching_context_type="NONE"
                )
            ).data.items

            mount_targets = [
                mount_target.identifier
                for mount_target in mount_target_resource
                if mount_target.resource_type == "MountTarget"
            ]
            if len(mount_targets) == 0:
                raise ValueError(
                    f"No `mount_target_id` found under ip {first_segment}. Specify a valid `src`."
                )
            if len(mount_targets) > 1:
                raise ValueError(
                    f"Multiple `mount_target_id` found under ip {first_segment}. Specify the `mount_target_id` in `src` instead."
                )

            arguments["mountTargetId"] = mount_targets[0]
        
        if second_segment.startswith("ocid") and "export" in second_segment:
            arguments["exportId"] = second_segment
        else:
            export_resource = resource_client.search_resources(
                search_details=oci.resource_search.models.FreeTextSearchDetails(
                    text=second_segment,
                    matching_context_type="NONE"
                )
            ).data.items

            exports = [
                export.identifier
                for export in export_resource
                if export.resource_type == "Export"
            ]
            if len(exports) == 0:
                raise ValueError(
                    f"No `export_id` found with `export_path` {second_segment}. Specify a valid `src`."
                )
            if len(exports) > 1:
                raise ValueError(
                    f"Multiple `export_id` found with `export_path` {second_segment}. Specify the `export_id` in `src` instead."
                )

            arguments["exportId"] = exports[0]

    @classmethod
    def update_from_dsc_model(cls, dsc_model) -> dict:
        """Updates arguments from dsc model.

        Parameters
        ----------
        dsc_model: oci.data_science.models.JobStorageMountConfigurationDetails
            An instance of oci.data_science.models.JobStorageMountConfigurationDetails.

        Returns
        -------
        dict
            A dictionary of arguments.
        """
        if not dsc_model.mount_target_id:
            raise ValueError(
                "Missing parameter `mount_target_id` from service. Check service log to see the error."
            )
        if not dsc_model.export_id:
            raise ValueError(
                "Missing parameter `export_id` from service. Check service log to see the error."
            )
        if not dsc_model.destination_directory_name:
            raise ValueError(
                "Missing parameter `destination_directory_name` from service. Check service log to see the error."
            )

        return {
            "src" : f"{dsc_model.mount_target_id}:{dsc_model.export_id}",
            "dest" : dsc_model.destination_directory_name
        }

@dataclass
class OCIObjectStorage(DSCFileSystem):

    storage_type: str = OBJECT_STORAGE_TYPE

    def update_to_dsc_model(self) -> dict:
        arguments = {
            "destinationDirectoryName" : self.dest,
            "storageType" : self.storage_type
        }
        src_list = self.src.split("@")
        bucket_segment = src_list[0]
        namespace_segment = src_list[1].strip("/")
        arguments["bucket"] = bucket_segment[6:]
        if "/" in namespace_segment:
            first_slash_index = namespace_segment.index("/")
            arguments["namespace"] = namespace_segment[:first_slash_index]
            arguments["prefix"] = namespace_segment[first_slash_index+1:]
        else:
            arguments["namespace"] = namespace_segment
        return arguments

    @classmethod
    def update_from_dsc_model(cls, dsc_model) -> dict:
        if not dsc_model.namespace:
            raise ValueError(
                "Missing parameter `namespace` from service. Check service log to see the error."
            )
        if not dsc_model.bucket:
            raise ValueError(
                "Missing parameter `bucket` from service. Check service log to see the error."
            )
        if not dsc_model.destination_directory_name:
            raise ValueError(
                "Missing parameter `destination_directory_name` from service. Check service log to see the error."
            )

        return {
            "src" : f"oci://{dsc_model.bucket}@{dsc_model.namespace}/{dsc_model.prefix or ''}",
            "dest" : dsc_model.destination_directory_name
        }


class DSCFileSystemManager:

    storage_mount_dest = set()

    @classmethod
    def initialize(cls, arguments: dict) -> dict:
        """Initialize and update arguments to dsc model.

        Parameters
        ----------
        arguments: dict
            A dictionary of arguments.
        """
        if "src" not in arguments:
            raise ValueError(
                "Parameter `src` is required for mounting file storage system."
            )

        if "dest" not in arguments:
            raise ValueError(
                "Parameter `dest` is required for mounting file storage system."
            )

        if arguments["dest"] in cls.storage_mount_dest:
            raise ValueError(
                "Duplicate `dest` found. Please specify different `dest` for each file system to be mounted."
            )
        cls.storage_mount_dest.add(arguments["dest"])

        # case oci://bucket@namespace/prefix
        if arguments["src"].startswith("oci://") and "@" in arguments["src"]:
            return OCIObjectStorage(**arguments).update_to_dsc_model()

        first_segment = arguments["src"].split(":")[0]
        # case <mount_target_id>:<export_id>
        if first_segment.startswith("ocid"):
            return OCIFileStorage(**arguments).update_to_dsc_model()

        # case <ip_address>:<export_path>
        try:
            ipaddress.IPv4Network(first_segment)
            return OCIFileStorage(**arguments).update_to_dsc_model()
        except:
            pass

        raise ValueError("Invalid dict for mounting file systems. Specify a valid one.")
