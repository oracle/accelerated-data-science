#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import ads
import oci
import copy
import ipaddress

from ads.common import utils
from dataclasses import asdict, dataclass

FILE_STORAGE_TYPE = "FILE_STORAGE"


@dataclass
class DSCFileSystem:

    src: str = None
    dsc: str = None
    storage_type: str = None
    destination_directory_name: str = None

    def to_dict(self) -> dict:
        """Converts the object to dictionary."""
        return {utils.snake_to_camel(k): v for k, v in asdict(self).items() if v}

    @classmethod
    def from_dict(cls, env: dict) -> "DSCFileSystem":
        """Initialize the object from a Python dictionary."""
        return cls(**{utils.camel_to_snake(k): v for k, v in env.items()})

    def update_to_dsc_model(self) -> dict:
        return self.to_dict()

    @classmethod
    def update_from_dsc_model(cls, dsc_model: dict) -> "DSCFileSystem":
        return cls.from_dict(dsc_model)


@dataclass
class OCIFileStorage(DSCFileSystem):

    mount_target_id: str = None
    mount_target: str = None
    export_id: str = None
    export_path: str = None
    storage_type: str = FILE_STORAGE_TYPE

    def __post_init__(self):
        if not self.src:
            if not self.mount_target_id:
                raise ValueError(
                    "Missing required parameter. Either `src` or `mount_target_id` is required for mounting file storage system."
                )

            if not self.export_id:
                raise ValueError(
                    "Missing required parameter. Either `src` or `export_id` is required for mounting file storage system."
                )

        if not self.dsc:
            if not self.destination_directory_name:
                raise ValueError(
                    "Parameter `dsc` is required for mounting file storage system."
                )

    def update_to_dsc_model(self) -> dict:
        """Updates arguments to dsc model.

        Returns
        -------
        dict:
            A dictionary of arguments.
        """
        arguments = self.to_dict()

        if "exportId" not in arguments:
            arguments["exportId"] = self._get_export_id(arguments)

        if "mountTargetId" not in arguments:
            arguments["mountTargetId"] = self._get_mount_target_id(arguments)
        
        arguments.pop("src")
        arguments["destinationDirectoryName"] = arguments.pop("dsc")

        return arguments

    def _get_export_id(self, arguments: dict) -> str:
        file_storage_client = oci.file_storage.FileStorageClient(**ads.auth.default_signer())
        src_list = arguments["src"].split(":")
        ip = src_list[0]
        export_path = src_list[1]

        resource_summary = self._get_resource(ip)

        list_exports_response = file_storage_client.list_exports(
            compartment_id=resource_summary.compartment_id
        ).data
        exports = [
            export.id
            for export in list_exports_response
            if export.path == export_path
        ]
        if len(exports) == 0:
            raise ValueError(
                f"No `export_id` found under ip {ip}. Specify a valid `src`."
            )
        if len(exports) > 1:
            raise ValueError(
                f"Multiple `export_id` found under ip {ip}. Specify `export_id` of the file system instead."
            )

        return exports[0]

    def _get_mount_target_id(self, arguments: dict) -> str:
        file_storage_client = oci.file_storage.FileStorageClient(**ads.auth.default_signer())
        ip = arguments["src"].split(":")[0]
        resource = self._get_resource(ip)

        mount_targets =  file_storage_client.list_mount_targets(
                            compartment_id=resource.compartment_id,
                            availability_domain=resource.availability_domain,
                            export_set_id=file_storage_client.get_export(arguments["exportId"]).data.export_set_id
                        ).data
        mount_targets = [
            mount_target.id
            for mount_target in mount_targets
            if resource.identifier in mount_target.private_ip_ids
        ]
        if len(mount_targets) == 0:
            raise ValueError(
                f"No `mount_target_id` found under ip {ip}. Specify a valid `src`."
            )
        if len(mount_targets) > 1:
            raise ValueError(
                f"Multiple `mount_target_id` found under ip {ip}. Specify `mount_target_id` of the file system instead."
            )
        return mount_targets[0]

    def _get_resource(self, ip: str) -> oci.resource_search.models.ResourceSummary:
        resource_client = oci.resource_search.ResourceSearchClient(**ads.auth.default_signer())
        resource = resource_client.search_resources(
            search_details=oci.resource_search.models.FreeTextSearchDetails(
                text=ip,
                matching_context_type="NONE"
            )
        ).data.items

        resource = sorted(resource, key=lambda resource_summary: resource_summary.time_created)

        if not resource or not hasattr(resource[-1], "compartment_id") or not hasattr(resource[-1], "identifier"):
            raise ValueError(f"Can't find the compartment id or identifier from ip {ip}. Specify a valid `src`.")

        return resource[-1]

    @classmethod
    def update_from_dsc_model(cls, dsc_model: dict) -> DSCFileSystem:
        """Updates arguments and builds DSCFileSystem object from dsc model.

        Parameters
        ----------
        dsc_model: dict
            A dictionary of arguments from dsc model.

        Returns
        -------
        DSCFileSystem
            An instance of DSCFileSystem.
        """
        argument = copy.deepcopy(dsc_model)

        file_storage_client = oci.file_storage.FileStorageClient(
            **ads.auth.default_signer()
        )
        if "mountTargetId" not in argument:
            raise ValueError(
                "Missing parameter `mountTargetId` from service. Check service log to see the error."
            )
        argument["mountTarget"] = file_storage_client.get_mount_target(
            mount_target_id=argument.get("mountTargetId")
        ).data.display_name
        if "exportId" not in argument:
            raise ValueError(
                "Missing parameter `exportId` from service. Check service log to see the error."
            )
        argument["exportPath"] = file_storage_client.get_export(
            export_id=argument.get("exportId")
        ).data.path

        return super().from_dict(argument)


class DSCFileSystemManager:

    @classmethod
    def initialize(cls, arguments: dict) -> DSCFileSystem:
        if "src" in arguments:
            try:
                # case <ip_address>:<export_path>
                ipaddress.IPv4Network(arguments["src"].split(":")[0])
                return OCIFileStorage(**arguments)
            except:
                pass
        elif "mount_target_id" in arguments or "export_id" in arguments:
            return OCIFileStorage(**arguments)

        raise ValueError("Invalid dict for mounting file systems. Specify a valid one.")
