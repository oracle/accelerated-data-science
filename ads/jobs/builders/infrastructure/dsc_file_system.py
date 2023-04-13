#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import ads
import oci
import copy

from ads.common import utils
from dataclasses import asdict, dataclass

FILE_STORAGE_TYPE = "FILE_STORAGE"


@dataclass
class DSCFileSystem:

    destination_directory_name: str = None

    def to_dict(self) -> dict:
        """Converts the object to dictionary."""
        return {utils.snake_to_camel(k): v for k, v in asdict(self).items() if v}

    @classmethod
    def from_dict(cls, env: dict) -> "DSCFileSystem":
        """Initialize the object from a Python dictionary."""
        return cls(**{utils.camel_to_snake(k): v for k, v in env.items()})

    def update_to_dsc_model(self, **kwargs) -> dict:
        return self.to_dict()

    @classmethod
    def update_from_dsc_model(cls, dsc_model: dict) -> "DSCFileSystem":
        return cls.from_dict(dsc_model)


@dataclass
class OCIFileStorage(DSCFileSystem):

    mount_target: str = None
    mount_target_id: str = None
    export_path: str = None
    export_id: str = None
    storage_type: str = FILE_STORAGE_TYPE

    def __post_init__(self):
        if not self.destination_directory_name:
            raise ValueError(
                "Parameter `destination_directory_name` must be provided to mount file system."
            )

        if not self.mount_target and not self.mount_target_id:
            raise ValueError(
                "Either parameter `mount_target` or `mount_target_id` must be provided to mount file system."
            )

        if not self.export_path and not self.export_id:
            raise ValueError(
                "Either parameter `export_path` or `export_id` must be provided to mount file system."
            )

    def update_to_dsc_model(self, **kwargs) -> dict:
        """Updates arguments to dsc model.

        Returns
        -------
        dict:
            A dictionary of arguments.
        """
        auth = ads.auth.default_signer()
        file_storage_client = oci.file_storage.FileStorageClient(**auth)
        identity_client = oci.identity.IdentityClient(**auth)

        arguments = self.to_dict()

        compartment_id = kwargs["compartment_id"]
        if "mountTargetId" not in arguments:
            list_availability_domains_response = (
                identity_client.list_availability_domains(
                    compartment_id=compartment_id
                ).data
            )
            mount_targets = []
            for availability_domain in list_availability_domains_response:
                mount_targets.extend(
                    file_storage_client.list_mount_targets(
                        compartment_id=compartment_id,
                        availability_domain=availability_domain.name,
                    ).data
                )
            mount_targets = [
                mount_target.id
                for mount_target in mount_targets
                if mount_target.display_name == self.mount_target
            ]
            if len(mount_targets) == 0:
                raise ValueError(
                    f"No `mount_target` with value {self.mount_target} found under compartment {compartment_id}. Specify a valid one."
                )
            if len(mount_targets) > 1:
                raise ValueError(
                    f"Multiple `mount_target` with value {self.mount_target} found under compartment {compartment_id}. Specify `mount_target_id` of the file system instead."
                )
            arguments["mountTargetId"] = mount_targets[0]
            arguments.pop("mountTarget")

        if "exportId" not in arguments:
            list_exports_response = file_storage_client.list_exports(
                compartment_id=compartment_id
            ).data
            exports = [
                export.id
                for export in list_exports_response
                if export.path == self.export_path
            ]
            if len(exports) == 0:
                raise ValueError(
                    f"No `export_path` with value {self.export_path} found under compartment {compartment_id}. Specify a valid one."
                )
            if len(exports) > 1:
                raise ValueError(
                    f"Multiple `export_path` with value {self.export_path} found under compartment {compartment_id}. Specify `export_id` of the file system instead."
                )
            arguments["exportId"] = exports[0]
            arguments.pop("exportPath")

        return arguments

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
