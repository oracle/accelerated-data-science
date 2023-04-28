#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

from ads.common import auth as auth_util
from ads.common import oci_client
from ads.config import JOB_RUN_COMPARTMENT_OCID, NB_SESSION_COMPARTMENT_OCID
from ads.data_labeling.interface.reader import Reader
from ads.data_labeling.metadata import Metadata
from ads.data_labeling.parser.export_metadata_parser import MetadataParser
from ads.data_labeling.reader.jsonl_reader import JsonlReader
from oci.exceptions import ServiceError


class EmptyMetadata(Exception):   # pragma: no cover
    """Empty Metadata."""

    pass


class ReadDatasetError(Exception):   # pragma: no cover
    def __init__(self, id: str):
        super().__init__(f"Error occurred in attempt to read dataset {id}.")


class DatasetNotFoundError(Exception):   # pragma: no cover
    def __init__(self, id: str):
        super().__init__(f"{id} not found.")


class MetadataReader:
    """MetadataReader class which reads and extracts the labeled dataset metadata.

    Examples
    --------
    >>> from ads.data_labeling import MetadataReader
    >>> import oci
    >>> import os
    >>> from ads.common import auth as authutil
    >>> reader = MetadataReader.from_export_file("metadata_export_file_path",
    ...                                 auth=authutil.api_keys())
    >>> reader.read()
    """

    def __init__(self, reader: Reader):
        """Initiate a MetadataReader instance.

        Parameters
        ----------
        reader: Reader
            Reader instance which reads and extracts the labeled dataset metadata.
        """
        self._reader = reader

    @classmethod
    def from_export_file(cls, path: str, auth: Dict = None) -> "MetadataReader":
        """Contructs a MetadataReader instance.

        Parameters
        ----------
        path: str
            metadata file path, can be either local or object storage path.
        auth: (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        MetadataReader
            The MetadataReader instance whose reader is a ExportMetadataReader instance.
        """
        return cls(ExportMetadataReader(path=path, auth=auth))

    @classmethod
    def from_DLS(
        cls, dataset_id: str, compartment_id: str = None, auth: dict = None
    ) -> "MetadataReader":
        """Contructs a MetadataReader instance.

        Parameters
        ----------
        dataset_id: str
            The dataset OCID.
        compartment_id: (str, optional). Default None
            The compartment OCID of the dataset.
        auth: (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        MetadataReader
            The MetadataReader instance whose reader is a DLSMetadataReader instance.
        """
        if compartment_id is None:
            compartment_id = NB_SESSION_COMPARTMENT_OCID or JOB_RUN_COMPARTMENT_OCID

        if not compartment_id:
            raise ValueError("The `compartment_id` must be provided.")

        auth = auth or auth_util.default_signer()

        return cls(
            DLSMetadataReader(
                dataset_id=dataset_id, compartment_id=compartment_id, auth=auth
            )
        )

    def read(self) -> Metadata:
        """Reads the content from the metadata file.

        Returns
        -------
        Metadata
            The metadata of the labeled dataset.
        """
        return self._reader.read()


class DLSMetadataReader(Reader):
    """DLSMetadataReader class which reads the metadata jsonl file from the cloud."""

    def __init__(self, dataset_id: str, compartment_id: str, auth: dict):
        """Initializes the DLS metadata reader instance.

        Parameters
        ----------
        dataset_id: str
            The dataset OCID.
        compartment_id: str
            The compartment OCID of the dataset.
        auth: dict
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Raises
        ------
            ValueError: When dataset_id is empty or not a string.
            TypeError: When dataset_id not a string.
        """
        if not dataset_id:
            raise ValueError("The dataset OCID must be specified.")

        if not isinstance(dataset_id, str):
            raise TypeError("The dataset_id must be a string.")
        self.dataset_id = dataset_id
        self.compartment_id = compartment_id

        self.dls_dp_client = oci_client.OCIClientFactory(**auth).data_labeling_dp

    def read(self) -> Metadata:
        """Reads the content from the metadata file.

        Returns
        -------
        Metadata
            The metadata of the labeled dataset.

        Raises
        ------
        DatasetNotFoundError
            If dataset not found.
        ReadDatasetError
            If any error occured in attempt to read dataset.
        """
        try:
            dataset_response = self.dls_dp_client.get_dataset(self.dataset_id)
        except ServiceError as service_err:
            if service_err.status == 404:
                raise DatasetNotFoundError(self.dataset_id)
            raise ReadDatasetError(self.dataset_id)
        return Metadata.from_dls_dataset(dataset_response.data)


class ExportMetadataReader(JsonlReader):
    """ExportMetadataReader class which reads the metadata jsonl file from local/object
    storage path."""

    def read(self) -> Metadata:
        """Reads the content from the metadata file.

        Returns
        -------
        Metadata
            The metadata of the labeled dataset.
        """

        try:
            return MetadataParser.parse(next(super().read()))
        except StopIteration:
            raise EmptyMetadata(
                "The dataset metadata file is invalid. It appears to be empty. "
                "Use the `DataLabeling.export()` method to create a new dataset metadata file."
            )
        except Exception as e:
            raise e
