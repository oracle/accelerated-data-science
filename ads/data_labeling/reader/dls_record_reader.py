#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Generator, List

from ads.common import auth as authutil
from ads.common import oci_client
from ads.data_labeling.interface.reader import Reader
from oci.data_labeling_service_dataplane.models import AnnotationSummary, RecordSummary
from oci.exceptions import ServiceError
from oci.pagination import list_call_get_all_results


class ReadRecordsError(Exception):   # pragma: no cover
    def __init__(self, dataset_id: str):
        super().__init__(
            f"Error occurred in attempt to read records of dataset {dataset_id}."
        )


class ReadAnnotationsError(Exception):   # pragma: no cover
    def __init__(self, dataset_id: str):
        super().__init__(
            f"Error occurred in attempt to read annotations of dataset {dataset_id}."
        )


@dataclass
class OCIRecordSummary:
    """The class that representing the labeled record in ADS format.

    Attributes
    ----------
    record: RecordSummary
        OCI RecordSummary.
    annotations: List[AnnotationSummary]
        List of OCI AnnotationSummary.
    """

    record: RecordSummary = None
    annotation: List[AnnotationSummary] = None


class DLSRecordReader(Reader):
    """DLS Record Reader Class that reads records from the cloud into ADS format."""

    def __init__(self, dataset_id: str, compartment_id: str, auth: dict = None):
        """Initiates a DLSRecordReader instance.

        Parameters
        ----------
        dataset_id: str
            The dataset OCID.
        compartment_id: str
            The compartment OCID of the dataset.
        auth: (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        """
        if not dataset_id:
            raise ValueError("The dataset OCID must be specified.")

        if not isinstance(dataset_id, str):
            raise TypeError("The dataset_id must be a string.")

        if not compartment_id:
            raise ValueError("The compartment OCID must be specified.")

        if not isinstance(compartment_id, str):
            raise TypeError("The compartment OCID must be a string.")

        auth = auth or authutil.default_signer()
        self.dataset_id = dataset_id
        self.compartment_id = compartment_id
        self.dls_dp_client = oci_client.OCIClientFactory(**auth).data_labeling_dp

    def _read_records(self):
        try:
            return list_call_get_all_results(
                self.dls_dp_client.list_records,
                self.compartment_id,
                self.dataset_id,
                lifecycle_state="ACTIVE",
            ).data
        except ServiceError:
            raise ReadRecordsError(self.dataset_id)

    def _read_annotations(self):
        try:
            return list_call_get_all_results(
                self.dls_dp_client.list_annotations,
                self.compartment_id,
                self.dataset_id,
                lifecycle_state="ACTIVE",
            ).data
        except ServiceError:
            raise ReadAnnotationsError(self.dataset_id)

    def read(self) -> Generator[OCIRecordSummary, Any, Any]:
        """Reads OCI records.

        Yields
        ------
        OCIRecordSummary
            The OCIRecordSummary instance.
        """
        records = self._read_records()
        annotations = self._read_annotations()

        annotations_map = defaultdict(list)
        for annotation in annotations:
            annotations_map[annotation.record_id].append(annotation)

        for record in records:
            yield OCIRecordSummary(record, annotations_map.get(record.id))
