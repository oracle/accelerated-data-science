#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from ads.common import auth, oci_client
from ads.data_labeling.metadata import Metadata
from ads.data_labeling.reader.metadata_reader import (
    DLSMetadataReader,
    EmptyMetadata,
    ExportMetadataReader,
    MetadataReader,
    ReadDatasetError,
    DatasetNotFoundError,
)
from oci.data_labeling_service_dataplane.models.dataset import Dataset
from oci.data_labeling_service_dataplane.models.label_name import LabelName
from oci.data_labeling_service_dataplane.models.label_set import LabelSet
from oci.data_labeling_service_dataplane.models.object_storage_dataset_source_details import (
    ObjectStorageDatasetSourceDetails,
)
from oci.data_labeling_service_dataplane.models.text_dataset_format_details import (
    TextDatasetFormatDetails,
)
from oci.exceptions import ServiceError
from oci.response import Response


class TestMetadataReader:
    encoding = "utf-8"
    json_data = {
        "id": "ocid1.datalabelingdataset.oc1.iad.<unique_ocid>",
        "compartmentId": "ocid1.compartment.oc1.<unique_ocid>",
        "displayName": "document",
        "labelsSet": [{"name": "ads"}, {"name": "other"}],
        "annotationFormat": "SINGLE_LABEL",
        "datasetSourceDetails": {
            "namespace": "test_namespace",
            "bucket": "test_bucket",
            "prefix": "document/",
        },
        "datasetFormatDetails": {"formatType": "DOCUMENT"},
        "recordFiles": [
            {
                "namespace": "test_namespace",
                "bucket": "test_bucket",
                "path": "document/records_1631762769846.jsonl",
            }
        ],
    }

    @patch("ads.common.auth.default_signer")
    def test_from_export_file(self, mock_signer):
        reader = MetadataReader.from_export_file(
            path="local_path",
        )
        assert reader._reader.path == "local_path"

    @patch.object(DLSMetadataReader, "__init__", return_value=None)
    def test_from_dls(self, mock_init):
        test_params = {
            "dataset_id": "oci.xxxx.xxxx",
            "compartment_id": "oci.xxxx.xxxx",
            "auth": {"config": {}},
        }
        metadata_reader = MetadataReader.from_DLS(**test_params)

        mock_init.assert_called_with(
            dataset_id=test_params["dataset_id"],
            compartment_id=test_params["compartment_id"],
            auth=test_params["auth"],
        )

        assert isinstance(metadata_reader._reader, DLSMetadataReader)

    @patch("ads.common.auth.default_signer")
    @patch.object(ExportMetadataReader, "read")
    def test_read(self, mock_reader, mock_signer):
        mock_reader.return_value = self.json_data
        reader = MetadataReader.from_export_file(
            path="local_path",
        )
        assert reader.read() == self.json_data


class TestDLSMetadataReader:
    encoding = "utf-8"
    with patch.object(oci_client, "OCIClientFactory"):
        reader = DLSMetadataReader(
            dataset_id="oci.1234", compartment_id="oci.5678", auth={"config": {}}
        )

    def test_init(self):
        assert self.reader.dataset_id == "oci.1234"
        assert self.reader.compartment_id == "oci.5678"

    @property
    def generate_get_dataset_response_data(self):
        entity_item = {
            "annotation_format": "SINGLE_LABEL",
            "compartment_id": "ocid1.compartment.oc1.<unique_ocid>",
            "dataset_format_details": TextDatasetFormatDetails(format_type="TEXT"),
            "dataset_source_details": ObjectStorageDatasetSourceDetails(
                bucket="ads-dls-examples",
                namespace="test_namespace",
                prefix="text/src/single-label/",
                source_type="OBJECT_STORAGE",
            ),
            "defined_tags": {},
            "description": None,
            "display_name": "unit_test_text_single_label",
            "freeform_tags": {},
            "id": "ocid1.datalabelingdataset.oc1.<unique_ocid>",
            "initial_record_generation_configuration": None,
            "label_set": LabelSet(items=[LabelName(name="1"), LabelName(name="0")]),
            "lifecycle_details": None,
            "lifecycle_state": "ACTIVE",
            "system_tags": None,
            "time_created": "2021-10-02T00:17:03.550000+00:00",
            "time_updated": "2021-10-18T17:47:04.662000+00:00",
        }

        get_dataset_response = Dataset(**entity_item)
        return get_dataset_response

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_read(self, mock_client, mock_signer):
        metadata_reader = DLSMetadataReader(
            compartment_id="ocid.compartment.oc1.<unique_ocid>",
            dataset_id="ocid.dataset.oc1.<unique_ocid>",
            auth={"config": {}},
        )
        metadata_reader.dls_dp_client = MagicMock()
        metadata_reader.dls_dp_client.get_dataset = MagicMock(
            return_value=Response(
                data=self.generate_get_dataset_response_data,
                status=200,
                headers=None,
                request=None,
            )
        )

        metadata_read = metadata_reader.read()
        assert isinstance(metadata_read, Metadata)
        assert metadata_read.dataset_name == "unit_test_text_single_label"
        assert metadata_read.dataset_type == "TEXT"
        assert metadata_read.labels == ["1", "0"]
        assert (
            metadata_read.source_path
            == "oci://ads-dls-examples@test_namespace/text/src/single-label/"
        )
        assert metadata_read.compartment_id == "ocid1.compartment.oc1.<unique_ocid>"
        assert metadata_read.dataset_id == "ocid1.datalabelingdataset.oc1.<unique_ocid>"

    @pytest.mark.parametrize(
        "status, exception",
        [
            (404, DatasetNotFoundError),
            (503, ReadDatasetError),
        ],
    )
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_read_fail_dataset_not_found(self, mock_client, status, exception):
        metadata_reader = DLSMetadataReader(
            compartment_id="ocid.compartment.oc1.<unique_ocid>",
            dataset_id="ocid.dataset.oc1.<unique_ocid>",
            auth={"config": {}},
        )
        metadata_reader.dls_dp_client = MagicMock()
        metadata_reader.dls_dp_client.get_dataset = MagicMock(
            side_effect=ServiceError(
                status=status, code="code", message="message", headers={}
            )
        )

        with pytest.raises(exception):
            metadata_reader.read()


class TestExportMetadataReader:
    encoding = "utf-8"
    with patch.object(auth, "default_signer"):
        reader = ExportMetadataReader(path="local_path", encoding=encoding)
    json_data = b'{"id":"ocid1.datalabelingdataset.oc1.iad.<unique_ocid>","compartmentId":"ocid1.compartment.oc1..<unique_ocid>","displayName":"document","labelsSet":[{"name":"ads"},{"name":"other"}],"annotationFormat":"SINGLE_LABEL","datasetSourceDetails":{"namespace":"test_namespace","bucket":"test_bucket","prefix":"document/"},"datasetFormatDetails":{"formatType":"DOCUMENT"},"recordFiles":[{"namespace":"test_namespace","bucket":"test_bucket","path":"document/records_1631762769846.jsonl"}]}'

    def test_init(self):
        assert self.reader.path == "local_path"
        assert self.reader.encoding == self.encoding

    @patch("ads.common.auth.default_signer")
    def test_read(self, mock_signer):
        reader = ExportMetadataReader(
            path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data_label_test_files",
                "document_document_1631762769846.jsonl",
            ),
            encoding=self.encoding,
        )
        metadata = reader.read()
        json_metadata = json.loads(self.json_data.decode(self.encoding))
        assert metadata.dataset_id == json_metadata["id"]
        assert metadata.annotation_type == json_metadata["annotationFormat"]
        assert metadata.compartment_id == json_metadata["compartmentId"]
        assert (
            metadata.dataset_type == json_metadata["datasetFormatDetails"]["formatType"]
        )
        assert metadata.source_path == "oci://test_bucket@test_namespace/document/"
        assert (
            metadata.records_path
            == "oci://test_bucket@test_namespace/document/records_1631762769846.jsonl"
        )

    @patch("ads.common.auth.default_signer")
    def test_empty_metadata(self, mock_signer):
        reader = ExportMetadataReader(
            path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data_label_test_files",
                "empty.jsonl",
            ),
            encoding=self.encoding,
        )
        with pytest.raises(EmptyMetadata):
            metadata = reader.read()
