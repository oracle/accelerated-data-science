#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import patch

from ads.data_labeling.parser.export_metadata_parser import MetadataParser
import pytest


class TestMetadataParser:

    json_data = {
        "id": "ocid1.datalabelingdataset.oc1.<unique_ocid>",
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
    def test_parse(self, mock_signer):
        metadata = MetadataParser.parse(json_data=self.json_data)
        assert metadata.dataset_id == self.json_data["id"]
        assert metadata.annotation_type == self.json_data["annotationFormat"]
        assert metadata.compartment_id == self.json_data["compartmentId"]
        assert (
            metadata.dataset_type
            == self.json_data["datasetFormatDetails"]["formatType"]
        )
        assert metadata.source_path == "oci://test_bucket@test_namespace/document/"
        assert (
            metadata.records_path
            == "oci://test_bucket@test_namespace/document/records_1631762769846.jsonl"
        )

    def test__validate_success(self):
        assert MetadataParser._validate(json_data=self.json_data) is None

    @pytest.mark.parametrize(
        "test_metadata_invalid_format",
        [
            {},
            {
                "compartmentId": "",
                "displayName": "",
                "labelsSet": [],
                "annotationFormat": "",
                "datasetSourceDetails": {},
                "datasetFormatDetails": {},
                "recordFiles": [{}],
            },
            {
                "id": "",
                "displayName": "",
                "labelsSet": [],
                "annotationFormat": "",
                "datasetSourceDetails": {},
                "datasetFormatDetails": {},
                "recordFiles": [{}],
            },
            {
                "id": "",
                "compartmentId": "",
                "displayNames": "",
                "labelsSet": [],
                "annotationFormat": "",
                "datasetSourceDetails": {},
                "datasetFormatDetails": {},
                "recordFiles": [],
            },
            {
                "id": "",
                "compartmentId": "",
                "displayName": "",
                "labelsSet": {},
                "annotationFormat": "",
                "datasetSourceDetails": {},
                "datasetFormatDetails": {},
                "recordFiles": [],
            },
            {
                "id": "",
                "compartmentId": "",
                "displayName": "",
                "labelsSet": [],
                "datasetSourceDetails": {},
                "datasetFormatDetails": {},
                "recordFiles": [],
            },
            {
                "id": "",
                "compartmentId": "",
                "displayName": "",
                "labelsSet": [],
                "annotationFormat": "]",
                "datasetFormatDetails": {},
                "recordFiles": [],
            },
            {
                "id": "",
                "compartmentId": "",
                "displayName": "",
                "labelsSet": [],
                "annotationFormat": "",
                "datasetSourceDetails": {},
                "datasetFormatDetails": "",
                "recordFiles": [],
            },
            {
                "id": "",
                "compartmentId": "",
                "displayName": "",
                "labelsSet": [],
                "annotationFormat": "",
                "datasetSourceDetails": {},
                "datasetFormatDetails": {},
                "recordFiles": {},
            },
        ],
    )
    def test__validate_no_success(self, test_metadata_invalid_format):
        """Ensures validation fails in case of invalid input parameters."""
        with pytest.raises(ValueError):
            MetadataParser._validate(test_metadata_invalid_format)
