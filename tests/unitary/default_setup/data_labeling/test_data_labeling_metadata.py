#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import patch

from ads.data_labeling.parser.export_metadata_parser import MetadataParser


class TestMetadata:
    json_data = {
        "id": "ocid1.datalabelingdataset.oc1.iad.<unique_ocid>",
        "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
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
    multi_label_text_json_data = {
        "id": "ocid1.datalabelingdataset.oc1.iad.<unique_ocid>",
        "compartmentId": "ocid1.compartment.oc1..<unique_ocid>",
        "displayName": "text_multilabel",
        "labelsSet": [{"name": "class2"}, {"name": "class3"}, {"name": "class1"}],
        "annotationFormat": "MULTI_LABEL",
        "datasetSourceDetails": {
            "namespace": "test_namespace",
            "bucket": "test_bucket",
            "prefix": "text/",
        },
        "datasetFormatDetails": {"formatType": "TEXT"},
        "recordFiles": [
            {
                "namespace": "test_namespace",
                "bucket": "test_bucket",
                "path": "text/records_1629385025547.jsonl",
            }
        ],
    }
    multi_label_text_yaml_data = "'annotation_type: MULTI_LABEL\ncompartment_id: ocid1.compartment.oc1..<unique_ocid>\ndataset_id: ocid1.datalabelingdataset.oc1.iad.<unique_ocid>\ndataset_name: text_multilabel\ndataset_type: TEXT\nlabels:\n- class2\n- class3\n- class1\nrecords_path:\n  bucket: test_bucket\n  filepath: text/records_1629385025547.jsonl\n  namespace: test_namespace\nsource_path:\n  bucket: test_bucket\n  filepath: text/\n  namespace: test_namespace\n'"

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_from_jsonl(self, mock_client, mock_signer):
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

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_to_dict(self, mock_client, mock_signer):
        metadata = MetadataParser.parse(json_data=self.json_data)
        assert isinstance(metadata.to_dict(), dict)

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_to_yaml(self, mock_client, mock_signer):
        metadata = MetadataParser.parse(json_data=self.multi_label_text_json_data)
        assert isinstance(metadata.to_yaml(), str)
