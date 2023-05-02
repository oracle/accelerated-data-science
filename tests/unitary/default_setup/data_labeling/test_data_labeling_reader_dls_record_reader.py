#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import MagicMock, patch

import pytest
from ads.data_labeling.reader.dls_record_reader import (
    OCIRecordSummary,
    ReadAnnotationsError,
    ReadRecordsError,
)
from ads.data_labeling.reader.record_reader import DLSRecordReader
from oci.data_labeling_service_dataplane.models.annotation_summary import (
    AnnotationSummary,
)
from oci.data_labeling_service_dataplane.models.record_summary import RecordSummary
from oci.exceptions import ServiceError
from oci.response import Response


class TestDLSRecordReader:
    """Unit tests for the DLSRecordReader class."""

    dataset_id = "test_dataset_id"
    compartment_id = "test_compartment_id"

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_init_success(self, mock_client, mock_signer):
        """Ensures initializing dls record reader pass when valid input parameters provided."""

        record_reader = DLSRecordReader(
            compartment_id=self.compartment_id,
            dataset_id=self.dataset_id,
        )
        assert record_reader.compartment_id == self.compartment_id
        assert record_reader.dataset_id == self.dataset_id

    def test_init_fail(self):
        """Ensures initializing dls record reader fails in case wrong input parameters provided."""
        with pytest.raises(ValueError, match="The dataset OCID must be specified."):
            DLSRecordReader(
                dataset_id="",
                compartment_id=self.compartment_id,
            )

        with pytest.raises(TypeError, match="The dataset_id must be a string."):
            DLSRecordReader(
                dataset_id=MagicMock(),
                compartment_id=self.compartment_id,
            )

        with pytest.raises(ValueError, match="The compartment OCID must be specified."):
            DLSRecordReader(
                dataset_id=self.dataset_id,
                compartment_id="",
            )

        with pytest.raises(TypeError, match="The compartment OCID must be a string."):
            DLSRecordReader(
                dataset_id=self.dataset_id,
                compartment_id=MagicMock(),
            )

    @property
    def generate_list_records_response_data(self):
        entity_item = {
            "compartment_id": self.compartment_id,
            "dataset_id": self.dataset_id,
            "defined_tags": {},
            "freeform_tags": {},
            "id": "test_record_id",
            "is_labeled": True,
            "lifecycle_state": "ACTIVE",
            "name": "single_label_text_datatext770_0.txt",
            "time_created": "2021-10-02T00:25:39.240000+00:00",
            "time_updated": "2021-10-02T00:31:53.681000+00:00",
        }

        list_records_response = RecordSummary(**entity_item)
        return list_records_response

    @property
    def generate_list_annotations_response_data(self):
        entity_item = {
            "compartment_id": self.compartment_id,
            "defined_tags": None,
            "freeform_tags": None,
            "id": "test_id",
            "lifecycle_state": "ACTIVE",
            "record_id": "test_record_id",
            "time_created": "2021-10-02T00:34:41.043000+00:00",
            "time_updated": "2021-10-02T00:34:41.043000+00:00",
        }

        list_annotations_response = AnnotationSummary(**entity_item)
        return list_annotations_response

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test__read_records(self, mock_client, mock_signer):
        record_reader = DLSRecordReader(
            compartment_id="test_compartment_id",
            dataset_id="test_dataset_id",
        )
        record_reader.dls_dp_client = MagicMock()
        record_reader.dls_dp_client.list_records = MagicMock(
            __name__="list_records",
            return_value=Response(
                data=[self.generate_list_records_response_data],
                status=None,
                headers=None,
                request=None,
            ),
        )

        with patch(
            "ads.data_labeling.reader.dls_record_reader.list_call_get_all_results",
            record_reader.dls_dp_client.list_records,
        ) as list_call_get_all_results:
            test_result = record_reader._read_records()
            list_call_get_all_results.assert_called_with(
                record_reader.dls_dp_client.list_records,
                record_reader.compartment_id,
                record_reader.dataset_id,
                lifecycle_state="ACTIVE",
            )
            assert test_result == [self.generate_list_records_response_data]

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test__read_records_fail(self, mock_client, mock_signer):
        record_reader = DLSRecordReader(
            compartment_id="test_compartment_id",
            dataset_id="test_dataset_id",
        )
        record_reader.dls_dp_client = MagicMock()
        record_reader.dls_dp_client.list_records = MagicMock(
            __name__="list_records",
            side_effect=ServiceError(
                status=404, code="code", message="message", headers={}
            ),
        )
        with pytest.raises(ReadRecordsError):
            record_reader._read_records()

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test__read_annotations(self, mock_client, mock_signer):
        record_reader = DLSRecordReader(
            compartment_id="test_compartment_id",
            dataset_id="test_dataset_id",
        )
        record_reader.dls_dp_client = MagicMock()
        record_reader.dls_dp_client.list_annotations = MagicMock(
            __name__="list_annotation",
            return_value=Response(
                data=[self.generate_list_annotations_response_data],
                status=None,
                headers=None,
                request=None,
            ),
        )

        with patch(
            "ads.data_labeling.reader.dls_record_reader.list_call_get_all_results",
            record_reader.dls_dp_client.list_annotations,
        ) as list_call_get_all_results:
            test_result = record_reader._read_annotations()
            list_call_get_all_results.assert_called_with(
                record_reader.dls_dp_client.list_annotations,
                record_reader.compartment_id,
                record_reader.dataset_id,
                lifecycle_state="ACTIVE",
            )
            assert test_result == [self.generate_list_annotations_response_data]

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test__read_annotations_fail(self, mock_client, mock_signer):
        record_reader = DLSRecordReader(
            compartment_id="test_compartment_id",
            dataset_id="test_dataset_id",
        )
        record_reader.dls_dp_client = MagicMock()
        record_reader.dls_dp_client.list_annotations = MagicMock(
            __name__="list_annotation",
            side_effect=ServiceError(
                status=404, code="code", message="message", headers={}
            ),
        )
        with pytest.raises(ReadAnnotationsError):
            record_reader._read_annotations()

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_read(self, mock_client, mock_signer):
        """Tests reading the labeled dataset records from dls."""

        record_reader = DLSRecordReader(
            compartment_id="test_compartment_id",
            dataset_id="test_dataset_id",
        )
        record_reader.dls_dp_client = MagicMock()
        record_reader._read_records = MagicMock(
            return_value=[self.generate_list_records_response_data]
        )
        record_reader._read_annotations = MagicMock(
            return_value=[self.generate_list_annotations_response_data]
        )
        tests_result = [record for record in record_reader.read()]

        record_reader._read_records.assert_called_once()
        record_reader._read_annotations.assert_called_once()

        assert isinstance(tests_result[0], OCIRecordSummary)
        assert tests_result[0].record.compartment_id == self.compartment_id
        assert tests_result[0].record.dataset_id == self.dataset_id
        assert tests_result[0].record.id == "test_record_id"
        assert tests_result[0].record.lifecycle_state == "ACTIVE"
        assert isinstance(tests_result[0].annotation, list)
