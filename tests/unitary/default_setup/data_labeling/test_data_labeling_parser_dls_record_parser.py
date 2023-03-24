#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from unittest.mock import MagicMock, patch

from oci.data_labeling_service_dataplane.models.annotation_summary import (
    AnnotationSummary,
)
from oci.response import Response
from oci.data_labeling_service_dataplane.models.annotation import Annotation
from oci.data_labeling_service_dataplane.models.generic_entity import GenericEntity
from oci.data_labeling_service_dataplane.models.text_selection_entity import (
    TextSelectionEntity,
)
from oci.data_labeling_service_dataplane.models.text_span import TextSpan
from oci.data_labeling_service_dataplane.models.label import Label
from oci.data_labeling_service_dataplane.models.image_object_selection_entity import (
    ImageObjectSelectionEntity,
)
from oci.data_labeling_service_dataplane.models.bounding_polygon import BoundingPolygon
from oci.data_labeling_service_dataplane.models.normalized_vertex import (
    NormalizedVertex,
)

from ads.data_labeling.ner import NERItem
from ads.data_labeling.boundingbox import BoundingBoxItem
from ads.data_labeling.parser.dls_record_parser import (
    DLSBoundingBoxRecordParser,
    DLSMultiLabelRecordParser,
    DLSNERRecordParser,
    DLSSingleLabelRecordParser,
)


class TestDLSSingleLabelRecordParser:
    """Unit tests for the DLSSingleLabelRecordParser class."""

    dataset_id = "test_dataset_id"
    dataset_source_path = "test_datasource_path"
    compartment_id = "test_compartment_id"
    format = "test_format"

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_init(self, mock_client, mock_signer):
        """Ensures initializing dls single label record parser pass when valid input parameters provided."""

        dls_record_parser = DLSSingleLabelRecordParser(
            dataset_source_path=self.dataset_source_path,
            format=self.format,
        )
        assert dls_record_parser.dataset_source_path == self.dataset_source_path
        assert dls_record_parser.format == self.format

    @property
    def generate_get_annotation_summary_data(self):
        single_label_entity_item_annotation = {
            "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
            "defined_tags": None,
            "freeform_tags": None,
            "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
            "lifecycle_state": "ACTIVE",
            "record_id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
            "time_created": "2021-10-02T00:31:53.681000+00:00",
            "time_updated": "2021-10-02T00:31:53.681000+00:00",
        }
        get_annotation_response = AnnotationSummary(
            **single_label_entity_item_annotation
        )

        return get_annotation_response

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test__get_annotation_details(self, mock_client, mock_signer):
        single_label_dls_record_parser = DLSSingleLabelRecordParser(
            dataset_source_path=self.dataset_source_path,
            format=self.format,
        )
        single_label_dls_record_parser.dls_dp_client = MagicMock()
        single_label_dls_record_parser.dls_dp_client.get_annotation = MagicMock(
            return_value=Response(
                data=self.generate_get_annotation_summary_data,
                status=None,
                headers=None,
                request=None,
            )
        )
        res = single_label_dls_record_parser._get_annotation_details(
            [self.generate_get_annotation_summary_data]
        )
        assert res == [self.generate_get_annotation_summary_data]

    @property
    def generate_get_annotation_response_data(self):
        entity_item = {
            "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
            "created_by": "ocid1.user.oc1..<unique_ocid>",
            "defined_tags": None,
            "entities": [
                GenericEntity(
                    entity_type="GENERIC",
                    extended_metadata=None,
                    labels=[Label(label="0")],
                )
            ],
            "freeform_tags": None,
            "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
            "lifecycle_state": "ACTIVE",
            "record_id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
            "time_created": "2021-10-02T00:34:41.043000+00:00",
            "time_updated": "2021-10-02T00:34:41.043000+00:00",
            "updated_by": "ocid1.user.oc1..<unique_ocid>",
        }

        get_annotation_response = Annotation(**entity_item)

        return get_annotation_response

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test___extract_annotations(self, mock_client, mock_signer):
        oci_annotation = self.generate_get_annotation_response_data
        single_label_dls_record_parser = DLSSingleLabelRecordParser(
            dataset_source_path=self.dataset_source_path,
            format=self.format,
        )
        assert (
            single_label_dls_record_parser._extract_annotations(
                oci_annotation=[oci_annotation]
            )
            == "0"
        )


class TestDLSMultiLabelRecordParser:
    """Unit tests for the DLSMultiLabelRecordParser class."""

    dataset_id = "test_dataset_id"
    dataset_source_path = "test_datasource_path"
    compartment_id = "test_compartment_id"
    format = "test_format"

    @property
    def generate_get_annotation_response_data(self):
        entity_item = {
            "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
            "created_by": "ocid1.user.oc1..<unique_ocid>",
            "defined_tags": None,
            "entities": [
                GenericEntity(
                    entity_type="GENERIC",
                    extended_metadata=None,
                    labels=[
                        Label(label="coffee"),
                        Label(label="cocoa"),
                        Label(label="sugar"),
                    ],
                )
            ],
            "freeform_tags": None,
            "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
            "lifecycle_state": "ACTIVE",
            "record_id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
            "time_created": "2021-10-02T00:34:41.043000+00:00",
            "time_updated": "2021-10-02T00:34:41.043000+00:00",
            "updated_by": "ocid1.user.oc1..<unique_ocid>",
        }

        get_annotation_response = Annotation(**entity_item)

        return get_annotation_response

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test___extract_annotations(self, mock_client, mock_signer):
        oci_annotation = self.generate_get_annotation_response_data
        multi_label_dls_record_parser = DLSMultiLabelRecordParser(
            dataset_source_path=self.dataset_source_path,
            format=self.format,
        )
        assert multi_label_dls_record_parser._extract_annotations(
            oci_annotation=[oci_annotation]
        ) == ["coffee", "cocoa", "sugar"]


class TestDLSNERRecordParser:
    """Unit tests for the DLSNERRecordParser class."""

    dataset_id = "test_dataset_id"
    dataset_source_path = "test_datasource_path"
    compartment_id = "test_compartment_id"
    format = "test_format"

    @property
    def generate_get_annotation_response_data(self):
        entity_item = {
            "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
            "created_by": "ocid1.user.oc1..<unique_ocid>",
            "defined_tags": None,
            "entities": [
                TextSelectionEntity(
                    entity_type="TEXTSELECTION",
                    extended_metadata=None,
                    labels=[Label(label="city")],
                    text_span=TextSpan(length=8.0, offset=54.0),
                )
            ],
            "freeform_tags": None,
            "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
            "lifecycle_state": "ACTIVE",
            "record_id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
            "time_created": "2021-10-02T00:34:41.043000+00:00",
            "time_updated": "2021-10-02T00:34:41.043000+00:00",
            "updated_by": "ocid1.user.oc1..<unique_ocid>",
        }

        get_annotation_response = Annotation(**entity_item)

        return get_annotation_response

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test___extract_annotations(self, mock_client, mock_signer):
        oci_annotation = self.generate_get_annotation_response_data
        ner_label_dls_record_parser = DLSNERRecordParser(
            dataset_source_path=self.dataset_source_path,
            format=self.format,
        )
        assert ner_label_dls_record_parser._extract_annotations(
            oci_annotation=[oci_annotation]
        )[0] == NERItem(label="city", offset=54, length=8)


class TestDLSBoundingBoxRecordParser:
    """Unit tests for the DLSBoundingBoxRecordParser class."""

    dataset_id = "test_dataset_id"
    dataset_source_path = "test_datasource_path"
    compartment_id = "test_compartment_id"
    format = "test_format"

    @property
    def generate_get_annotation_response_data(self):
        entity_item = {
            "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
            "created_by": "ocid1.user.oc1..<unique_ocid>",
            "defined_tags": None,
            "entities": [
                ImageObjectSelectionEntity(
                    entity_type="IMAGEOBJECTSELECTION",
                    extended_metadata=None,
                    labels=[Label(label="not_fish")],
                    bounding_polygon=BoundingPolygon(
                        normalized_vertices=[
                            NormalizedVertex(x=0.041935485, y=0.49405572),
                            NormalizedVertex(x=0.041935485, y=0.49405572),
                            NormalizedVertex(x=0.041935485, y=0.49405572),
                            NormalizedVertex(x=0.041935485, y=0.49405572),
                        ]
                    ),
                )
            ],
            "freeform_tags": None,
            "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
            "lifecycle_state": "ACTIVE",
            "record_id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
            "time_created": "2021-10-02T00:34:41.043000+00:00",
            "time_updated": "2021-10-02T00:34:41.043000+00:00",
            "updated_by": "ocid1.user.oc1..<unique_ocid>",
        }

        get_annotation_response = Annotation(**entity_item)

        return get_annotation_response

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test___extract_annotations(self, mock_client, mock_signer):
        oci_annotation = self.generate_get_annotation_response_data
        object_detection_dls_record_parser = DLSBoundingBoxRecordParser(
            dataset_source_path=self.dataset_source_path,
            format=self.format,
        )
        assert object_detection_dls_record_parser._extract_annotations(
            oci_annotation=[oci_annotation]
        )[0] == BoundingBoxItem(
            top_left=(0.041935485, 0.49405572),
            bottom_left=(0.041935485, 0.49405572),
            bottom_right=(0.041935485, 0.49405572),
            top_right=(0.041935485, 0.49405572),
            labels=["not_fish"],
        )
