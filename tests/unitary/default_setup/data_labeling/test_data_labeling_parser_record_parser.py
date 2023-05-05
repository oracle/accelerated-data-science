#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest

from ads.data_labeling.constants import Formats
from ads.data_labeling.parser.export_record_parser import (
    BoundingBoxRecordParser,
    MultiLabelRecordParser,
    NERRecordParser,
    SingleLabelRecordParser,
)


class TestRecordParser:

    ner_record = {
        "id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
        "timeCreated": "2021-08-16 09:11:14",
        "sourceDetails": {"sourceType": "OBJECT_STORAGE", "path": "text/2_jd.txt"},
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
                "timeCreated": "2021-08-16 09:13:41",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {
                        "entityType": "TEXTSELECTION",
                        "labels": [{"label_name": "yes"}],
                        "textSpan": {"offset": 64, "length": 9},
                    },
                    {
                        "entityType": "TEXTSELECTION",
                        "labels": [{"label_name": "yes"}],
                        "textSpan": {"offset": 114, "length": 14},
                    },
                    {
                        "entityType": "TEXTSELECTION",
                        "labels": [{"label_name": "no"}],
                        "textSpan": {"offset": 188, "length": 12},
                    },
                    {
                        "entityType": "TEXTSELECTION",
                        "labels": [{"label_name": "no"}],
                        "textSpan": {"offset": 230, "length": 25},
                    },
                ],
            }
        ],
    }

    invalid_ner_record_1 = {
        "id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
        "timeCreated": "2021-08-16 09:11:14",
        "sourceDetails": {"sourceType": "OBJECT_STORAGE", "path": "text/2_jd.txt"},
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
                "timeCreated": "2021-08-16 09:13:41",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {"entityType": "GENERIC", "labels": [{"label_name": "yes"}]}
                ],
            }
        ],
    }

    invalid_ner_record_2 = {
        "id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
        "timeCreated": "2021-08-16 09:11:14",
        "sourceDetails": {"sourceType": "OBJECT_STORAGE", "path": "text/2_jd.jpg"},
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
                "timeCreated": "2021-08-16 09:13:41",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {"entityType": "TEXTSELECTION", "labels": [{"label_name": "yes"}]}
                ],
            }
        ],
    }

    single_label_record = {
        "id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
        "timeCreated": "2021-08-16 09:11:06",
        "sourceDetails": {"sourceType": "OBJECT_STORAGE", "path": "text/2_jd.txt"},
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
                "timeCreated": "2021-08-16 09:11:55",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {"entityType": "GENERIC", "labels": [{"label_name": "no"}]}
                ],
            }
        ],
    }

    invalid_single_label_record_1 = {
        "id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
        "timeCreated": "2021-08-16 09:11:06",
        "sourceDetails": {"sourceType": "OBJECT_STORAGE", "path": "text/2_jd.txt"},
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
                "timeCreated": "2021-08-16 09:11:55",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {"entityType": "TEXTSELECTION", "labels": [{"label_name": "no"}]}
                ],
            }
        ],
    }

    invalid_single_label_record_2 = {
        "id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
        "timeCreated": "2021-08-16 09:11:06",
        "sourceDetails": {"sourceType": "OBJECT_STORAGE", "path": "text/2_jd.txt"},
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
                "timeCreated": "2021-08-16 09:11:55",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {
                        "entityType": "GENERIC",
                        "labels": [{"label_name": "no"}, {"label_name": "no"}],
                    }
                ],
            }
        ],
    }

    multi_label_record = {
        "id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
        "timeCreated": "2021-08-17 04:01:40",
        "sourceDetails": {"sourceType": "OBJECT_STORAGE", "path": "text/2_jd.txt"},
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
                "timeCreated": "2021-08-17 04:37:17",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {
                        "entityType": "GENERIC",
                        "labels": [{"label_name": "class3"}, {"label_name": "class2"}],
                    }
                ],
            }
        ],
    }

    invalid_multi_label_record_1 = {
        "id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
        "timeCreated": "2021-08-17 04:01:40",
        "sourceDetails": {"sourceType": "OBJECT_STORAGE", "path": "text/2_jd.txt"},
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
                "timeCreated": "2021-08-17 04:37:17",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {
                        "entityType": "IMAGE",
                        "labels": [{"label_name": "class3"}, {"label_name": "class2"}],
                    }
                ],
            }
        ],
    }

    invalid_multi_label_record_2 = {
        "id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
        "timeCreated": "2021-08-17 04:01:40",
        "sourceDetails": {"sourceType": "OBJECT_STORAGE", "path": "text/2_jd.txt"},
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.iad.<unique_ocid>",
                "timeCreated": "2021-08-17 04:37:17",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {
                        "entityType": "GENERIC",
                        "labels": [],
                    }
                ],
            }
        ],
    }

    bounding_box_record = {
        "id": "ocid1.datalabelingrecord.oc1.phx.<unique_ocid>",
        "timeCreated": "2021-06-24 05:35:08",
        "sourceDetails": {
            "sourceType": "OBJECT_STORAGE",
            "path": "277_DLS_IMG_3137_jpeg_jpg.rf.f066fa3a50b9e2b392c9889bc3caaff8.jpg",
        },
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.phx.<unique_ocid>",
                "timeCreated": "2021-07-05 04:25:22",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {
                        "entityType": "IMAGEOBJECTSELECTION",
                        "labels": [{"label_name": "puffin"}],
                        "boundingPolygon": {
                            "normalizedVertices": [
                                {"x": "0.31292516", "y": "0.4461894"},
                                {"x": "0.31292516", "y": "0.7191486"},
                                {"x": "0.6564626", "y": "0.7191486"},
                                {"x": "0.6564626", "y": "0.4461894"},
                            ]
                        },
                    }
                ],
            }
        ],
    }

    invalid_bounding_box_record_1 = {
        "id": "ocid1.datalabelingrecord.oc1.phx.<unique_ocid>",
        "timeCreated": "2021-06-24 05:35:08",
        "sourceDetails": {
            "sourceType": "OBJECT_STORAGE",
            "path": "277_DLS_IMG_3137_jpeg_jpg.rf.f066fa3a50b9e2b392c9889bc3caaff8.jpg",
        },
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.phx.<unique_ocid>",
                "timeCreated": "2021-07-05 04:25:22",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {
                        "entityType": "TEXT",
                        "labels": [{"label_name": "puffin"}],
                        "boundingPolygon": {
                            "normalizedVertices": [
                                {"x": "0.31292516", "y": "0.4461894"},
                                {"x": "0.31292516", "y": "0.7191486"},
                                {"x": "0.6564626", "y": "0.7191486"},
                                {"x": "0.6564626", "y": "0.4461894"},
                            ]
                        },
                    }
                ],
            }
        ],
    }

    invalid_bounding_box_record_2 = {
        "id": "ocid1.datalabelingrecord.oc1.phx.<unique_ocid>",
        "timeCreated": "2021-06-24 05:35:08",
        "sourceDetails": {
            "sourceType": "OBJECT_STORAGE",
            "path": "277_DLS_IMG_3137_jpeg_jpg.rf.f066fa3a50b9e2b392c9889bc3caaff8.txt",
        },
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.phx.<unique_ocid>",
                "timeCreated": "2021-07-05 04:25:22",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {
                        "entityType": "IMAGEOBJECTSELECTION",
                        "labels": [{"label_name": "puffin"}],
                        "boundingPolygon": {
                            "normalizedVertices": [
                                {"x": "0.31292516", "y": "0.4461894"},
                                {"x": "0.31292516", "y": "0.7191486"},
                                {"x": "0.6564626", "y": "0.7191486"},
                                {"x": "0.6564626", "y": "0.4461894"},
                            ]
                        },
                    }
                ],
            }
        ],
    }

    invalid_record_1 = {
        "id": "ocid1.datalabelingrecord.oc1.phx.<unique_ocid>",
        "timeCreated": "2021-06-24 05:35:08",
        "sourceDetails": {
            "sourceType": "OBJECT_STORAGE",
            "path": "277_DLS_IMG_3137_jpeg_jpg.rf.f066fa3a50b9e2b392c9889bc3caaff8.txt",
        },
        "annotation": [
            {
                "id": "ocid1.datalabelingannotation.oc1.phx.<unique_ocid>",
                "timeCreated": "2021-07-05 04:25:22",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": [
                    {
                        "label": [{"label_name": "puffin"}],
                    }
                ],
            }
        ],
    }

    invalid_record_2 = {
        "id": "ocid1.datalabelingrecord.oc1.phx.<unique_ocid>",
        "timeCreated": "2021-06-24 05:35:08",
        "sourceDetails": {
            "sourceType": "OBJECT_STORAGE",
            "path": "277_DLS_IMG_3137_jpeg_jpg.rf.f066fa3a50b9e2b392c9889bc3caaff8.txt",
        },
        "annotations": [
            {
                "id": "ocid1.datalabelingannotation.oc1.phx.<unique_ocid>",
                "timeCreated": "2021-07-05 04:25:22",
                "createdBy": "ocid1.user.oc1..<unique_ocid>",
                "entities": {},
            }
        ],
    }

    def test_ner_parser(self):
        ner_item = NERRecordParser("local_path").parse(self.ner_record)
        assert isinstance(ner_item.annotation, list)
        assert ner_item.annotation[0].label == "yes"
        assert ner_item.annotation[0].offset == 64
        assert ner_item.annotation[0].length == 9

        assert ner_item.annotation[1].label == "yes"
        assert ner_item.annotation[1].offset == 114
        assert ner_item.annotation[1].length == 14

        assert ner_item.annotation[2].label == "no"
        assert ner_item.annotation[2].offset == 188
        assert ner_item.annotation[2].length == 12

        assert ner_item.annotation[3].label == "no"
        assert ner_item.annotation[3].offset == 230
        assert ner_item.annotation[3].length == 25

    def test_ner_parser_spacy(self):
        ner_item = NERRecordParser(
            dataset_source_path="local_path", format=Formats.SPACY
        ).parse(self.ner_record)
        assert isinstance(ner_item.annotation, list)
        assert ner_item.path == "local_pathtext/2_jd.txt"
        assert ner_item.annotation == [
            (64, 73, "yes"),
            (114, 128, "yes"),
            (188, 200, "no"),
            (230, 255, "no"),
        ]

    def test_ner_validator(self):
        assert NERRecordParser("local_path/")._validate(self.ner_record) == None

    @pytest.mark.parametrize(
        "test_options_ner",
        [invalid_ner_record_1, invalid_ner_record_2],
    )
    def test__ner_validator_fail(self, test_options_ner):
        """Ensures validation fails in case of invalid input parameters."""
        with pytest.raises(ValueError):
            NERRecordParser("local_path/")._validate(test_options_ner)

    def test_single_label_parser(self):
        single_item = SingleLabelRecordParser("local_path/").parse(
            self.single_label_record
        )

        assert single_item.path == "local_path/text/2_jd.txt"
        assert single_item.annotation == "no"

    def test_single_label_validator(self):
        assert (
            SingleLabelRecordParser("local_path")._validate(self.single_label_record)
            == None
        )
        assert (
            SingleLabelRecordParser("local_path/")._validate(self.single_label_record)
            == None
        )

    @pytest.mark.parametrize(
        "test_options_single_label",
        [invalid_single_label_record_1, invalid_single_label_record_2],
    )
    def test__single_label_validator_fail(self, test_options_single_label):
        """Ensures validation fails in case of invalid input parameters."""
        with pytest.raises(ValueError):
            SingleLabelRecordParser("local_path/")._validate(test_options_single_label)

    def test_multi_label_parser(self):
        multi_item = MultiLabelRecordParser("local_path/").parse(
            self.multi_label_record
        )
        assert multi_item.path == "local_path/text/2_jd.txt"
        assert multi_item.annotation == ["class3", "class2"]

    def test_multi_label_validator(self):
        assert (
            MultiLabelRecordParser("local_path/")._validate(self.multi_label_record)
            == None
        )

    @pytest.mark.parametrize(
        "test_options_multi_label",
        [invalid_multi_label_record_1, invalid_multi_label_record_2],
    )
    def test__multi_label_validator_fail(self, test_options_multi_label):
        """Ensures validation fails in case of invalid input parameters."""
        with pytest.raises(ValueError):
            MultiLabelRecordParser("local_path/")._validate(test_options_multi_label)

    def test_bounding_box_parser(self):
        bounding_box_item = BoundingBoxRecordParser("local_path/").parse(
            self.bounding_box_record
        )
        assert isinstance(bounding_box_item.annotation, list)
        assert (
            bounding_box_item.path
            == "local_path/277_DLS_IMG_3137_jpeg_jpg.rf.f066fa3a50b9e2b392c9889bc3caaff8.jpg"
        )
        assert bounding_box_item.annotation[0].labels == ["puffin"]
        assert bounding_box_item.annotation[0].top_left == (
            0.31292516,
            0.4461894,
        )
        assert bounding_box_item.annotation[0].bottom_left == (
            0.31292516,
            0.7191486,
        )
        assert bounding_box_item.annotation[0].bottom_right == (
            0.6564626,
            0.7191486,
        )
        assert bounding_box_item.annotation[0].top_right == (
            0.6564626,
            0.4461894,
        )

    def test_bounding_box_parser_yolo(self):
        bounding_box_item = BoundingBoxRecordParser(
            dataset_source_path="local_path/",
            format=Formats.YOLO,
            categories=["puffin"],
        ).parse(self.bounding_box_record)
        assert bounding_box_item.annotation == [
            [(0, 0.48469388, 0.582669, 0.34353744, 0.2729592)]
        ]
        assert (
            bounding_box_item.path
            == "local_path/277_DLS_IMG_3137_jpeg_jpg.rf.f066fa3a50b9e2b392c9889bc3caaff8.jpg"
        )

    def test_bounding_box_validator(self):
        assert (
            BoundingBoxRecordParser("local_path/")._validate(self.bounding_box_record)
            == None
        )

    @pytest.mark.parametrize(
        "test_options_image_label",
        [invalid_bounding_box_record_1, invalid_bounding_box_record_2],
    )
    def test__image_label_validator_fail(self, test_options_image_label):
        """Ensures validation fails in case of invalid input parameters."""
        with pytest.raises(ValueError):
            BoundingBoxRecordParser("local_path/")._validate(test_options_image_label)

    @pytest.mark.parametrize(
        "parser",
        [(SingleLabelRecordParser), (MultiLabelRecordParser), (NERRecordParser)],
    )
    def test_text_no_label(self, parser):
        json_record = {
            "id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
            "timeCreated": "2021-08-25 10:21:06",
            "sourceDetails": {"sourceType": "OBJECT_STORAGE", "path": "text385_1.txt"},
        }
        record = parser("fake_path/").parse(json_record)
        assert record.path == "fake_path/text385_1.txt"
        assert record.content is None
        assert record.annotation is None

    @pytest.mark.parametrize(
        "parser",
        [(SingleLabelRecordParser), (MultiLabelRecordParser), (NERRecordParser)],
    )
    def test_image_no_label(self, parser):
        json_record = {
            "id": "ocid1.datalabelingrecord.oc1.iad.<unique_ocid>",
            "timeCreated": "2021-08-25 10:21:06",
            "sourceDetails": {"sourceType": "OBJECT_STORAGE", "path": "image.jpg"},
        }
        record = parser("fake_path/").parse(json_record)
        assert record.path == "fake_path/image.jpg"
        assert record.content is None
        assert record.annotation is None

    @pytest.mark.parametrize(
        "test_options_invalid_format",
        [
            {},
            {"annotations": ""},
            {"annotations": [{"key": ""}]},
            {"annotations": [{"entities": ""}]},
            {"annotations": [{"entities": [{"key": ""}]}]},
            {"annotations": [{"entities": [{"entityType": ""}]}]},
            {"annotations": [{"entities": [{"labels": ""}]}]},
        ],
    )
    def test_invalid_record(self, test_options_invalid_format):
        """Ensures validation fails in case of invalid input parameters."""
        with pytest.raises(
            ValueError,
            match=r".*At least one record is in the wrong format.*",
        ):
            MultiLabelRecordParser("local_path")._validate(test_options_invalid_format)
