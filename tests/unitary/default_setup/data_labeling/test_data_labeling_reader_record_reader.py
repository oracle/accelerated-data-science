#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from unittest.mock import MagicMock, patch

import pytest
from ads.data_labeling.loader.file_loader import FileLoaderFactory
from ads.data_labeling.parser.export_record_parser import RecordParserFactory
from ads.data_labeling.reader.export_record_reader import ExportRecordReader
from ads.data_labeling.reader.record_reader import RecordReader


class TestRecordReader:
    image_sl_path = "./data_label_test_files/image_single_label_records.jsonl"
    image_ml_path = "./data_label_test_files/image_multi_label_records.jsonl"
    image_od_path = "./data_label_test_files/image_object_detection_records.jsonl"

    text_sl_path = "./data_label_test_files/text_single_label_records.jsonl"
    text_ml_path = "./data_label_test_files/text_multi_label_records.jsonl"
    text_ner_path = "./data_label_test_files/text_ner_records.jsonl"

    @patch("ads.common.auth.default_signer")
    def test_read_not_supported_annotation_type(self, mock_signer):
        dataset_type = "IMAGE"
        annotation_type = "NOT_SUPPORTED"
        with pytest.raises(ValueError):
            RecordReader.from_export_file(
                self.image_od_path,
                dataset_type,
                annotation_type,
                "local_path",
            )

    @patch("ads.common.auth.default_signer")
    def test_read_not_supported_dataset_type(self, mock_signer):
        dataset_type = "NOT_SUPPORTED"
        annotation_type = "SINGLE_LABEL"
        with pytest.raises(ValueError):
            RecordReader.from_export_file(
                self.image_od_path,
                dataset_type,
                annotation_type,
                "local_path",
            )

    @pytest.mark.parametrize(
        "dataset_type, annotation_type, file_path",
        [
            (
                "IMAGE",
                "SINGLE_LABEL",
                "./data_label_test_files/image_single_label_records.jsonl",
            ),
            (
                "IMAGE",
                "MULTI_LABEL",
                "./data_label_test_files/image_multi_label_records.jsonl",
            ),
            (
                "IMAGE",
                "BOUNDING_BOX",
                "./data_label_test_files/image_object_detection_records.jsonl",
            ),
            (
                "TEXT",
                "SINGLE_LABEL",
                "./data_label_test_files/text_single_label_records.jsonl",
            ),
            (
                "TEXT",
                "MULTI_LABEL",
                "./data_label_test_files/text_multi_label_records.jsonl",
            ),
            (
                "TEXT",
                "ENTITY_EXTRACTION",
                "./data_label_test_files/text_ner_records.jsonl",
            ),
        ],
    )
    @patch("ads.common.auth.default_signer")
    def test_read_load_false_include_false(
        self, mock_signer, dataset_type, annotation_type, file_path
    ):
        record_reader = RecordReader.from_export_file(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path),
            dataset_type,
            annotation_type,
            "local_path",
        )
        path, content, label = next(record_reader.read())
        if dataset_type == "IMAGE":
            assert isinstance(path, str) and path.split("/")[-1].endswith(".jpg")
        assert content is None
        if annotation_type == "SINGLE_LABEL":
            assert isinstance(label, str)
        elif annotation_type == "MULTI_LABEL":
            assert isinstance(label, list)
        elif annotation_type == "BOUNDING_BOX":
            assert isinstance(label, list)
        elif annotation_type == "ENTITY_EXTRACTION":
            assert isinstance(label, list)
        assert record_reader.include_unlabeled is False

    @patch("ads.common.auth.default_signer")
    def test_read_load_true_include_true(self, mock_signer):
        dataset_type = "TEXT"
        annotation_type = "ENTITY_EXTRACTION"
        file_path = "./data_label_test_files/text_ner_records.jsonl"
        record_reader = RecordReader.from_export_file(
            path=os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path),
            dataset_type=dataset_type,
            annotation_type=annotation_type,
            dataset_source_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data_label_test_files/"
            ),
            materialize=True,
            include_unlabeled=True,
        )
        data_iterator = record_reader.read()
        path, content, label = next(data_iterator)

        assert record_reader.include_unlabeled
        assert isinstance(path, str) and path.split("/")[-1].endswith(".txt")
        assert isinstance(content, str)
        assert isinstance(label, list)

        next(data_iterator)
        path, content, label = next(data_iterator)
        assert label is None

    def test_from_export_file(self):
        """Ensures that record reader can be instantiated using from_export_file method."""
        mock_auth = {"test_auth"}
        mock_categories = ["test_category"]
        with patch.object(
            ExportRecordReader, "__init__"
        ) as mock_export_record_reader_init:
            mock_export_record_reader_init.return_value = None
            with patch.object(RecordParserFactory, "parser") as mock_parser:
                mock_parser_instance = MagicMock()
                mock_parser.return_value = mock_parser_instance
                with patch.object(FileLoaderFactory, "loader") as mock_loader:
                    mock_loader_instance = MagicMock()
                    mock_loader.return_value = mock_loader_instance

                    mock_record_reader = RecordReader.from_export_file(
                        path="test_path",
                        dataset_type="test_dataset_type",
                        annotation_type="test_annotation_type",
                        dataset_source_path="test_dataset_source_path",
                        format="test_format",
                        categories=mock_categories,
                        auth=mock_auth,
                        materialize=True,
                        include_unlabeled=True,
                        includes_metadata=True,
                        encoding="utf-8",
                    )
                    mock_export_record_reader_init.assert_called_with(
                        path="test_path",
                        auth=mock_auth,
                        encoding="utf-8",
                        includes_metadata=True,
                    )
                    mock_parser.assert_called_with(
                        annotation_type="test_annotation_type",
                        dataset_source_path="test_dataset_source_path",
                        format="test_format",
                        categories=mock_categories,
                    )
                    mock_loader.assert_called_with(
                        dataset_type="test_dataset_type",
                        auth=mock_auth,
                    )
                    assert isinstance(mock_record_reader.reader, ExportRecordReader)
                    assert mock_record_reader.parser == mock_parser_instance
                    assert mock_record_reader.loader == mock_loader_instance
                    assert mock_record_reader.materialize == True
                    assert mock_record_reader.include_unlabeled == True
                    assert mock_record_reader.encoding == "utf-8"
