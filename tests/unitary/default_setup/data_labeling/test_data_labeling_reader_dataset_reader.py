#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for labeled dataset reader module. Includes tests for:
 - DataSetReader
 - ExportReader
"""

from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ads.data_labeling import Metadata
from ads.data_labeling.reader.dataset_reader import (
    _LABELED_DF_COLUMNS,
    DLSDatasetReader,
    ExportReader,
    LabeledDatasetReader,
)
from ads.data_labeling.reader.metadata_reader import (
    DLSMetadataReader,
    ExportMetadataReader,
    MetadataReader,
)
from ads.data_labeling.reader.record_reader import RecordReader
from oci.data_labeling_service_dataplane.models.dataset import Dataset
from oci.data_labeling_service_dataplane.models.label_name import LabelName
from oci.data_labeling_service_dataplane.models.label_set import LabelSet
from oci.data_labeling_service_dataplane.models.object_storage_dataset_source_details import (
    ObjectStorageDatasetSourceDetails,
)
from oci.data_labeling_service_dataplane.models.text_dataset_format_details import (
    TextDatasetFormatDetails,
)
from oci.response import Response


class TestLabeledDatasetReader:
    """Unittests for the LabeledDatasetReader class."""

    @patch.object(ExportReader, "__init__", return_value=None)
    def test_from_export(self, mock_init):
        """Tests constructing Labeled Dataset Reader instance."""

        test_params = {
            "path": "path",
            "auth": {"config": {}},
            "encoding": "encoding",
            "materialize": False,
            "include_unlabeled": True,
        }

        ldr = LabeledDatasetReader.from_export(**test_params)

        mock_init.assert_called_with(
            path=test_params["path"],
            auth=test_params["auth"],
            encoding=test_params["encoding"],
            materialize=test_params["materialize"],
            include_unlabeled=test_params["include_unlabeled"],
        )

        assert isinstance(ldr._reader, ExportReader)

    @patch.object(DLSDatasetReader, "__init__", return_value=None)
    def test_from_DLS(self, mock_init):
        test_params = {
            "dataset_id": "oci.xxxx.xxxx",
            "compartment_id": "oci.xxxx.xxxx",
            "auth": {"config": {}},
            "encoding": "encoding",
            "materialize": False,
            "include_unlabeled": True,
        }

        ldr = LabeledDatasetReader.from_DLS(**test_params)

        mock_init.assert_called_with(
            dataset_id=test_params["dataset_id"],
            compartment_id=test_params["compartment_id"],
            auth=test_params["auth"],
            encoding=test_params["encoding"],
            materialize=test_params["materialize"],
            include_unlabeled=test_params["include_unlabeled"],
        )

        assert isinstance(ldr._reader, DLSDatasetReader)

    def test_info(self):
        """Tests getting labeled dataset metadata."""
        mock_reader = MagicMock()
        mock_reader.info = MagicMock()

        ldr = LabeledDatasetReader(reader=mock_reader)
        ldr.info()
        mock_reader.info.assert_called_once()

    @patch.object(LabeledDatasetReader, "_bulk_read")
    def test_read(self, mock__bulk_read):
        """Tests reading labeled dataset records."""
        test_records_amount = 5
        test_data = [
            [f"path_{i}", f"content_{i}", f"annotation_{i}"]
            for i in range(0, test_records_amount)
        ]
        ldr = LabeledDatasetReader(reader=MagicMock())
        ldr._reader = MagicMock()
        ldr._reader.read = MagicMock(return_value=test_data)

        # Test with iterator flag.
        ldr.read(iterator=True)
        ldr._reader.read.assert_called()

        # Test with Pandas DataFrame.
        result_df = ldr.read()
        ldr._reader.read.assert_called()
        expected_df = pd.DataFrame(
            test_data,
            columns=_LABELED_DF_COLUMNS,
        )
        assert pd.DataFrame.equals(result_df, expected_df)

        # Test with chunksize
        result_df = ldr.read(chunksize=1)
        mock__bulk_read.assert_called_with(iterator=False, format=None, chunksize=1)
        result_df = ldr.read(chunksize=1, iterator=True)
        mock__bulk_read.assert_called_with(iterator=True, format=None, chunksize=1)
        result_df = ldr.read(chunksize=1, iterator=True, format="spacy")
        mock__bulk_read.assert_called_with(iterator=True, format="spacy", chunksize=1)

    @pytest.mark.parametrize("test_chunksize", [None, {}, 0, -1, "1"])
    def test__bulk_read_fail(self, test_chunksize):
        """Ensures _bulk_read fails in case of incorrect chunksize."""
        ldr = LabeledDatasetReader(reader=MagicMock())
        generator = ldr._bulk_read(chunksize=test_chunksize)
        with pytest.raises(ValueError, match="`chunksize` must be a positive integer."):
            next(generator)

    @pytest.mark.parametrize(
        "test_chunksize, test_iterator, expected_result_length, expected_iterations_count",
        [
            (1, True, 1, 6),
            (1, False, 1, 6),
            (2, True, 2, 3),
            (2, False, 2, 3),
            (6, True, 6, 1),
            (6, False, 6, 1),
            (10, True, 6, 1),
            (10, False, 6, 1),
        ],
    )
    @patch("ads.common.auth.default_signer")
    def test__bulk_read(
        self,
        mock_signer,
        test_chunksize,
        test_iterator,
        expected_result_length,
        expected_iterations_count,
    ):
        test_records_amount = 6
        test_data = [
            [f"path_{i}", f"content_{i}", f"annotation_{i}"]
            for i in range(0, test_records_amount)
        ]
        ldr = LabeledDatasetReader.from_export(path="path")
        ldr._reader = MagicMock()
        ldr._reader.read = MagicMock(return_value=test_data)

        i = 0
        generator = ldr._bulk_read(chunksize=test_chunksize, iterator=test_iterator)
        for records in generator:
            assert len(records) == expected_result_length
            expected_records = test_data[
                i * test_chunksize : i * test_chunksize + test_chunksize
            ]
            if test_iterator:
                assert isinstance(records, List) == True
                assert records == expected_records
            else:
                assert isinstance(records, pd.DataFrame) == True
                assert pd.DataFrame.equals(
                    records, pd.DataFrame(expected_records, columns=_LABELED_DF_COLUMNS)
                )
            i += 1
        assert i == expected_iterations_count


class TestDLSDatasetReader:
    """Unittests for DLSDatasetReader class."""

    def test_init_success(self):
        """Ensures initializing dls dataset reader pass when valid input parameters provided."""
        # test with default parameters
        dls_dataset_reader = DLSDatasetReader(
            dataset_id="oci.xxxx.xxxx",
            compartment_id="oci.xxxx.xxxx",
            auth={"config": {}},
        )

        assert dls_dataset_reader.dataset_id == "oci.xxxx.xxxx"
        assert dls_dataset_reader.compartment_id == "oci.xxxx.xxxx"
        assert dls_dataset_reader.auth == {"config": {}}
        assert dls_dataset_reader.encoding == "utf-8"
        assert dls_dataset_reader.materialize == False

        # test with specific params
        dls_dataset_reader = DLSDatasetReader(
            dataset_id="oci.1234",
            compartment_id="oci.5678",
            auth={"options": "options"},
            encoding="encoding",
            materialize=True,
        )
        assert dls_dataset_reader.dataset_id == "oci.1234"
        assert dls_dataset_reader.compartment_id == "oci.5678"
        assert dls_dataset_reader.auth == {"options": "options"}
        assert dls_dataset_reader.encoding == "encoding"
        assert dls_dataset_reader.materialize == True

    def test_init_fail(self):
        """Ensures initializing dls dataset reader fails in case wrong input parameters provided."""
        with pytest.raises(ValueError, match="The dataset OCID must be specified."):
            DLSDatasetReader(dataset_id="", compartment_id="123", auth={})

        with pytest.raises(TypeError, match="The dataset_id must be a string."):
            DLSDatasetReader(dataset_id=MagicMock(), compartment_id="123", auth={})

    @property
    def generate_get_dataset_response_data(self):
        entity_item = {
            "annotation_format": "SINGLE_LABEL",
            "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
            "dataset_format_details": TextDatasetFormatDetails(format_type="TEXT"),
            "dataset_source_details": ObjectStorageDatasetSourceDetails(
                bucket="ads-dls-examples",
                namespace="ociodscdev",
                prefix="text/src/single-label/",
                source_type="OBJECT_STORAGE",
            ),
            "defined_tags": {},
            "description": None,
            "display_name": "unit_test_text_single_label",
            "freeform_tags": {},
            "id": "ocid1.datalabelingdataset.oc1.iad.<unique_ocid>",
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

    @patch.object(DLSDatasetReader, "read")
    @patch.object(MetadataReader, "from_DLS")
    @patch("ads.common.auth.default_signer")
    def test_info(self, mock_signer, mock_from_dls_file, mock_metadata_reader_read):
        """Tests getting the labeled dataset metadata from dls."""
        test_metadata = Metadata(
            source_path="source_path",
            compartment_id="compartment_id",
        )

        mock_metadata_reader = MetadataReader(
            reader=DLSDatasetReader(dataset_id="1234", compartment_id="5678", auth={})
        )

        mock_metadata_reader_read.return_value = test_metadata
        mock_from_dls_file.return_value = mock_metadata_reader

        dls_reader = DLSDatasetReader(dataset_id="1234", compartment_id="5678", auth={})
        dls_reader.info()

        result_metadata = mock_metadata_reader.read()
        assert result_metadata == test_metadata
        mock_metadata_reader_read.assert_called()

    @patch.object(RecordReader, "from_DLS")
    @patch.object(DLSDatasetReader, "info")
    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_read(self, mock_client, mock_signer, mock_info, mock_from_dls_file):
        """Tests reading the labeled dataset records from dls."""
        test_metadata = Metadata(
            dataset_type="dataset_type",
            annotation_type="annotation_type",
            dataset_id="1234",
            compartment_id="5678",
        )

        mock_info.return_value = test_metadata

        mock_reader = MagicMock()
        mock_from_dls_file.return_value = mock_reader

        record_reader = DLSMetadataReader(
            compartment_id="ocid.compartment.oc1.<unique_ocid>",
            dataset_id="ocid.dataset.oc1.<unique_ocid>",
            auth={},
        )
        record_reader.dls_dp_client = MagicMock()
        record_reader.dls_dp_client.get_dataset = MagicMock(
            return_value=Response(
                data=self.generate_get_dataset_response_data,
                status=None,
                headers=None,
                request=None,
            )
        )

        dls_reader = DLSDatasetReader(dataset_id="1234", compartment_id="5678", auth={})
        dls_reader.read()

        mock_info.assert_called()

        mock_from_dls_file.assert_called_with(
            dataset_type=test_metadata.dataset_type,
            annotation_type=test_metadata.annotation_type,
            dataset_source_path=test_metadata.source_path,
            compartment_id=test_metadata.compartment_id,
            dataset_id=test_metadata.dataset_id,
            auth=dls_reader.auth,
            materialize=dls_reader.materialize,
            include_unlabeled=False,
            encoding="utf-8",
            format=None,
            categories=[],
        )

        mock_reader.read.assert_called_once()


class TestExportReader:
    """Unittests for ExportReader class."""

    def test_init_success(self):
        """Ensures initializing export reader pass when valid input parameters provided."""
        # test with default parameters
        export_reader = ExportReader(path="path", auth={"config": {}})

        assert export_reader.path == "path"
        assert export_reader.auth == {"config": {}}
        assert export_reader.encoding == "utf-8"
        assert export_reader.materialize == False

        # test with specific params
        export_reader = ExportReader(
            path="path",
            auth={"options": "options"},
            encoding="encoding",
            materialize=True,
        )
        assert export_reader.path == "path"
        assert export_reader.auth == {"options": "options"}
        assert export_reader.encoding == "encoding"
        assert export_reader.materialize == True

    def test_init_fail(self):
        """Ensures initializing export reader fails in case wrong input parameters provided."""
        with pytest.raises(ValueError, match="The parameter `path` is required."):
            ExportReader(path="")

        with pytest.raises(TypeError, match="The parameter `path` must be a string."):
            ExportReader(path=MagicMock())

    @patch.object(ExportMetadataReader, "read")
    @patch.object(MetadataReader, "from_export_file")
    @patch("ads.common.auth.default_signer")
    def test_info(
        self, mock_signer, mock_from_export_file, mock_jsonl_metadata_reader_read
    ):
        """Tests getting the labeled dataset metadata."""
        test_metadata = Metadata(
            source_path="source_path",
            records_path="records_path",
        )

        mock_metadata_reader = MetadataReader(reader=ExportMetadataReader(path="path"))

        mock_jsonl_metadata_reader_read.return_value = test_metadata
        mock_from_export_file.return_value = mock_metadata_reader

        export_reader = ExportReader(path="path")
        export_reader.info()

        mock_from_export_file.assert_called_with(
            path=export_reader.path,
            auth=export_reader.auth,
        )

        result_metadata = mock_metadata_reader.read()
        assert result_metadata == test_metadata
        mock_jsonl_metadata_reader_read.assert_called()

    @pytest.mark.parametrize("mock_records_path", ["", None, "record_path"])
    @patch.object(RecordReader, "from_export_file")
    @patch.object(ExportReader, "info")
    @patch("ads.common.auth.default_signer")
    def test_read(
        self, mock_signer, mock_info, mock_from_export_file, mock_records_path
    ):
        """Tests reading the labeled dataset records."""
        test_metadata = Metadata(
            source_path="source_path",
            records_path=mock_records_path,
            dataset_type="dataset_type",
            annotation_type="annotation_type",
        )

        mock_info.return_value = test_metadata

        mock_reader = MagicMock()
        mock_from_export_file.return_value = mock_reader

        export_reader = ExportReader(path="source_path")
        export_reader.read()

        mock_info.assert_called()

        mock_from_export_file.assert_called_with(
            path=test_metadata.records_path or "source_path",
            dataset_type=test_metadata.dataset_type,
            annotation_type=test_metadata.annotation_type,
            dataset_source_path=test_metadata.source_path,
            auth=export_reader.auth,
            materialize=export_reader.materialize,
            include_unlabeled=False,
            encoding="utf-8",
            format=None,
            categories=[],
            includes_metadata=not mock_records_path,
        )

        mock_reader.read.assert_called_once()
