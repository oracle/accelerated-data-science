#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import patch

import pytest
from ads.data_labeling.reader.export_record_reader import ExportRecordReader
from ads.data_labeling.reader.jsonl_reader import JsonlReader


class TestExportRecordReader:
    def test_init(self):
        """Ensures the TestExportRecordReader instance can be initialized."""
        reader = ExportRecordReader(
            path="test_path",
            auth={"test_auth"},
            encoding="test_encoding",
            includes_metadata=True,
        )
        assert reader.path == "test_path"
        assert reader.auth == {"test_auth"}
        assert reader.encoding == "test_encoding"
        assert reader._includes_metadata == True

    @pytest.mark.parametrize("mock_includes_metadata", [True, False, None])
    def test_read(self, mock_includes_metadata):
        """Tests reading labeled dataset records."""
        reader = ExportRecordReader(
            path="test_path",
            auth={"test_auth"},
            encoding="test_encoding",
            includes_metadata=mock_includes_metadata,
        )
        with patch.object(JsonlReader, "read", return_value=None) as mock_jsonl_read:
            reader.read()
            test_skip = 1 if mock_includes_metadata else None
            mock_jsonl_read.assert_called_with(skip=test_skip)
