#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from unittest.mock import patch

import pytest
from ads.common import auth
from ads.data_labeling.reader.jsonl_reader import JsonlReader

encoding = "utf-8"


def _mock_reader(src_file: str):
    with patch.object(auth, "default_signer"):
        return JsonlReader(
            path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data_label_test_files",
                src_file,
            ),
            encoding=encoding,
        )


class TestJsonlReader:
    def test_init(self):
        reader = _mock_reader(src_file="text_ner_records.jsonl")
        assert "text_ner_records.jsonl" in reader.path
        assert reader.encoding == encoding

    def test_read_local(self):
        reader = _mock_reader(src_file="document_document_1631762769846.jsonl")
        content = reader.read()
        for item in content:
            assert "id" in item
            assert "compartmentId" in item

    def test_read_empty(self):
        reader = _mock_reader(src_file="empty.jsonl")
        content = reader.read()
        for item in content:
            assert item is None

    @pytest.mark.parametrize("mock_skip", [{"vaue"}, -1, "1"])
    def test_read_fail_if_skip_wrong_format(self, mock_skip):
        """Ensures `read` fails in case of wrong `skip` parameter."""
        reader = _mock_reader(src_file="document_document_1631762769846.jsonl")
        generator = reader.read(skip=mock_skip)
        with pytest.raises(ValueError, match="must be a positive integer"):
            next(generator)

    def test_read_fail_if_file_not_found(self):
        """Ensures `read` fails in case of the source file not found."""
        reader = _mock_reader(src_file="not_exist.jsonl")
        generator = reader.read()
        with pytest.raises(FileNotFoundError, match="not found"):
            next(generator)

    @pytest.mark.parametrize(
        "mock_skip, expected_number_of_records", [(0, 3), (1, 2), (2, 1), (100, 0)]
    )
    def test_read_with_skip(self, mock_skip, expected_number_of_records):
        """Ensures the `read` function supports `skip` parameter."""
        reader = _mock_reader(src_file="text_ner_records.jsonl")
        generator = reader.read(skip=mock_skip)
        result = [item for item in generator]
        assert len(result) == expected_number_of_records
