#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from unittest.mock import patch

import pytest
from ads.data_labeling.loader.file_loader import (
    FileLoader,
    ImageFileLoader,
    TextFileLoader,
)


class TestFileLoader:

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./data_label_test_files/text/1_jd.txt",
    )
    path1 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./data_label_test_files/text/2_jd.txt",
    )

    @patch("ads.common.auth.default_signer")
    def test_load(self, mock_signer):
        file_content = FileLoader().load(self.path)
        assert file_content == b"test\n"

    @patch("ads.common.auth.default_signer")
    def test_load_file_not_exist(self, mock_signer):
        with pytest.raises(FileNotFoundError):
            FileLoader().load("invalid_path")

    @pytest.mark.parametrize("test_value", [None, {}, "", []])
    @patch("ads.common.auth.default_signer")
    def test_bulk_load_returns_empty_result(self, mock_signer, test_value):
        """Ensures bulk load retuens empty dictionary when incorrect input paramters passed in."""
        assert FileLoader().bulk_load(test_value) == {}

    @patch("ads.common.auth.default_signer")
    def test_bulk_load_success(self, mock_signer):
        """Ensures the files can be loaded in parallel threads."""
        test_files = [self.path, self.path1]
        test_result = FileLoader().bulk_load(
            [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
                for path in test_files
            ]
        )
        expected_result = {
            self.path: b"test\n",
            self.path1: b"test\n",
        }
        assert expected_result == test_result

    @patch("ads.common.auth.default_signer")
    def test_bulk_load_fail(self, mock_signer):
        """Ensures files bulk loader fails in case of problems with input files."""
        with pytest.raises(FileNotFoundError):
            FileLoader().bulk_load(["invalid_path"])


class TestTextFileLoader:
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./data_label_test_files/test.txt"
    )

    @patch("ads.common.auth.default_signer")
    def test_load(self, mock_signer):
        file_content = TextFileLoader().load(self.path)
        assert file_content == "test\n"

    @patch("ads.common.auth.default_signer")
    def test_load_file_not_exist(self, mock_signer):
        with pytest.raises(FileNotFoundError):
            FileLoader().load("invalid_path")


class TestImageFileLoader:
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./data_label_test_files/fish.jpg"
    )

    @patch("ads.common.auth.default_signer")
    def test_load(self, mock_signer):
        file_content = ImageFileLoader().load(self.path)
        assert file_content.size == (768, 1024)

    @patch("ads.common.auth.default_signer")
    def test_load_file_not_exist(self, mock_signer):
        with pytest.raises(FileNotFoundError):
            FileLoader().load("invalid_path")
