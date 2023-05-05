#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from unittest.mock import MagicMock, mock_open, patch

import pytest
from ads.feature_engineering.adsimage.image_reader import (
    ADSImageReader,
    ImageFileReader,
)


class TestADSImageReader:
    """Tests the ADSImageReader helper class."""

    def test_init(self):
        """Ensures the ADSImageReader object can be successfully instantiated."""
        mock_reader = MagicMock()
        image_reader = ADSImageReader(mock_reader)
        assert image_reader._reader == mock_reader

    @patch("ads.common.auth.default_signer")
    @patch.object(ImageFileReader, "__init__")
    def test_from_uri(self, mock_imagefilereader_init, mock_default_signer):
        "Tests constructing ADSImageReader object."
        mock_imagefilereader_init.return_value = None
        default_signer_value = {"config": {"test": "value"}}
        mock_default_signer.return_value = default_signer_value
        mock_uri_path = "oci://bucket-name@namespace/test_image.jpg"

        img_reader = ADSImageReader.from_uri(mock_uri_path)
        mock_default_signer.assert_called()
        mock_imagefilereader_init.assert_called_with(
            path=mock_uri_path, auth=default_signer_value
        )
        assert isinstance(img_reader, ADSImageReader)

    @patch("ads.common.auth.default_signer")
    def test_read(self, mock_default_signer):
        """Tests reading images."""
        default_signer_value = {"config": {"test": "value"}}
        mock_default_signer.return_value = default_signer_value

        mock_reader = MagicMock()
        mock_read = MagicMock()
        mock_read.return_value = "test_value"
        mock_reader.read = mock_read
        image_reader = ADSImageReader(mock_reader)
        test_result = image_reader.read()
        mock_read.assert_called_once()
        assert test_result == "test_value"


class TestImageFileReader:
    """Tests the ImageFileReader helper class."""

    @classmethod
    def setup_class(cls):
        cls.curdir = os.path.dirname(os.path.abspath(__file__))

    def test_init_success(self):
        """Ensures that ImageFileReader object can be successfully instantiated."""

        test_path = "test_path"
        img_file_reader = ImageFileReader(test_path, {"config": ""})

        assert isinstance(img_file_reader, ImageFileReader) == True
        assert img_file_reader.path == test_path
        assert img_file_reader.auth == {"config": ""}

    @pytest.mark.parametrize(
        "path, err_type, err_msg",
        [
            (None, ValueError, "The parameter `path` is required."),
            (
                {"not_valid_path"},
                TypeError,
                "The `path` parameter must be a string or list of strings.",
            ),
        ],
    )
    def test_init_fail(self, path, err_type, err_msg):
        """Ensures that initialization of ADSImage fails in case of wrong input parameters."""
        with pytest.raises(err_type, match=err_msg):
            ImageFileReader(path, {"config": ""})

    def test_read(self):
        """Tests reading images."""
        auth = {"config": ""}
        test_images_dir = os.path.join(self.curdir, "test_files")
        test_image_files = [f"{test_images_dir}/001.jpg", f"{test_images_dir}/002.jpg"]
        image_reader = ImageFileReader(test_image_files, auth=auth)
        index = 0
        for ads_image in image_reader.read():
            ads_image.filename == os.path.basename(test_image_files[index])
            assert len(list(ads_image.img.getdata())) > 0
            index += 1
        assert index == len(test_image_files)
