#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
from io import BytesIO
from unittest.mock import mock_open, patch

import pytest
from ads.feature_engineering.adsimage.image import ADSImage
from PIL import Image


class TestADSImage:
    @classmethod
    def setup_class(cls):
        cls.curdir = os.path.dirname(os.path.abspath(__file__))

    def test_init_sucess(self):
        """Ensures the ADSImage object can be successfully instantiated."""
        pil_img = Image.Image()
        file_name = "test_file_name.jpg"
        ads_img = ADSImage(pil_img, file_name)

        assert isinstance(ads_img, ADSImage) == True
        assert isinstance(ads_img.img, Image.Image)
        assert ads_img.filename == file_name

    @pytest.mark.parametrize(
        "img, err_type, err_msg",
        [
            (None, ValueError, "The parameter `img` is required."),
            (
                "not an image",
                TypeError,
                "The `img` parameter must be a `PIL.Image.Image` object.",
            ),
        ],
    )
    def test_init_fail(self, img, err_type, err_msg):
        """Ensures that initialization of ADSImage fails in case of wrong input parameters."""
        with pytest.raises(err_type, match=err_msg):
            ADSImage(img, "")

    @patch("ads.common.auth.default_signer")
    def test_open_success(self, mock_default_signer):
        """Ensures that image can be sucessfully opened."""
        mock_default_signer.return_value = {"config": "value"}
        test_img_file = os.path.join(self.curdir, "test_files/001.jpg")
        pil_img = Image.open(test_img_file)
        ads_img = ADSImage.open(test_img_file)
        mock_default_signer.assert_called()
        assert isinstance(ads_img, ADSImage)
        assert ads_img.filename == "001.jpg"
        assert isinstance(ads_img.img, Image.Image)
        assert list(ads_img.img.getdata())[:10] == list(pil_img.getdata())[:10]

    @patch("ads.common.auth.default_signer")
    def test_open_fail(self, mock_default_signer):
        """Ensures openning image fails in case of wrong imge path."""
        with pytest.raises(FileNotFoundError):
            ADSImage.open("not_existing_file.jpg")

    @patch("ads.common.auth.default_signer")
    def test_save(self, mock_default_signer):
        """Ensures that image can be sucessfully saved under the given filename."""
        ads_img = ADSImage.open(os.path.join(self.curdir, "test_files/001.jpg"))

        with tempfile.TemporaryDirectory() as temp_dir:
            to_path = os.path.join(temp_dir, "dst_img.jpg")
            ads_img.save(to_path)
            new_ads_img = ADSImage.open(to_path)
            assert (
                list(ads_img.img.getdata())[:10] == list(new_ads_img.img.getdata())[:10]
            )

    @patch("ads.common.auth.default_signer")
    def test_save_mock(self, mock_default_signer):
        """Validates image saving steps."""
        mock_default_signer.return_value = {"config": {"test": "value"}}
        mock_file_path = "oci://bucket-name@namespace/test_image.jpg"

        ads_img = ADSImage.open(os.path.join(self.curdir, "test_files/001.jpg"))
        imgByteArr = BytesIO()
        ads_img.img.save(imgByteArr, ads_img.img.format)

        open_mock = mock_open()
        with patch("fsspec.open", open_mock, create=True):
            ads_img.save(mock_file_path)
            mock_default_signer.asseret_called()
            open_mock.assert_called_with(
                mock_file_path, mode="wb", **{"config": {"test": "value"}}
            )
            open_mock.return_value.write.assert_called()
