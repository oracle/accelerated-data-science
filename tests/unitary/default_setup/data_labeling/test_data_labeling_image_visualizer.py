#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


"""
Unit tests for image_visualizer module. Includes tests for:
 - LabeledImageItem
 - RenderOptions
 - ImageLabeledDataFormatter
 - render()
"""


import os
from unittest.mock import patch

import pandas as pd
import PIL
import pytest
from ads.data_labeling.boundingbox import BoundingBoxItem
from ads.data_labeling.visualizer.image_visualizer import (
    ImageLabeledDataFormatter,
    LabeledImageItem,
    RenderOptions,
    WrongEntityFormat,
    _df_to_bbox_items,
    render,
)
from pytest import approx


class TestLabeledImageItem:
    """Unittests for TestLabeledImageItem class."""

    img_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "image_files",
        "img1.jpeg",
    )
    MOCK_IMG = PIL.Image.open(img_path)

    MOCK_bbox = BoundingBoxItem(
        bottom_left=(0.3, 0.4),
        top_left=(0.3, 0.09),
        top_right=(0.86, 0.09),
        bottom_right=(0.86, 0.4),
        labels=["dolphin"],
    )

    def test__validate_success(self):
        """Ensures validation passes in case of valid input parameters."""
        test_item = LabeledImageItem(self.MOCK_IMG, [self.MOCK_bbox])
        assert isinstance(test_item.img, PIL.ImageFile.ImageFile) == True
        assert isinstance(test_item.boxes, list) == True
        assert isinstance(test_item.boxes[0], BoundingBoxItem) == True

    def test__validate_fail_with_empty_img(self):
        """Ensures validation fails in case of invalid input parameters."""
        with pytest.raises(ValueError):
            LabeledImageItem("", [])

    @pytest.mark.parametrize(
        "test_ents",
        [
            "wrong format",
            [("sports", "baseball")],
        ],
    )
    def test__validate_fail_with_wrong_ents(self, test_ents):
        """Ensures validation fails in case of wrong entities format."""
        with pytest.raises(WrongEntityFormat):
            LabeledImageItem(self.MOCK_IMG, test_ents)


class TestRenderOptions:
    """Unittests for TestRenderOptions class."""

    MOCK_VALID_OPTIONS = {
        "default_color": "blue",
        "colors": {"yes": "red", "no": "orange"},
    }

    def test__validate_success(self):
        """Ensures validation passes in case of valid input parameters."""
        RenderOptions._validate(self.MOCK_VALID_OPTIONS)

    @pytest.mark.parametrize(
        "test_options",
        [
            {"default_color": "invalid", "colors": {"baseball": "red"}},
            {"default_color": "#DDEECC", "colors": {"some_key": "invalid_value"}},
        ],
    )
    def test__validate_fail(self, test_options):
        """Ensures validation fails in case of invalid input parameters."""
        with pytest.raises(ValueError):
            RenderOptions._validate(test_options)

    def test_from_dict_success(self):
        """Ensures RenderOptions instance can be created from dict."""
        render_options = RenderOptions.from_dict(self.MOCK_VALID_OPTIONS)
        assert render_options.default_color == self.MOCK_VALID_OPTIONS["default_color"]
        assert render_options.colors == self.MOCK_VALID_OPTIONS["colors"]

    def test_from_dict_fail(self):
        """Ensures instantiating RwnderOptions from dict fails in case of invalid input parameters."""
        with pytest.raises(AttributeError):
            RenderOptions.from_dict({"colors": "invalid_val"})

    def test_to_dict(self):
        """Tests converting RenderOptions instance to dictionary format."""
        render_options = RenderOptions.from_dict(self.MOCK_VALID_OPTIONS)
        assert render_options.to_dict() == self.MOCK_VALID_OPTIONS


class TestImageLabeledDataFormatter:
    """Unittests for TestImageLabeledDataFormatter class."""

    img_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "image_files",
        "img1.jpeg",
    )
    MOCK_IMG = PIL.Image.open(img_path)

    MOCK_bbox = BoundingBoxItem(
        bottom_left=(0.3, 0.4),
        top_left=(0.3, 0.09),
        top_right=(0.86, 0.09),
        bottom_right=(0.86, 0.4),
        labels=["dolphin", "mammal"],
    )

    @pytest.mark.parametrize(
        "test_labels",
        [["dolphin"], ["dolphin", "mammal"]],
    )
    def test_render_image_item_labels(self, test_labels):
        MOCK_bbox = BoundingBoxItem(
            bottom_left=(0.3, 0.4),
            top_left=(0.3, 0.09),
            top_right=(0.86, 0.09),
            bottom_right=(0.86, 0.4),
            labels=test_labels,
        )
        test_record = LabeledImageItem(self.MOCK_IMG, [MOCK_bbox])
        img_output_path = "test.png"
        ImageLabeledDataFormatter.render_item(item=test_record, path=img_output_path)
        assert os.path.exists(img_output_path) == True
        os.remove(img_output_path)

    @pytest.mark.parametrize(
        "test_path",
        ["test.png", "test.test.jpeg"],
    )
    def test_render_image_item_paths(self, test_path):
        test_record = LabeledImageItem(self.MOCK_IMG, [self.MOCK_bbox])
        ImageLabeledDataFormatter.render_item(item=test_record, path=test_path)
        assert os.path.exists(test_path) == True
        os.remove(test_path)

    @pytest.mark.parametrize(
        "test_options",
        [
            {"default_color": "orange", "colors": {"dolphin": "red", "shark": "green"}},
            {
                "default_color": "orange",
                "colors": {"dolphin": "red", ("dolphin", "fish"): "blue"},
            },
            None,
        ],
    )
    def test_render_ner_item_with_customized_color(self, test_options):
        test_record = LabeledImageItem(self.MOCK_IMG, [self.MOCK_bbox])

        img_output_path = "test.png"
        ImageLabeledDataFormatter.render_item(
            item=test_record, options=test_options, path=img_output_path
        )
        assert os.path.exists(img_output_path) == True
        os.remove(img_output_path)

    @pytest.mark.parametrize(
        "test_item",
        [
            "image",
            None,
        ],
    )
    def test_render_image_fail_item(self, test_item):
        with pytest.raises((ValueError, TypeError)):
            ImageLabeledDataFormatter.render_item(item=test_item)

    @pytest.mark.parametrize(
        "test_path",
        [
            "/tmp/images",  # dir not exist
            "/tmp/images/1.tiff",
        ],
    )
    def test_render_image_fail_path(self, test_path):
        test_record = LabeledImageItem(self.MOCK_IMG, [self.MOCK_bbox])
        with pytest.raises((ValueError, TypeError)):
            ImageLabeledDataFormatter.render_item(item=test_record, path=test_path)

    def test_render_bounding_box_yolo(self):
        """Tests rendering bounding box dataset."""
        expected_item = BoundingBoxItem(
            top_left=(0.3, 0.4),
            bottom_left=(0.3, 0.09),
            bottom_right=(0.86, 0.09),
            top_right=(0.86, 0.4),
            labels=["dolphin", "mammal"],
        )
        expected_items = [expected_item] * 2
        test_record = [
            PIL.Image.open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "./data_label_test_files/fish.jpg",
                )
            ),
            [
                [
                    (0, 0.58, 0.245, 0.56, -0.31000000000000005),
                    (1, 0.58, 0.245, 0.56, -0.31000000000000005),
                ]
            ],
        ]

        test_df = pd.DataFrame([test_record] * 2, columns=["Content", "Annotations"])

        test_bbox_items = _df_to_bbox_items(
            df=test_df,
            content_column="Content",
            annotations_column="Annotations",
            categories=["dolphin", "mammal"],
        )
        for items in test_bbox_items:
            for item in items.boxes:
                for att in item.__dict__.keys():
                    print(att)

                    if isinstance(getattr(item, att), tuple) and isinstance(
                        getattr(item, att)[0], float
                    ):
                        assert getattr(item, att)[0] == approx(
                            getattr(expected_item, att)[0]
                        )
                        assert getattr(item, att)[1] == approx(
                            getattr(expected_item, att)[1]
                        )
                    else:
                        assert set(getattr(item, att)) == set(
                            getattr(expected_item, att)
                        )


class TestRender:
    """Unittests for TestRender class."""

    img_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "image_files",
        "img1.jpeg",
    )
    MOCK_IMG = PIL.Image.open(img_path)

    MOCK_bbox = BoundingBoxItem(
        bottom_left=(0.3, 0.4),
        top_left=(0.3, 0.09),
        top_right=(0.86, 0.09),
        bottom_right=(0.86, 0.4),
        labels=["dolphin"],
    )
    test_record1 = LabeledImageItem(MOCK_IMG, [MOCK_bbox])

    @patch.object(ImageLabeledDataFormatter, "render_item", return_value=None)
    def test_render(self, mock_render):
        test_options = {
            "default_color": "orange",
            "colors": {"dolphin": "red", "shark": "green"},
        }

        render([self.test_record1], options=test_options)

        mock_render.assert_called_with(self.test_record1, test_options)

    @patch.object(ImageLabeledDataFormatter, "render_item", return_value=None)
    def test_render_multiple(self, mock_render):
        img_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "image_files",
            "img2.jpeg",
        )
        MOCK_IMG2 = PIL.Image.open(img_path)

        bbox2 = BoundingBoxItem(
            bottom_left=(0.2, 0.4),
            top_left=(0.2, 0.2),
            top_right=(0.8, 0.2),
            bottom_right=(0.8, 0.4),
            labels=["dolphin"],
        )
        bbox3 = BoundingBoxItem(
            bottom_left=(0.5, 1.0),
            top_left=(0.5, 0.8),
            top_right=(0.8, 0.8),
            bottom_right=(0.8, 1.0),
            labels=["whale"],
        )
        test_record2 = LabeledImageItem(MOCK_IMG2, [bbox2, bbox3])

        test_options = {
            "default_color": "orange",
            "colors": {"dolphin": "red", "shark": "green"},
        }

        render([self.test_record1, test_record2], options=test_options)
        mock_render.call_count = 2
