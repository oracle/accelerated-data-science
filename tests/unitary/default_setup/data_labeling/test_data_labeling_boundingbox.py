#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Unit tests for boundingbox module. Includes tests for:
 - BoundingBoxItem
 - BoundingBoxItems
"""


import pytest
from ads.data_labeling.boundingbox import BoundingBoxItem, BoundingBoxItems
from pytest import approx


class TestBoundingBoxItem:
    """Unittests for the BoundingBoxItem class."""

    test_bbox = BoundingBoxItem(
        top_left=(0.30246404, 0.092855014),
        bottom_left=(0.30246404, 0.42459726),
        bottom_right=(0.8620821, 0.42459726),
        top_right=(0.8620821, 0.092855014),
        labels=["dolphin", "fish"],
    )

    def test__validate_success(self):
        """Ensures validation passes in case of valid input parameters."""
        test_bbox = BoundingBoxItem(
            bottom_left=(0.2, 0.4),
            top_left=(0.2, 0.2),
            top_right=(0.8, 0.2),
            bottom_right=(0.8, 0.4),
            labels=["dolphin"],
        )
        assert test_bbox.top_left == (0.2, 0.2)
        assert test_bbox.bottom_left == (0.2, 0.4)
        assert test_bbox.top_right == (0.8, 0.2)
        assert test_bbox.bottom_right == (0.8, 0.4)
        assert test_bbox.labels == ["dolphin"]

    @pytest.mark.parametrize(
        "test_labels",
        [
            "label",
            [],
            {"labels": "value"},
        ],
    )
    def test__validate_fail_with_wrong_labels(self, test_labels):
        """Ensures validation fails in case of invalid input parameters."""
        with pytest.raises(
            ValueError, match=r".*must contain a non-empty list of string labels*"
        ):
            BoundingBoxItem(
                bottom_left=(0.2, 0.4),
                top_left=(0.2, 0.2),
                top_right=(0.8, 0.2),
                bottom_right=(0.8, 0.4),
                labels=test_labels,
            )

    @pytest.mark.parametrize(
        "test_box",
        [
            {
                "bottom_left": "",
                "top_left": (0.2, 0.2),
                "top_right": (0.8, 0.2),
                "bottom_right": (0.8, 0.4),
            },
            {
                "bottom_left": (0.2, 0.4),
                "top_left": (0.2,),
                "top_right": (0.8, 0.2),
                "bottom_right": (0.8, 0.4),
            },
            {
                "bottom_left": (0.2, 0.4),
                "top_left": (0.2, 0.2),
                "top_right": ("digit", 0.2),
                "bottom_right": (0.8, 0.4),
            },
            {
                "bottom_left": (0.2, 0.4),
                "top_left": (0.2, 0, 2),
                "top_right": (0.8, 0.2),
                "bottom_right": (3, 0.4),
            },
            {
                "bottom_left": (0.2, 0.4),
                "top_left": (0.2, 0, 2),
                "top_right": (0.8, 0.2),
                "bottom_right": (0.8, 3),
            },
        ],
    )
    def test__validate_fail_with_wrong_bounding_box(self, test_box):
        """Ensures validation fails in case of wrong bounding box format."""
        with pytest.raises(ValueError):
            BoundingBoxItem(
                bottom_left=test_box["bottom_left"],
                top_left=test_box["top_left"],
                top_right=test_box["top_right"],
                bottom_right=test_box["bottom_right"],
                labels=["fish"],
            )

    def test_to_yolo_success(self):
        """Ensures YOLO converting passes in case of valid input parameters."""
        test_categories = ["dolphin", "fish", "star"]
        expected_result = [
            (0, 0.58227307, 0.25872613699999997, 0.55961806, 0.33174224599999996),
            (1, 0.58227307, 0.25872613699999997, 0.55961806, 0.33174224599999996),
        ]
        assert self.test_bbox.to_yolo(test_categories) == expected_result

    def test_to_yolo_fail(self):
        """Ensures YOLO converter fails in case of invalid input parameters."""
        with pytest.raises(
            ValueError, match=r"The parameter `categories` is required. *"
        ):
            self.test_bbox.to_yolo(categories=[])

        with pytest.raises(
            TypeError, match=r"The parameter `categories` is invalid. *"
        ):
            self.test_bbox.to_yolo(categories={"category": "value"})

        with pytest.raises(
            ValueError, match=r"The parameter `categories` is invalid.*"
        ):
            self.test_bbox.to_yolo(categories=["one", "two", "three"])


class TestBoundingBoxItems:
    """Unittests for the BoundingBoxItems class."""

    test_data = [
        BoundingBoxItem(
            top_left=(0.30246404, 0.092855014),
            bottom_left=(0.30246404, 0.42459726),
            bottom_right=(0.8620821, 0.42459726),
            top_right=(0.8620821, 0.092855014),
            labels=["dolphin"],
        ),
        BoundingBoxItem(
            top_left=(0.30246404, 0.092855014),
            bottom_left=(0.30246404, 0.42459726),
            bottom_right=(0.8620821, 0.42459726),
            top_right=(0.8620821, 0.092855014),
            labels=["fish"],
        ),
    ]
    test_bounding_box_items = BoundingBoxItems(items=test_data)
    test_categories = ["dolphin", "fish", "star"]

    expected_result = [
        (0, 0.58227307, 0.25872613699999997, 0.55961806, 0.33174224599999996),
        (1, 0.58227307, 0.25872613699999997, 0.55961806, 0.33174224599999996),
    ]
    assert test_bounding_box_items.to_yolo(test_categories) == expected_result

    def test_to_yolo_fail(self):
        """Ensures YOLO converter fails in case of invalid input parameters."""
        test_bbox_items = BoundingBoxItems()

        with pytest.raises(ValueError):
            test_bbox_items.to_yolo(categories=[])

        with pytest.raises(TypeError):
            test_bbox_items.to_yolo(categories={"category": "value"})

    def test_from_yolo(self):
        expected_item = BoundingBoxItem(
            top_left=(0.3, 0.4),
            bottom_left=(0.3, 0.09),
            bottom_right=(0.86, 0.09),
            top_right=(0.86, 0.4),
            labels=["dolphin", "fish"],
        )
        yolo_item = expected_item.to_yolo(categories=["dolphin", "fish"])
        item = BoundingBoxItem.from_yolo(yolo_item, categories=["dolphin", "fish"])

        for att in item.__dict__.keys():
            if isinstance(getattr(item, att), tuple) and isinstance(
                getattr(item, att)[0], float
            ):
                assert getattr(item, att)[0] == approx(getattr(expected_item, att)[0])
                assert getattr(item, att)[1] == approx(getattr(expected_item, att)[1])
            else:
                assert getattr(item, att) == getattr(expected_item, att)
