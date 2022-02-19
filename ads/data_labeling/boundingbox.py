#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BoundingBoxItem:
    """BoundingBoxItem class representing bounding box label.

    Attributes
    ----------
    labels: List[str]
        List of labels for this bounding box.
    top_left: Tuple[float, float]
        Top left corner of this bounding box.
    bottom_left: Tuple[float, float]
        Bottom left corner of this bounding box.
    bottom_right: Tuple[float, float]
        Bottom right corner of this bounding box.
    top_right: Tuple[float, float]
        Top right corner of this bounding box.

    Examples
    --------
    >>> item = BoundingBoxItem(
    ...     labels = ['cat','dog']
    ...     bottom_left=(0.2, 0.4),
    ...     top_left=(0.2, 0.2),
    ...     top_right=(0.8, 0.2),
    ...     bottom_right=(0.8, 0.4))
    >>> item.to_yolo(categories = ['cat','dog', 'horse'])
    """

    top_left: Tuple[float, float]
    bottom_left: Tuple[float, float]
    bottom_right: Tuple[float, float]
    top_right: Tuple[float, float]
    labels: List[str] = field(default_factory=list)

    def _validate(self):
        """Validates the instance.

        Raises
        ------
        ValueError
            If the bounding box coordinate is not between [0.0, 1.0].
        """
        if (
            not self.labels
            or not isinstance(self.labels, list)
            or len(self.labels) == 0
        ):
            raise ValueError(
                "The parameter `df` is invalid. The BoundingBoxItem objects in the column "
                f"{self.labels}  must contain a non-empty list of string labels."
            )

        if any(
            (
                not isinstance(entity, Tuple)
                or len(entity) != 2
                or not isinstance(entity[0], float)
                or not isinstance(entity[1], float)
                or not 0.0 <= entity[0] <= 1.0
                or not 0.0 <= entity[1] <= 1.0
            )
            for entity in [
                self.bottom_left,
                self.top_left,
                self.top_right,
                self.bottom_right,
            ]
        ):
            raise ValueError(
                "The parameter `df` is invalid. The BoundingBoxItem objects in the column "
                f"{[self.bottom_left, self.top_left, self.top_right, self.bottom_right,]} must "
                "contain a tuple of two real numbers in the range of [0, 1]. "
                "One BoundingBoxItem contains the values (invalid_tuple). Use the `DataLabeling.export()` "
                "method to create a new dataset record file."
            )

    def __post_init__(self):
        self._validate()

    def to_yolo(
        self, categories: List[str]
    ) -> List[Tuple[int, float, float, float, float]]:
        """Converts BoundingBoxItem to the YOLO format.

        Parameters
        ----------
        categories: List[str]
            The list of object categories in proper order for model training.
            Example: ['cat','dog','horse']

        Returns
        -------
        List[Tuple[int, float, float, float, float]]
            The list of YOLO formatted bounding boxes.

        Raises
        ------
        ValueError
            When categories list not provided.
            When categories list not matched with the labels.
        TypeError
            When categories list has a wrong format.
        """
        if not categories:
            raise ValueError(
                "The parameter `categories` is required. Use the `.info()` method to obtain a list of categories for this dataset."
            )
        if not isinstance(categories, list):
            raise TypeError(
                "The parameter `categories` is invalid. It must be an object of type `List[str]`."
            )
        if not set(self.labels).issubset(categories):
            raise ValueError(
                "The parameter `categories` is invalid. It must be a list of all the unique labels in the dataset."
            )

        category_map = {k: v for v, k in enumerate(categories)}
        coords = (
            (self.top_left[0] + self.top_right[0]) / 2,
            (self.top_left[1] + self.bottom_left[1]) / 2,
            self.top_right[0] - self.top_left[0],
            self.bottom_left[1] - self.top_left[1],
        )

        return [(category_map[label],) + coords for label in self.labels]

    @classmethod
    def from_yolo(
        cls, bbox: List[Tuple], categories: List[str] = None
    ) -> "BoundingBoxItem":
        """Converts the YOLO formated annotations to BoundingBoxItem.

        Parameters
        ----------
        bboxes: List[Tuple]
            The list of bounding box annotations in YOLO format.
            Example: [(0, 0.511560675, 0.50234826, 0.47013485, 0.57803468)]
        categories: List[str]
            The list of object categories in proper order for model training.
            Example: ['cat','dog','horse']

        Returns
        -------
        BoundingBoxItem
            The BoundingBoxItem.

        Raises
        ------
        TypeError
            When categories list has a wrong format.
        """

        if bbox:
            _, x1, y1, w_size, h_size = bbox[0]
            labels = [label[0] for label in bbox]
            if categories:
                if not isinstance(categories, list):
                    raise TypeError("The categories must be a List[str].")
                if max(labels) + 1 > len(categories):
                    raise ValueError(
                        "Index out of the range of the categories list. "
                        "The categories must contain all the labels "
                        "in the order that the integer label corresponds to."
                    )

                labels = [categories[label[0]] for label in bbox]

            top_left = (x1 - (w_size / 2), y1 - (h_size / 2))
            top_right = (x1 + (w_size / 2), y1 - (h_size / 2))
            bottom_right = (x1 + (w_size / 2), y1 + (h_size / 2))
            bottom_left = (x1 - (w_size / 2), y1 + (h_size / 2))
            bbox_item = BoundingBoxItem(
                top_left, bottom_left, bottom_right, top_right, labels
            )
            bbox_item._validate()
            return bbox_item
        else:
            return None


@dataclass
class BoundingBoxItems:
    """BoundingBoxItems class which consists of a list of BoundingBoxItem.

    Attributes
    ----------
    items: List[BoundingBoxItem]
        List of BoundingBoxItem.

    Examples
    --------
    >>> item = BoundingBoxItem(
    ...     labels = ['cat','dog']
    ...     bottom_left=(0.2, 0.4),
    ...     top_left=(0.2, 0.2),
    ...     top_right=(0.8, 0.2),
    ...     bottom_right=(0.8, 0.4))
    >>> items = BoundingBoxItems(items = [item])
    >>> items.to_yolo(categories = ['cat','dog', 'horse'])
    """

    items: List[BoundingBoxItem] = field(default_factory=list)

    def __getitem__(self, index: int) -> BoundingBoxItem:
        return self.items[index]

    def to_yolo(
        self, categories: List[str]
    ) -> List[Tuple[int, float, float, float, float]]:
        """Converts BoundingBoxItems to the YOLO format.

        Parameters
        ----------
        categories: List[str]
            The list of object categories in proper order for model training.
            Example: ['cat','dog','horse']

        Returns
        -------
        List[Tuple[int, float, float, float, float]]
            The list of YOLO formatted bounding boxes.

        Raises
        ------
        ValueError
            When categories list not provided.
            When categories list not matched with the labels.
        TypeError
            When categories list has a wrong format.
        """
        if not categories:
            raise ValueError(
                "The parameter `categories` is required. Use the `.info()` method to obtain a list of categories for this dataset."
            )
        if not isinstance(categories, list):
            raise TypeError(
                "The parameter `categories` is invalid. It must be an object of type `List[str]`."
            )

        result = []
        for item in self.items:
            result.extend(item.to_yolo(categories))
        return result
