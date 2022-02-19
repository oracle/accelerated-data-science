#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that helps to visualize Image Dataset.

Methods
-------
    render(items: List[LabeledImageItem], options: Dict = None)
        Renders Labeled Image dataset.

Examples
--------
>>> bbox1 = BoundingBoxItem(bottom_left=(0.3, 0.4),
>>>                        top_left=(0.3, 0.09),
>>>                        top_right=(0.86, 0.09),
>>>                        bottom_right=(0.86, 0.4),
>>>                        labels=['dolphin', 'fish'])

>>> record1 = LabeledImageItem(img_obj1, [bbox1])

>>> bbox2 = BoundingBoxItem(bottom_left=(0.2, 0.4),
>>>                        top_left=(0.2, 0.2),
>>>                        top_right=(0.8, 0.2),
>>>                        bottom_right=(0.8, 0.4),
>>>                        labels=['dolphin'])
>>> bbox3 = BoundingBoxItem(bottom_left=(0.5, 1.0),
>>>                        top_left=(0.5, 0.8),
>>>                        top_right=(0.8, 0.8),
>>>                        bottom_right=(0.8, 1.0),
>>>                        labels=['shark'])

>>> record2 = LabeledImageItem(img_obj2, [bbox2, bbox3])
>>> render(items = [record1, record2], options={"default_color":"blue", "colors": {"dolphin":"blue", "whale":"red"}})
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from matplotlib.colors import is_color_like
import matplotlib.pyplot as plt
from ads.common import logger
import os
import numpy as np
import pandas as pd
from ads.data_labeling.boundingbox import BoundingBoxItem
from PIL.ImageFile import ImageFile
from ads.data_labeling.constants import AnnotationType

DEFAULT_COLOR = "white"
IMG_FORMAT = [".jpg", ".jpeg", ".png"]


class WrongEntityFormat(ValueError):
    def __init__(self):
        super().__init__(
            "Invalid labels from the dataset, "
            f"cannot construct a valid BoundingBoxItem."
        )


@dataclass
class LabeledImageItem:
    """Data class representing Image Item.

    Attributes
    ----------
    img: ImageFile
       the labeled image object.
    boxes: List[BoundingBoxItem]
        a list of BoundingBoxItem
    """

    img: ImageFile
    boxes: List[BoundingBoxItem]

    def _validate(self):
        """Validates the instance.

        Raises
        ------
        ValueError
            If image object is empty.
        WrongEntityFormat
            If the list of entities has a wrong format.
        """
        if self.img is None:
            raise ValueError("The parameter `img` is required.")

        if not isinstance(self.img, ImageFile):
            raise ValueError(
                "The parameter `img` must be an object of type `PIL.ImageFile.ImageFile`."
            )

        if any(not isinstance(entity, BoundingBoxItem) for entity in self.boxes):
            raise WrongEntityFormat()

    def __post_init__(self):
        self._validate()


@dataclass
class RenderOptions:
    """Data class representing render options.

    Attributes
    ----------
    default_color: str
        The specified default color.
    colors: Optional[dict]
        The multiple specified colors.
    """

    default_color: str
    colors: Optional[dict]

    @staticmethod
    def _validate(options: dict) -> None:
        """Validate whether the options passed in fits the defined schema.

        Parameters
        ----------
        options: dict
            The multiple specified colors.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        ValueError
            If color provided is not valid.
        """
        if not options:
            return None

        colorvalues = set(options.get("colors", {}).values())
        if "default_color" in options:
            colorvalues.add(options["default_color"])

        for colorval in colorvalues:
            if not is_color_like(colorval):
                raise ValueError(
                    f"{colorval} is not supported. "
                    f"Use RGB format for colors. For instance: `'#EEEEEE'` or `'green'`."
                )

    @classmethod
    def from_dict(cls, options: dict) -> "RenderOptions":
        """Constructs an instance of RenderOptions from a dictionary.

        Parameters
        ----------
        options: dict
            Render options in dictionary format.

        Returns
        -------
        RenderOptions
            The instance of RenderOptions.
        """
        if not options:
            return cls(default_color=DEFAULT_COLOR, colors={})

        RenderOptions._validate(options)

        return cls(
            options.get("default_color", DEFAULT_COLOR), options.get("colors", {}) or {}
        )

    def to_dict(self):
        """Converts RenderOptions instance to dictionary format.

        Returns
        -------
        dict
            The render options in dictionary format.
        """
        return asdict(self)

    def __repr__(self) -> str:
        return repr(self.to_dict())


class ImageLabeledDataFormatter:
    """The ImageRender class to render Image items in a notebook session."""

    @staticmethod
    def render_item(
        item: LabeledImageItem, options: Dict = None, path: str = None
    ) -> None:
        """Renders image dataset.

        Parameters
        ----------
        item: LabeledImageItem
            Item to render.
        options: Optional[dict]
            Render options.
        path: str
            Path to save the image with annotations to local directory.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        ValueError
            If items not provided.
            If path is not valid.
        TypeError
            If items provided in a wrong format.
        """
        if not item:
            raise ValueError("The parameter `item` is required.")

        if not isinstance(item, LabeledImageItem):
            raise TypeError(
                "The parameter `item` must be an object of type `LabeledImageItem`."
            )

        render_options = RenderOptions.from_dict(options)

        if path:
            if os.path.isdir(path):
                path += "1.jpg"
            elif not path.lower().endswith(tuple(IMG_FORMAT)):
                img_format_str = (
                    IMG_FORMAT[0]
                    if len(IMG_FORMAT) == 1
                    else ", ".join(IMG_FORMAT[:-1] + ["and " + IMG_FORMAT[-1]])
                )
                raise ValueError(
                    f"Invalid {path}. It is not a directory or the image format "
                    f"in {path} is not supported. Currently the support types "
                    f"are `{img_format_str}`."
                )

        # drow the image with annotations
        ImageLabeledDataFormatter()._draw_labels(
            item=item, options=render_options, path=path
        )

    def _draw_labels(
        self,
        item: LabeledImageItem,
        options: Dict = None,
        path: str = None,
        figure_size: Tuple = (6, 8),
        fontsize: int = 14,
    ):
        """Draw image with annotations.

        Parameters
        ----------
        item: LabeledImageItem
            Item to render.
        options: Optional[dict]
            Render options.
        path: str
            Path to save the image with annotations to local directory.
        figure_size: Tuple
            Figure size of the rendered image.
        fontsize: int
            Font size of the annotations.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        TypeError
            If image type is not PIL.ImageFile.ImageFile.
        """
        img = item.img

        # using matplotlib to open image return numpy array
        if hasattr(img, "shape"):
            im_height, im_width, _ = img.shape
        elif hasattr(img, "size"):
            im_width, im_height = img.size
        else:
            raise TypeError("The input image type must be `PIL.ImageFile.ImageFile`.")

        fig, ax = plt.subplots(1, 1, figsize=figure_size)
        ax.imshow(img)

        for ent in item.boxes:
            category_name = ", ".join([str(label) for label in ent.labels])
            left, top, width, height = self._calculate_bbx(im_width, im_height, ent)

            if len(ent.labels) > 1:
                # sort the multiple labels and use the tuple as key to look up in colormap
                # note: tuple of labels provided as key in colormap must be sorted.
                color_key = tuple(sorted(ent.labels))
                color = options.colors.get(color_key, options.default_color)
            else:
                color = options.colors.get(category_name, options.default_color)

            rect = plt.Rectangle(
                (left, top),
                width,
                height,
                fill=False,
                linewidth=2,
                edgecolor=color,
            )
            ax.add_patch(rect)
            props = dict(boxstyle="round", facecolor=color, alpha=0.6)
            ax.text(
                left,
                top,
                category_name,
                fontsize=fontsize,
                color="black",
                verticalalignment="top",
                bbox=props,
            )

        ax.axis("off")
        if path:
            plt.savefig(path, bbox_inches="tight")
            logger.info(f"The annotated image file is saved in {path}.")

            # Add this line to not show image
            plt.close(fig)
            return

        plt.show()

    def _calculate_bbx(self, im_width, im_height, bbox):
        """calculate bounding box coordinates

        Parameters
        ----------
        im_width: float
            width of the image in pixels
        im_height: float
            height of the image in pixels
        bbox: BoundingBoxItem

        Returns
        -------
        Tuple
            left, top, width, height of the image
        """
        left = bbox.top_left[0] * im_width
        top = bbox.top_left[1] * im_height
        width = (bbox.top_right[0] - bbox.top_left[0]) * im_width
        height = (bbox.bottom_left[1] - bbox.top_left[1]) * im_height
        return left, top, width, height

    def _convert_pil_to_nparray(self, img):
        """convert pil image object to numpy array

        Parameters
        ----------
        img: PIL.ImageFile.ImageFile

        Returns
        -------
        numpy.ndarray
        """
        return np.array(img)


def _df_to_bbox_items(
    df: pd.DataFrame,
    content_column="Content",
    annotations_column: str = "Annotations",
    categories: List[str] = None,
) -> List[LabeledImageItem]:
    """Converts pandas dataframe into a list of LabeledImageItem objects.

    Parameters
    ----------
    df: pd.DataFrame
        The Pandas dataframe to convert.
    content_column: Optional[str]
            The column name with the content data.
    annotations_column: Optional[str]
        The column name for the annotations list.
    categories: Optional List[str]
        The list of object categories in proper order for model training.
        Only used when bounding box annotations are in YOLO format.
            Example: ['cat','dog','horse']

    Returns
    -------
    List[LabeledImageItem]
        The list of LabeledImageItem objects.

    Raises
    ------
    TypeError
        If input data is not a pandas dataframe.
    ValueError
        If input data has a wrong format.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The parameter `df` must be a Pandas dataframe.")
    if content_column not in list(df.columns):
        raise ValueError(
            "Wrong format of input dataframe. It must have "
            f"`{content_column}` column."
        )
    if annotations_column not in list(df.columns):
        raise ValueError(
            "Wrong format of input dataframe. It must have "
            f"`{annotations_column}` column."
        )

    if df[content_column].isnull().values.any():
        logger.warning(
            "The source Dataframe includes records where content is not loaded. "
            "Use `materialize=True` to load the content. "
            "The records with empty content will be ignored."
        )

    result = []
    for item in df.T.to_dict().values():
        if item[annotations_column] and not isinstance(item[annotations_column], list):
            raise ValueError(
                "The parameter `df` is invalid. "
                f"The column {annotations_column}  must be of type `List[BoundingBoxItem]`."
            )
        if item[content_column]:
            if (
                isinstance(item[annotations_column][0], list)
                and item[annotations_column][0][0]
                and isinstance(item[annotations_column][0][0], Tuple)
                and len(item[annotations_column][0][0]) == 5
            ):
                bbox_items = [
                    BoundingBoxItem.from_yolo(bbox, categories)
                    for bbox in item[annotations_column]
                ]
            else:
                bbox_items = item[annotations_column] or []
            result.append(LabeledImageItem(item[content_column], bbox_items))

    return result


def render(
    items: List[LabeledImageItem], options: Dict = None, path: str = None
) -> None:
    """Render image dataset.

    Parameters
    ----------
    items: List[LabeledImageItem]
        The list of LabeledImageItem to render.
    options: dict, optional
        The options for rendering.
    path: str
        Path to save the images with annotations to local directory.

    Returns
    -------
    None
        Nothing.

    Raises
    ------
    ValueError
        If items not provided.
        If path is not valid.
    TypeError
        If items provided in a wrong format.

    Examples
    --------
    >>> bbox1 = BoundingBoxItem(bottom_left=(0.3, 0.4),
    >>>                        top_left=(0.3, 0.09),
    >>>                        top_right=(0.86, 0.09),
    >>>                        bottom_right=(0.86, 0.4),
    >>>                        labels=['dolphin', 'fish'])

    >>> record1 = LabeledImageItem(img_obj1, [bbox1])
    >>> render(items = [record1])
    """
    if not items:
        raise ValueError("The parameter `items` is required.")

    if not isinstance(items, list) or not all(
        isinstance(x, LabeledImageItem) for x in items
    ):
        raise TypeError(
            "Wrong format for the items. The items must be `List[LabeledImageItem]`."
        )

    for idx, item in enumerate(items):
        if not path:
            ImageLabeledDataFormatter.render_item(item, options)
        else:
            if os.path.isdir(path):
                ImageLabeledDataFormatter.render_item(
                    item, options, path=f"{path}_{idx+1}.jpg"
                )
            elif not path.lower().endswith(tuple(IMG_FORMAT)):
                img_format_str = (
                    IMG_FORMAT[0]
                    if len(IMG_FORMAT) == 1
                    else ", ".join(IMG_FORMAT[:-1] + ["and " + IMG_FORMAT[-1]])
                )
                raise ValueError(
                    f"Invalid {path}. It is not a directory or the image format "
                    f"in {path} is not supported. Currently the support "
                    f"types are {img_format_str}."
                )
            else:
                img_type = path.split(".")[-1]
                path_root = ".".join(path.split(".")[:-1])
                new_path = f"{path_root}_{idx+1}.{img_type}"
                ImageLabeledDataFormatter.render_item(item, options, path=new_path)
