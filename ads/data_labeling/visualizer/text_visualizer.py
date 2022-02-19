#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that helps to visualize NER Text Dataset.

Methods
-------
    render(items: List[LabeledTextItem], options: Dict = None) -> str
        Renders NER dataset to Html format.

Examples
--------
>>> record1 = LabeledTextItem("London is the capital of the United Kingdom", [NERItem('city', 0, 6), NERItem("country", 29, 14)])
>>> record2 = LabeledTextItem("Houston area contractor seeking a Sheet Metal Superintendent.", [NERItem("city", 0, 6)])
>>> result = render(items = [record1, record2], options={"default_color":"#DDEECC", "colors": {"city":"#DDEECC", "country":"#FFAAAA"}})
>>> display(HTML(result))
"""

import logging
from dataclasses import asdict, dataclass
from string import Template
from typing import Dict, List, Optional

import pandas as pd
from ads.data_labeling.constants import AnnotationType
from ads.data_labeling.ner import NERItem
from cerberus import Validator


logger = logging.getLogger(__name__)

HTML_OPTIONS_SCHEMA = {
    "default_color": {"nullable": True, "required": False, "type": "string"},
    "colors": {
        "nullable": True,
        "required": False,
        "type": "dict",
    },
}

DEFAULT_COLOR = "#cedddd"
LEN_OF_SPACY_ITEM = 3


@dataclass
class LabeledTextItem:
    """Data class representing NER Item.

    Attributes
    ----------
    txt: str
        The labeled sentence.
    ents: List[NERItem]
        The list of entities.
    """

    txt: str
    ents: List[NERItem]

    def _validate(self):
        """Validates the instance.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        ValueError
            If txt is empty.
        WrongEntityFormat
            If the list of entities has a wrong format.
        AssertionError
            In case of overlapped entities.
        """
        if not self.txt:
            raise ValueError(
                "The parameter `txt` is required and must not be an empty string."
            )

        if not isinstance(self.ents, List):
            raise ValueError(
                "Invalid format for the entities. The entities must be a List[NERItem]."
            )

        for entity in self.ents:
            if entity.offset + entity.length > len(self.txt):
                raise ValueError(
                    f"At least one of the entities (start index: {entity.length}, offset: {entity.offset}) "
                    f"exceeds the length of the text ({len(self.txt)})."
                )

        self.ents.sort(key=lambda x: x.offset)

        for i in range(len(self.ents) - 1):
            if self.ents[i].offset + self.ents[i].length >= self.ents[i + 1].offset:
                raise AssertionError(
                    "The entity data contains overlapping tokens. The first token has a start index"
                    f" of {self.ents[i].length}, and an offset of {self.ents[i].offset}. The second token has a start "
                    f"index of {self.ents[i + 1].length}, and an offset of {self.ents[i + 1].offset}. "
                )

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
        """
        if not options:
            return None

        validator = Validator(HTML_OPTIONS_SCHEMA)
        valid = validator.validate(options)
        if not valid:
            raise ValueError(validator.errors)

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
        return self.to_dict()


class TextLabeledDataFormatter:
    """The TextLabeledDataFormatter class to render NER items into Html format."""

    _ITEM_TEMPLATE = (
        '<span style="background-color: $color; padding: 5px; margin: 0px 5px; border-radius: 5px;">'
        '<span style="margin-right: 5px;">$entity</span>'
        '<span style="text-transform: uppercase; font-weight: bold; font-size:0.8em;">$label</span>'
        "</span>"
    )
    _ROW_TEMPLATE = '<div key=$key style="margin-top:10px; line-height:2em">$row</div>'

    @staticmethod
    def render(items: List[LabeledTextItem], options: Dict = None) -> str:
        """Renders NER dataset to Html format.

        Parameters
        ----------
        items: List[LabeledTextItem]
            Items to render.
        options: Optional[dict]
            Render options.

        Returns
        -------
        str
            Html representation of rendered NER dataset.

        Raises
        ------
        ValueError
            If items not provided.
        TypeError
            If items provided in a wrong format.
        """
        if not items:
            raise ValueError("The parameter `items` is required.")

        if not isinstance(items, list) or not all(
            isinstance(x, LabeledTextItem) for x in items
        ):
            raise TypeError(
                "Wrong format for the items. Items should be a `List[LabeledTextItem]`."
            )

        render_options = RenderOptions.from_dict(options)
        item_template = Template(TextLabeledDataFormatter._ITEM_TEMPLATE)
        row_template = Template(TextLabeledDataFormatter._ROW_TEMPLATE)
        result = []

        for item_index, item in enumerate(items):
            current_index = 0
            accum = []
            for e in item.ents:
                start = e.offset
                end = e.offset + e.length
                label = e.label
                accum.append(item.txt[current_index:start])
                accum.append(
                    item_template.substitute(
                        {
                            "color": render_options.colors.get(
                                label, render_options.default_color
                            ),
                            "entity": item.txt[start:end],
                            "label": label,
                        }
                    )
                )
                current_index = end

            accum.append(item.txt[current_index : len(item.txt)])
            result.append(
                row_template.substitute({"key": str(item_index), "row": "".join(accum)})
            )

        return "".join(result)


def _df_to_ner_items(
    df: pd.DataFrame,
    content_column: str = "Content",
    annotations_column: str = "Annotations",
) -> List[LabeledTextItem]:
    """Converts pandas dataframe into a list of LabeledTextItem.

    Parameters
    ----------
    df: pd.DataFrame
        The Pandas dataframe to convert.
    content_column: Optional[str]
            The column name with the content data.
    annotations_column: Optional[str]
        The column name for the annotations list.

    Returns
    -------
    List[LabeledTextItem]
        The list of LabeledTextItem objects.

    Raises
    ------
    TypeError
        If input data not a pandas dataframe.
    ValueError
        If input data has wrong format.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The parameter `df` must be a Pandas dataframe.")
    if content_column not in list(df.columns):
        raise ValueError(
            f"The parameter `df` is invalid. It must have a column named `{content_column}`."
        )
    if annotations_column not in list(df.columns):
        raise ValueError(
            f"The parameter `df` is invalid. It must have a column named `{annotations_column}`."
        )

    if df[content_column].isnull().values.any():
        logger.warning(
            "The parameter `df` includes records where the text content is not "
            "materialized. These records will be ignored. Use `materialize=True` "
            "to load the content."
        )

    result = []
    for item in df.T.to_dict().values():
        if item[annotations_column] and not isinstance(item[annotations_column], list):
            raise ValueError(
                f"The parameter `df` is invalid. The column {annotations_column} "
                "must be of type `List[NERItem]`."
            )

        if item[content_column]:
            if (
                isinstance(item[annotations_column][0], tuple)
                and len(item[annotations_column][0]) == LEN_OF_SPACY_ITEM
            ):
                ents = [NERItem.from_spacy(ent) for ent in item[annotations_column]]
            else:
                ents = item[annotations_column] or []
            result.append(
                LabeledTextItem(
                    txt=item[content_column],
                    ents=ents,
                )
            )

    return result


def render(items: List[LabeledTextItem], options: Dict = None) -> str:
    """Renders NER dataset to Html format.

    Parameters
    ----------
    items: List[LabeledTextItem]
        The list of NER items to render.
    options: dict, optional
        The options for rendering.

    Returns
    -------
    str
        Html string.

    Examples
    --------
    >>> record = LabeledTextItem("London is the capital of the United Kingdom", [NERItem('city', 0, 6), NERItem("country", 29, 14)])
    >>> result = render(items = [record], options={"default_color":"#DDEECC", "colors": {"city":"#DDEECC", "country":"#FFAAAA"}})
    >>> display(HTML(result))
    """
    return TextLabeledDataFormatter.render(items, options)
