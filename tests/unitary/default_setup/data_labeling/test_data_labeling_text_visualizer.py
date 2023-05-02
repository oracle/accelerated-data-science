#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import patch

import pandas as pd
import pytest
from ads.data_labeling.ner import NERItem, NERItems
from ads.data_labeling.visualizer.text_visualizer import (
    TextLabeledDataFormatter,
    LabeledTextItem,
    RenderOptions,
    _df_to_ner_items,
    render,
)


class TestLabeledTextItem:
    """Unittests for LabeledTextItem class."""

    MOCK_TXT = "London is the capital of the United Kingdom."

    def test__validate_success(self):
        """Ensures validation passes in case of valid input parameters."""
        test_item = LabeledTextItem(
            self.MOCK_TXT, [NERItem("city", 0, 6), NERItem("country", 29, 14)]
        )
        assert test_item.txt == "London is the capital of the United Kingdom."
        assert test_item.ents == [NERItem("city", 0, 6), NERItem("country", 29, 14)]

    def test__validate_fail_with_empty_txt(self):
        """Ensures validation fails in case of invalid input parameters."""
        with pytest.raises(ValueError):
            LabeledTextItem("", [])


class TestRenderOptions:
    """Unittests for TestRenderOptions class."""

    MOCK_VALID_OPTIONS = {
        "default_color": "#DDEECC",
        "colors": {"yes": "#DDEECC", "no": "#FFAAAA"},
    }

    def test__validate_success(self):
        """Ensures validation passes in case of valid input parameters."""
        RenderOptions._validate(self.MOCK_VALID_OPTIONS)

    @pytest.mark.parametrize(
        "test_options",
        [
            {"color": "#DDEECC"},
            {"default_color": "#DDEECC", "invalid_key": ""},
            {"default_color": "#DDEECC", "colors": "invalid value"},
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

    def test_from_dict_fail(sels):
        """Ensures instantiating RwnderOptions from dict fails in case of invalid input parameters."""
        with pytest.raises(ValueError):
            RenderOptions.from_dict({"color": "#DDEECC"})

    def test_to_dict(self):
        """Tests converting RenderOptions instance to dictionary format."""
        render_options = RenderOptions.from_dict(self.MOCK_VALID_OPTIONS)
        assert render_options.to_dict() == self.MOCK_VALID_OPTIONS


class TestTextLabeledDataFormatter:
    """Unittests for TestTextLabeledDataFormatter class."""

    def test_render_ner_item(self):
        test_record = LabeledTextItem(
            "Houston area contractor seeking a Sheet Metal Superintendent to supervise a crew of sheet metal mechanics.",
            [NERItem("yes", 0, 12), NERItem("yes", 13, 10)],
        )
        result = TextLabeledDataFormatter.render(items=[test_record])
        assert (
            result == '<div key=0 style="margin-top:10px; line-height:2em"><span style='
            '"background-color: #cedddd; padding: 5px; margin: 0px 5px; border-radius: '
            '5px;"><span style="margin-right: 5px;">Houston area</span><span style="text-transform:'
            ' uppercase; font-weight: bold; font-size:0.8em;">yes</span></span> <span style="background-color:'
            ' #cedddd; padding: 5px; margin: 0px 5px; border-radius: 5px;"><span style="margin-right: 5px;">'
            'contractor</span><span style="text-transform: uppercase; font-weight: bold; font-size:0.8em;">yes</span></span>'
            " seeking a Sheet Metal Superintendent to supervise a crew of sheet metal mechanics.</div>"
        )

    def test_render_ner_item_with_customized_color(self):
        test_record = LabeledTextItem(
            "Seeking an Electrical Engineer or high level Electrical Tech.",
            [NERItem("yes", 8, 3)],
        )

        result = TextLabeledDataFormatter.render(
            items=[test_record],
            options={
                "default_color": "#DDEECC",
                "colors": {"yes": "#DDEECC", "no": "#FFAAAA"},
            },
        )
        assert (
            result
            == '<div key=0 style="margin-top:10px; line-height:2em">Seeking <span style="background-color: #DDEECC; '
            'padding: 5px; margin: 0px 5px; border-radius: 5px;"><span style="margin-right: 5px;">an </span><span style="text-transform:'
            ' uppercase; font-weight: bold; font-size:0.8em;">yes</span></span>Electrical Engineer or high level Electrical Tech.</div>'
        )

    @pytest.mark.xfail(raises=AssertionError)
    def test_render_ner_item_with_overlapping_entities(self):
        test_record = LabeledTextItem(
            "Houston area contractor seeking a Sheet Metal Superintendent to supervise a crew of sheet metal mechanics.",
            [NERItem("yes", 0, 12), NERItem("yes", 10, 13)],
        )
        TextLabeledDataFormatter.render(items=[test_record])

    def test_render_ner_item_with_unsorted_entities(self):
        test_record = LabeledTextItem(
            "Houston area contractor seeking a Sheet Metal Superintendent to supervise a crew of sheet metal mechanics.",
            [NERItem("yes", 13, 10), NERItem("yes", 0, 12)],
        )
        result = TextLabeledDataFormatter.render(items=[test_record])
        assert (
            result == '<div key=0 style="margin-top:10px; line-height:2em"><span style='
            '"background-color: #cedddd; padding: 5px; margin: 0px 5px; border-radius: '
            '5px;"><span style="margin-right: 5px;">Houston area</span><span style="text-transform:'
            ' uppercase; font-weight: bold; font-size:0.8em;">yes</span></span> <span style="background-color:'
            ' #cedddd; padding: 5px; margin: 0px 5px; border-radius: 5px;"><span style="margin-right: 5px;">'
            'contractor</span><span style="text-transform: uppercase; font-weight: bold; font-size:0.8em;">yes</span></span>'
            " seeking a Sheet Metal Superintendent to supervise a crew of sheet metal mechanics.</div>"
        )


class TestNerVisualizer:
    @patch.object(TextLabeledDataFormatter, "render", return_value=None)
    def test_render(self, mock_render):
        """Tests rendering NER dataset to Html format."""
        test_record = LabeledTextItem(
            "Houston area contractor seeking a Sheet Metal Superintendent to supervise a crew of sheet metal mechanics.",
            [NERItem("yes", 13, 10), NERItem("yes", 0, 12)],
        )
        test_options = {
            "default_color": "#DDEECC",
            "colors": {"yes": "#DDEECC", "no": "#FFAAAA"},
        }

        render([test_record], test_options)

        mock_render.assert_called_with([test_record], test_options)

    def test__df_to_ner_items_success(self):
        test_item = [
            "Houston area contractor seeking a Sheet Metal Superintendent to supervise a crew of sheet metal mechanics.",
            [
                NERItem(label="yes", offset=0, length=12),
                NERItem(label="yes", offset=13, length=10),
            ],
        ]

        expected_result = LabeledTextItem(
            "Houston area contractor seeking a Sheet Metal Superintendent to supervise a crew of sheet metal mechanics.",
            [NERItem("yes", 0, 12), NERItem("yes", 13, 10)],
        )

        test_df = pd.DataFrame([test_item], columns=["Content", "Annotations"])
        assert _df_to_ner_items(
            test_df, content_column="Content", annotations_column="Annotations"
        ) == [expected_result]

    def test__df_to_ner_items_fail(self):
        """Ensures converting DF to NER items fails in case of wrong input DF."""

        with pytest.raises(TypeError):
            _df_to_ner_items("wrong input fromat")

        test_df = pd.DataFrame([], columns=["Content", "Annotations"])
        with pytest.raises(ValueError):
            _df_to_ner_items(
                test_df, content_column="wrong_name", annotations_column="Annotations"
            )

        test_df = pd.DataFrame([], columns=["Content", "Annotations"])
        with pytest.raises(ValueError):
            _df_to_ner_items(
                test_df, content_column="Content", annotations_column="wrong_column"
            )

    def test__df_to_ner_items_from_spacy(self):
        test_item = [
            "Houston area contractor seeking a Sheet Metal Superintendent to supervise a crew of sheet metal mechanics.",
            [(0.0, 12.0, "yes"), (13, 15, "yes")],
        ]

        expected_result = LabeledTextItem(
            "Houston area contractor seeking a Sheet Metal Superintendent to supervise a crew of sheet metal mechanics.",
            [NERItem("yes", 0.0, 12.0), NERItem("yes", 13, 2)],
        )

        test_df = pd.DataFrame([test_item], columns=["Content", "Annotations"])
        ner_items = _df_to_ner_items(
            test_df, content_column="Content", annotations_column="Annotations"
        )
        for item, expected_item in zip(ner_items[0].ents, expected_result.ents):
            for field in expected_result.ents[0].__dict__.keys():
                assert getattr(item, field) == getattr(expected_item, field)
