#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from ads.data_labeling.ner import NERItem, NERItems, WrongEntityFormatLabelNotString
from ads.data_labeling.visualizer.text_visualizer import LabeledTextItem


class TestNERItem:
    """Unittests for TestNERItem class."""

    MOCK_TXT = "London is the capital of the United Kingdom."

    @pytest.mark.parametrize(
        "test_ents",
        [
            [("label", 3, 5, 6)],
            [(1, 100, "label")],
            [(1, 7, 8)],
            [(1, "", "label")],
            [("", 1, "label")],
            [(10, 1, "label")],
            [(-1, 1, "label")],
            [(1, -1, "label")],
        ],
    )
    def test__validate_fail_with_wrong_ents(self, test_ents):
        """Ensures validation fails in case of wrong entities format."""
        with pytest.raises(WrongEntityFormatLabelNotString):
            NERItem(test_ents)

    @pytest.mark.parametrize(
        "test_ents",
        [
            [
                NERItem(label="label", offset=1, length=6),
                NERItem(label="label", offset=8, length=2),
                NERItem(label="label3", offset=2, length=7),
            ],
            [
                NERItem(label="label", offset=6, length=4),
                NERItem(label="label1", offset=1, length=3),
                NERItem(label="label3", offset=1, length=2),
            ],
        ],
    )
    def test__validate_fail_with_nested_ents(self, test_ents):
        """Ensures validation fails in case of nested entities."""
        with pytest.raises(AssertionError):
            LabeledTextItem(self.MOCK_TXT, test_ents)

    @pytest.mark.parametrize(
        "token,converted_token",
        [
            pytest.param(NERItem("yes", 0, 12), (0, 12, "yes")),
            pytest.param(NERItem("no", 1, 3), (1, 4, "no")),
        ],
    )
    def test_spacy_converter_neritem(self, token, converted_token):
        assert token.to_spacy() == converted_token


class TestNERItems:
    @pytest.mark.parametrize(
        "tokens,converted_tokens",
        [
            pytest.param(
                NERItems([NERItem("yes", 0, 12), NERItem("no", 13, 2)]),
                [(0, 12, "yes"), (13, 15, "no")],
            ),
            pytest.param(
                NERItems([NERItem("yes", 1, 2), NERItem("no", 5, 1)]),
                [(1, 3, "yes"), (5, 6, "no")],
            ),
        ],
    )
    def test_spacy_converter_neritems(self, tokens, converted_tokens):
        assert tokens.to_spacy() == converted_tokens

    def test_from_spacy(self):
        spacy_items = [(0, 12, "yes"), (13, 15, "no")]
        expected_items = [NERItem("yes", 0, 12), NERItem("no", 13, 2)]
        ner_items = [NERItem.from_spacy(item) for item in spacy_items]
        for item, expected_item in zip(ner_items, expected_items):
            for attr in item.__dict__.keys():
                assert getattr(item, attr) == getattr(expected_item, attr)
