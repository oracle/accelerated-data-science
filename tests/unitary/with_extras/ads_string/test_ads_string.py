#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.feature_engineering.feature_type.adsstring.string import ADSString
from ads.feature_engineering.feature_type.adsstring.parsers.spacy_parser import (
    SpacyParser,
)
from unittest.mock import patch, call
import io
import nltk
import os
import sys


class CustomPlugin(object):
    def __init__(self):
        self.constant = 42

    @property
    def magic(self):
        return self.constant


class AnotherCustomPlugin(object):
    def __init__(self):
        self.constant_2 = 24

    @property
    def magic_2(self):
        return self.constant_2


class YetAnotherCustomPlugin(object):
    def __init__(self):
        self.constant_3 = 6

    @property
    def magic_3(self):
        return self.constant_3


class TestADSString:
    def setup_class(self):
        nltk.download("punkt", download_dir=os.environ["CONDA_PREFIX"] + "/nltk")
        nltk.download(
            "averaged_perceptron_tagger",
            download_dir=os.environ["CONDA_PREFIX"] + "/nltk",
        )

    def test_basic_string_methods(self):
        s = ADSString("HELLO WORLD")
        assert "lower" in dir(s)
        assert "split" in dir(s)
        assert "replace" in dir(s)

        s2 = s.lower().upper()
        assert s2 == s and isinstance(s2, ADSString)
        assert " ".join(s.split()) == s
        s3 = s.replace("L", "N").replace("N", "L")
        assert s3 == s and isinstance(s3, ADSString)

    def test_nlp_methods(self):
        ADSString.nlp_backend("nltk")
        s = ADSString("Walking my dog on a breezy day is the best way to recharge.")
        assert list(s.adjective) == ["breezy", "best"]
        assert list(s.noun) == ["dog", "day", "way"]
        assert list(s.adverb) == []
        assert list(s.verb) == ["Walking", "is", "recharge"]
        assert len(s.token) == 14
        assert len(s.sentence) == 1
        assert s.pos.values.tolist()[0] == ["Walking", "VBG"]
        assert all([count == 1 for _, count in s.word_count.values.tolist()])
        assert s.bigram.values.tolist()[0] == ["walking", "my"]
        assert s.trigram.values.tolist()[0] == ["walking", "my", "dog"]

    def test_spacy_method(self):
        from spacy.cli import download

        download("en_core_web_sm")
        ADSString.nlp_backend("spacy")
        assert SpacyParser in ADSString.plugins
        s = ADSString(
            "The Indian Space Research Organisation or is the national space agency of India, headquartered in Bengaluru."
        )
        assert SpacyParser in ADSString.plugins
        assert s.entity_location == ["India", "Bengaluru"]
        assert s.entity_extract.values.tolist() == [
            ["The Indian Space Research Organisation", "ORG"],
            ["India", "GPE"],
            ["Bengaluru", "GPE"],
        ]
        ADSString.plugin_clear()
        assert SpacyParser not in ADSString.plugins

    def test_plugin_register(self):
        ADSString.plugin_register(CustomPlugin)
        ADSString.plugin_register(AnotherCustomPlugin)
        ADSString.plugin_register(YetAnotherCustomPlugin)

        s = ADSString("HELLO WORLD")
        assert "magic" in dir(s)
        assert "magic_2" in dir(s)
        assert "magic_3" in dir(s)

        assert s.magic == 42
        assert s.magic_2 == 24
        assert s.magic_3 == 6

        ADSString.plugin_clear()

    def test_clear_plugin(self):
        ADSString.plugin_clear()
        ADSString.plugin_register(CustomPlugin)
        assert "magic" in dir(ADSString("a"))
        ADSString.plugin_clear()
        assert "magic" not in dir(ADSString("a"))

    @patch("builtins.print")
    def test_plugin_list_no_plugin(self, mocked_print):
        ADSString.plugin_clear()
        ADSString.plugin_list()
        assert mocked_print.mock_calls == [call("No plugin registered.")]

    def test_plugin_list(self):
        ADSString.plugin_clear()
        registered_plugins = [CustomPlugin, AnotherCustomPlugin, YetAnotherCustomPlugin]
        for plugin in registered_plugins:
            ADSString.plugin_register(plugin)
        output = ADSString.plugin_list()
        for plugin in registered_plugins:
            assert plugin.__name__ in output

        ADSString.plugin_clear()
