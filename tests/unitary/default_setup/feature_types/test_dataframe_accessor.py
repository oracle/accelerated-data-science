#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
import pytest
from unittest import TestCase

from ads.feature_engineering.feature_type.base import Tag, FeatureType
from ads.feature_engineering.feature_type.continuous import Continuous
from ads.feature_engineering.feature_type.integer import Integer
from ads.feature_engineering.feature_type.string import String
from ads.feature_engineering.feature_type.boolean import Boolean
from ads.feature_engineering.feature_type_manager import FeatureTypeManager
from ads.feature_engineering.feature_type.adsstring.string import ADSString


class TestADSDataFrameAccessor(TestCase):
    """Unittest for ADSDataFrameAccessor class."""

    def setUp(self):
        """Sets up the test case."""
        super(TestADSDataFrameAccessor, self).setUp()
        FeatureTypeManager.feature_type_reset()
        self.df = pd.DataFrame(
            data={
                "str": ["a", "b", "c", "d", "e"],
                "num": [1.0, 1.1, 1.2, 1.3, 1.4],
                "int": [0, 1, 2, 2, 3],
                "bool": [True, True, False, False, False],
                "txt": ["t1", "t2", "t3", "t4", "t5"],
            }
        )
        self.df.ads.feature_type = {
            "txt": ["ads_string"]
        }  # register txt column as ADSString

    def test_default_feature_type(self):
        assert self.df.ads.feature_type["str"] == ["string"]
        assert self.df.ads.feature_type["num"] == ["continuous"]
        assert self.df.ads.feature_type["int"] == ["integer"]
        assert self.df.ads.feature_type["bool"] == ["boolean"]
        assert self.df.ads.feature_type["txt"] == ["ads_string", "string"]
        assert self.df.ads._feature_type["str"][0] == String
        assert self.df.ads._feature_type["num"][0] == Continuous
        assert self.df.ads._feature_type["int"][0] == Integer
        assert self.df.ads._feature_type["bool"][0] == Boolean
        assert self.df.ads._feature_type["txt"][0] == ADSString

    def test_add_and_remove_feature_type(self):
        self.df.ads._add_feature_type("str", Continuous)
        assert "continuous" in self.df.ads.feature_type["str"]
        self.df.ads._remove_feature_type("str", Continuous)
        assert "continuous" not in self.df.ads.feature_type["str"]
        with pytest.raises(TypeError):
            self.df.ads.remove_feature_type("str", "continuous")

        with pytest.raises(ValueError):
            self.df.ads._add_feature_type("col", "continuous")
        with pytest.raises(ValueError):
            self.df.ads._remove_feature_type("col", "continuous")

    def test_add_and_remove_generic_tag(self):
        self.df.ads._add_feature_type("num", Tag("abc"))
        assert "abc" in self.df.ads.feature_type["num"]
        self.df.ads._remove_feature_type("num", "abc")
        assert "abc" not in self.df.ads.feature_type["num"]

        self.df.ads._add_feature_type("num", Tag("abc"))
        self.df.ads._remove_feature_type("num", Tag("abc"))
        assert "abc" not in self.df.ads.feature_type["num"]

    def test_set_feature_type(self):
        self.df.ads.feature_type = {
            "num": [Continuous, "integer", Tag("abc")],
            "str": [Integer],
        }
        assert len(self.df.ads.feature_type["num"]) == 3
        assert "integer" in self.df.ads.feature_type["num"]
        assert len(self.df.ads.feature_type["str"]) == 2
        assert "integer" in self.df.ads.feature_type["str"]

    def test_sync(self):
        default_type = self.df["num"].ads.default_type
        old_types = [default_type]

        df = self.df.copy()
        self.df.ads.feature_type = {"num": [Tag("abc"), Tag("def")]}
        assert df.ads.feature_type["num"] == old_types
        df.ads.sync(self.df)
        assert df.ads.feature_type["num"] == ["continuous", "abc", "def"]

        df2 = self.df.copy().dropna()
        self.df.ads.feature_type = {"num": [Tag("abc"), Tag("def")]}
        assert df2.ads.feature_type["num"] == old_types
        df2.ads.sync(self.df)
        assert df2.ads.feature_type["num"] == ["continuous", "abc", "def"]

        ser = pd.Series(data=[1, 2, 3, 4, 5], name="num")
        ser.ads.feature_type = ["integer"]
        self.df.ads.sync(ser)
        assert self.df.ads.feature_type["num"] == ["integer", "continuous"]

    def test_invoke_methods(self):
        class NewType(FeatureType):
            @staticmethod
            def add(x, a, b):
                return a + b

            def add2(self, x, a, b):
                return a + b

            @classmethod
            def add3(cls, x, a, b):
                return a + b

        FeatureTypeManager.feature_type_register(NewType)
        self.df.ads._add_feature_type("num", NewType())
        out = self.df.ads.add(1, 2)
        assert out["num"] == 3
        assert not out["str"]
        out = self.df.ads.add2(1, 2)
        assert out["num"] == 3
        assert not out["str"]
        out = self.df.ads.add3(1, 2)
        assert out["num"] == 3
        assert not out["str"]

        self.df.ads._remove_feature_type("num", NewType())
        FeatureTypeManager.feature_type_unregister(NewType)

    def test_help(self):
        class NewType(FeatureType):
            @staticmethod
            def new_method(x):
                """This is a docstring"""
                return x.values

        FeatureTypeManager.feature_type_register(NewType)
        self.df.ads._add_feature_type("num", "new_type")
        self.df.ads.help("new_method")
        self.df.ads.help()
        FeatureTypeManager.feature_type_unregister(NewType)

    def check_columns(self, cols):
        assert len(cols) == 3
        assert "Column 1" in cols
        assert "Column 2" in cols
        assert "Value" in cols

    def test_eda_pearson_columns(self):
        pearson_output = self.df.ads.pearson()
        self.check_columns(pearson_output.columns)

    def test_eda_cramersv_columns(self):
        cramersv_output = self.df.ads.cramersv()
        self.check_columns(cramersv_output.columns)

    def test_eda_correlation_ratio_columns(self):
        correlation_ratio_output = self.df.ads.correlation_ratio()
        self.check_columns(correlation_ratio_output.columns)

    def test_feature_select(self):
        assert set(self.df.ads.feature_select(include=["boolean"]).columns) == set(
            ["bool"]
        )
        assert set(
            self.df.ads.feature_select(exclude=["boolean", Boolean]).columns
        ) == set(["str", "num", "int", "txt"])
        with pytest.raises(ValueError) as excinfo:
            self.df.ads.feature_select(
                self.df.ads.feature_select(include=["boolean"], exclude=[Boolean])
            )
        assert "include and exclude overlap" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            self.df.ads.feature_select(
                self.df.ads.feature_select(include=[], exclude=[])
            )
        assert "at least one of include or exclude must be nonempty" in str(
            excinfo.value
        )

    def test_extract_columns_of_target_types(self):
        class Ordinal1(FeatureType):
            @classmethod
            def isvalid_all(cls, x: pd.Series) -> bool:
                _x = x.dropna()
                return True

        FeatureTypeManager.feature_type_register(Ordinal1)
        self.df.int.ads.feature_type = [Ordinal1]
        assert self.df.ads._extract_columns_of_target_types(["ordinal1", Ordinal1]) == [
            "int"
        ]

    def test_feature_type_description(self):
        """Tests getting the list of registered feature types in a DataFrame format."""
        df = pd.DataFrame(
            data={
                "str": ["a", "b", "c", "d", "e"],
                "num": [1.0, 1.1, 1.2, 1.3, 1.4],
                "txt": ["t1", "t2", "t3", "t4", "t5"],
            }
        )
        df.ads._add_feature_type("str", Tag("TestTag"))
        df.ads.feature_type = {"txt": ["ads_string"]}
        expected_result = pd.DataFrame(
            (
                ("str", String.name, String.description),
                ("str", "TestTag", "Tag"),
                ("num", Continuous.name, Continuous.description),
                ("txt", ADSString.name, ADSString.description),
                ("txt", String.name, String.description),
            ),
            columns=["Column", "Feature Type", "Description"],
        )
        assert pd.DataFrame.equals(expected_result, df.ads.feature_type_description)

    def test_init_fail(self):
        """Ensures init raises exception when DataFrame has duplicate columns."""
        card = ["4532640527811543", "4556929308150929"]
        s = pd.Series(card, name="creditcard")
        df = pd.concat([s, s], axis=1)
        expected_result = ValueError(
            "Failed to initialize a DataFrame accessor. " "Duplicate column found."
        )
        with pytest.raises(ValueError) as exc:
            df.ads
        assert str(expected_result) == str(exc.value)
