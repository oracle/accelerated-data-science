#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
import pytest
from unittest import TestCase

from ads.feature_engineering.feature_type.base import FeatureType, Tag
from ads.feature_engineering.feature_type.continuous import Continuous
from ads.feature_engineering.feature_type.string import String
from ads.feature_engineering.exceptions import TypeNotFound
from ads.feature_engineering.feature_type_manager import FeatureTypeManager


class TestADSSeriesAccessor(TestCase):
    """Unittest for ADSSeriesAccessor class."""

    def setUp(self):
        """Sets up the test case."""
        super(TestADSSeriesAccessor, self).setUp()
        FeatureTypeManager.feature_type_reset()
        self.ser = pd.Series(data=["a", "b", "c", "d", "e"], name="str")

    def test_default_feature_type(self):
        assert self.ser.ads.default_type == "string"
        assert self.ser.ads.feature_type == ["string"]
        self.ser.ads.feature_type = [Continuous]
        assert self.ser.ads.feature_type == ["continuous", "string"]
        self.ser.ads.feature_type = ["string", "continuous"]
        assert self.ser.ads.feature_type == ["string", "continuous"]
        self.ser.ads.feature_type = []
        assert self.ser.ads.feature_type == ["string"]

    def test_add_and_remove_feature_type(self):
        self.ser.ads._add_feature_type(Continuous)
        assert self.ser.ads.feature_type == ["string", "continuous"]
        assert "continuous" in self.ser.ads.feature_type
        assert self.ser.ads.feature_type[-1] == "continuous"
        self.ser.ads._remove_feature_type(Continuous)
        assert "continuous" not in self.ser.ads.feature_type
        with pytest.raises(TypeNotFound):
            self.ser.ads._remove_feature_type("continuous")

        class NewType(FeatureType):
            pass

        with pytest.raises(TypeNotFound):
            self.ser.ads._add_feature_type(NewType)

        class NewType2:
            pass

        with pytest.raises(TypeError):
            self.ser.ads._add_feature_type(NewType2())

        with pytest.raises(TypeNotFound):
            self.ser.ads._add_feature_type("xxx")

    def test_add_and_remove_generic_tag(self):
        self.ser.ads._add_feature_type(Tag("abc"))
        assert "abc" in self.ser.ads.feature_type
        self.ser.ads._remove_feature_type("abc")
        assert "abc" not in self.ser.ads.feature_type

        self.ser.ads._add_feature_type(Tag("ab c"))
        # assert self.ser.ads.implements_ab_c
        self.ser.ads._remove_feature_type(Tag("ab c"))
        assert "ab c" not in self.ser.ads.feature_type

        with pytest.raises(TypeNotFound):
            self.ser.ads._remove_feature_type(Tag("def"))

    def test_set_feature_type(self):
        self.ser.ads.feature_type = [Continuous, "ordinal"]
        assert self.ser.ads.feature_type[0] == "continuous"
        assert self.ser.ads.feature_type[1] == "ordinal"

    def test_set_feature_type_fail(self):
        expected_result = TypeError("Argument must be a list of feature types.")
        with pytest.raises(TypeError) as exc:
            self.ser.ads.feature_type = "wrong input data"
        assert str(exc.value) == str(expected_result)

    def test_sync(self):
        default_type = self.ser.ads.default_type
        old_types = [default_type]
        ser = self.ser.copy()
        self.ser.ads.feature_type = [Tag("abc"), Tag("def")]

        assert ser.ads.feature_type == old_types
        ser.ads.sync(self.ser)
        assert ser.ads.feature_type == [default_type, "abc", "def"]

        ser2 = self.ser.dropna()
        assert ser2.ads.feature_type == old_types
        ser2.ads.sync(self.ser)
        assert ser2.ads.feature_type == [default_type, "abc", "def"]

        df = pd.DataFrame(
            data={
                "str": ["a", "b", "c", "d", "e"],
                "num": [1.0, 1.1, 1.2, 1.3, 1.4],
            }
        )

        self.ser.ads.sync(df)
        assert self.ser.ads.feature_type == ["string"]

    def test_sync_private(self):
        default_type = self.ser.ads.default_type
        old_types = [default_type]
        ser = self.ser.copy()
        self.ser.ads.feature_type = [Tag("abc"), Tag("def")]

        assert ser.ads.feature_type == old_types
        ser.ads._sync(self.ser)
        assert ser.ads.feature_type == [default_type, "abc", "def"]

        ser2 = self.ser.dropna()
        assert ser2.ads.feature_type == old_types
        ser2.ads._sync(self.ser)
        assert ser2.ads.feature_type == [default_type, "abc", "def"]

    def test_invoke_methods(self):
        class NewType(FeatureType):
            @staticmethod
            def identity(x):
                return x.values

            def identity2(self, x):
                return x.values

            @classmethod
            def identity3(cls, x):
                return x.values

        FeatureTypeManager.feature_type_register(NewType)
        self.ser.ads._add_feature_type(NewType())
        assert all(
            v1 == v2
            for v1, v2 in zip(self.ser.ads.identity(), ["a", "b", "c", "d", "e"])
        )
        assert all(
            v1 == v2
            for v1, v2 in zip(self.ser.ads.identity2(), ["a", "b", "c", "d", "e"])
        )
        assert all(
            v1 == v2
            for v1, v2 in zip(self.ser.ads.identity3(), ["a", "b", "c", "d", "e"])
        )
        self.ser.ads._remove_feature_type(NewType())
        FeatureTypeManager.feature_type_unregister(NewType)

    def test_help(self):
        class NewType(FeatureType):
            @staticmethod
            def new_method(x):
                """This is a docstring"""
                return x.values

        FeatureTypeManager.feature_type_register(NewType)
        self.ser.ads._add_feature_type("new_type")
        self.ser.ads.help("new_method")
        self.ser.ads.help()
        FeatureTypeManager.feature_type_unregister(NewType)

    def test_feature_type_description(self):
        """Tests getting the list of registered feature types in a DataFrame format."""

        class MyType(FeatureType):
            description = "My custom type"

        FeatureTypeManager.feature_type_register(MyType)
        self.ser.ads.feature_type = [MyType, Tag("TestTag")]

        expected_result = pd.DataFrame(
            (
                (MyType.name, MyType.description),
                (String.name, String.description),
                ("TestTag", "Tag"),
            ),
            columns=["Feature Type", "Description"],
        )
        assert pd.DataFrame.equals(
            expected_result, self.ser.ads.feature_type_description
        )
