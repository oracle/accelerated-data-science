#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from unittest import TestCase

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from ads.feature_engineering.accessor.mixin.correlation import (
    _list_to_dataframe,
    cat_vs_cat,
    cat_vs_cont,
    cont_vs_cont,
)
from ads.feature_engineering.accessor.mixin.utils import (
    _categorical_columns,
    _continuous_columns,
)
from ads.feature_engineering.feature_type.phone_number import PhoneNumber
from ads.feature_engineering.feature_type_manager import FeatureTypeManager


class TestEDAMixin(TestCase):
    def setUp(self):
        """Sets up the test case."""
        super().setUp()
        FeatureTypeManager.feature_type_reset()

    data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "survived": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        "pass_class": [1, 3, 1, 1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
        "name": [
            "ad fb",
            "tde",
            "uf di",
            "kl",
            "tfdfr",
            "uy",
            "iuf df",
            "mi",
            "uhky",
            "pa",
            "pkjp",
            "wsdfe",
            "ifffi",
            "btqq",
            "gg",
            "tdfsl",
            "mn",
            "ldfro",
            "qajk",
            "kkby",
        ],
        "sex": [
            "x",
            "y",
            "y",
            "x",
            "y",
            "x",
            "x",
            "x",
            "y",
            "y",
            "x",
            "x",
            "x",
            "x",
            "x",
            "y",
            "x",
            "x",
            "x",
            "y",
        ],
        "age": [
            15,
            63,
            32,
            25,
            56,
            82,
            28,
            38,
            40,
            34,
            32,
            22,
            9,
            37,
            48,
            52,
            50,
            44,
            3,
            28,
        ],
        "siblings": np.random.randint(10, size=20),
        "fare": 100 * np.random.rand(20) + 10,
        "cabin": [
            "NaN",
            "NaN",
            "M89",
            "NaN",
            "O99",
            "NaN",
            "G77 S34",
            "S345",
            "NaN",
            "NaN",
            "NaN",
            "NaN",
            "L8",
            "NaN",
            "NaN",
            "NaN",
            "NaN",
            "NaN",
            "NaN",
            "NaN",
        ],
    }
    df = pd.DataFrame(data=data)
    continuous_cols = ["id", "pass_class", "age", "siblings", "fare"]
    categorical_cols = ["survived", "name", "sex", "cabin"]

    cols_tuples = [("a", "b"), ("a", "c")]
    corr_list = [1, 2]

    def test_cont_vs_cont(self):
        df = cont_vs_cont(self.df[self.continuous_cols])
        for col in self.continuous_cols:
            assert (
                (df["Column 1"] == col) & (df["Column 2"] == col) & (df["Value"] == 1.0)
            ).any()
        assert (
            (df["Column 1"] == "id")
            & (df["Column 2"] == "pass_class")
            & (df["Value"] == 0.6764)
        ).any()
        assert (
            (df["Column 1"] == "pass_class")
            & (df["Column 2"] == "id")
            & (df["Value"] == 0.6764)
        ).any()
        assert not ((df["Column 1"] == "name") & (df["Column 2"] == "cabin")).any()

    def test_cat_vs_cat(self):
        df = cat_vs_cat(self.df[self.categorical_cols])
        for col in self.categorical_cols:
            assert (
                (df["Column 1"] == col) & (df["Column 2"] == col) & (df["Value"] == 1.0)
            ).any()
        assert (
            (df["Column 1"] == "cabin")
            & (df["Column 2"] == "sex")
            & (df["Value"] == 0.0669)
        ).any()
        assert (
            (df["Column 1"] == "sex")
            & (df["Column 2"] == "cabin")
            & (df["Value"] == 0.0669)
        ).any()
        assert not ((df["Column 1"] == "name") & (df["Column 2"] == "cabin")).any()

    def test_cat_vs_cont(self):
        df = cat_vs_cont(self.df, self.categorical_cols, self.continuous_cols)
        for col in self.df.columns:
            assert (
                (df["Column 1"] == col) & (df["Column 2"] == col) & (df["Value"] == 1.0)
            ).any()
        assert (
            (df["Column 1"] == "survived")
            & (df["Column 2"] == "age")
            & (df["Value"] == 0.0420)
        ).any()
        assert (
            (df["Column 1"] == "age")
            & (df["Column 2"] == "survived")
            & (df["Value"] == 0.0420)
        ).any()
        assert not ((df["Column 1"] == "fare") & (df["Column 2"] == "age")).any()

    def test_correlation_value_range(self):
        for df in [
            cat_vs_cat(self.df[self.categorical_cols]),
            cat_vs_cont(self.df, self.categorical_cols, self.continuous_cols),
            cont_vs_cont(self.df[self.continuous_cols]),
        ]:
            assert not (df["Value"] > 1).any() and not (df["Value"] < -1).any()

    def test_list_to_dataframe_nf(self):
        nf = _list_to_dataframe(self.cols_tuples, self.corr_list, True)
        assert len(nf.columns) == 3
        assert "Column 1" in nf.columns
        assert "Column 2" in nf.columns
        assert "Value" in nf.columns
        assert (
            (nf["Column 1"] == "a") & (nf["Column 2"] == "a") & (nf["Value"] == 1.0)
        ).any()
        assert (
            (nf["Column 1"] == "a") & (nf["Column 2"] == "b") & (nf["Value"] == 1.0)
        ).any()
        assert (
            (nf["Column 1"] == "c") & (nf["Column 2"] == "a") & (nf["Value"] == 2.0)
        ).any()
        assert not ((nf["Column 1"] == "c") & (nf["Column 2"] == "b")).any()

    def test_list_to_dataframe_not_nf(self):
        df = _list_to_dataframe(self.cols_tuples, self.corr_list, False)
        assert len(df.columns) == 3
        assert "a" in df.columns and "b" in df.columns and "c" in df.columns
        assert df["a"]["a"] == 1.0
        assert df["b"]["a"] == 1.0
        assert df["c"]["a"] == 2.0
        assert df["a"]["b"] == 1.0
        assert np.isnan(df["c"]["b"])

    def test_utils_cat_cont_columns(self):
        feature_types = {
            "a": ["category", "ordinal", "Geo"],
            "b": ["continuous", "Geo"],
            "c": ["continuous"],
            "d": ["category"],
            "e": ["Geo"],
            "f": ["Geo", "continuous"],
            "g": ["ordinal", "category"],
        }
        assert _categorical_columns(feature_types) == ["a", "d"]
        assert _continuous_columns(feature_types) == ["b", "c", "f", "g"]

    @pytest.mark.skipif(
        "NoDependency" in os.environ, reason="skip for dependency test: seaborn"
    )
    def test_correlation_plots(self):
        self.df.ads.feature_type = {col: ["category"] for col in self.categorical_cols}

        pearson_plt = self.df.ads.pearson_plot()
        assert isinstance(pearson_plt, mpl.axes._axes.Axes)

        cramersv_plt = self.df.ads.cramersv_plot()
        assert isinstance(cramersv_plt, mpl.axes._axes.Axes)

        corr_plt = self.df.ads.correlation_ratio_plot()
        assert isinstance(corr_plt, mpl.axes._axes.Axes)

    def test_warning(self):
        """Tests generating a data frame that lists feature specific warnings."""
        mock_df = pd.DataFrame(
            {
                "Name": ["Alex", "Liam", "Noah"],
                "PhoneNumber": [
                    "+1-202-555-0141",
                    "+1-202-555-0198",
                    "+1-202-555-0199",
                ],
            }
        )

        mock_df.ads.feature_type = {"PhoneNumber": ["phone_number", "category"]}

        def test_handler1(data):
            return pd.DataFrame(
                [["Zeros", "Age has 38 zeros", "Count", 38]],
                columns=["Warning", "Message", "Metric", "Value"],
            )

        def test_handler2(data):
            return pd.DataFrame(
                [["Zeros", "Age has 12.2 zeros", "Percentage", "12.2%"]],
                columns=["Warning", "Message", "Metric", "Value"],
            )

        PhoneNumber.warning.register("test_warning1", test_handler1)
        PhoneNumber.warning.register("test_warning2", test_handler2)

        expected_df = pd.DataFrame(
            [
                [
                    "Name",
                    "string",
                    "high-cardinality",
                    "every value is distinct",
                    "count",
                    3,
                ],
                [
                    "PhoneNumber",
                    "phone_number",
                    "high-cardinality",
                    "every value is distinct",
                    "count",
                    3,
                ],
                [
                    "PhoneNumber",
                    "phone_number",
                    "Zeros",
                    "Age has 38 zeros",
                    "Count",
                    38,
                ],
                [
                    "PhoneNumber",
                    "phone_number",
                    "Zeros",
                    "Age has 12.2 zeros",
                    "Percentage",
                    "12.2%",
                ],
                [
                    "PhoneNumber",
                    "category",
                    "high-cardinality",
                    "every value is distinct",
                    "count",
                    3,
                ],
                [
                    "PhoneNumber",
                    "string",
                    "high-cardinality",
                    "every value is distinct",
                    "count",
                    3,
                ],
            ],
            columns=["Column", "Feature Type", "Warning", "Message", "Metric", "Value"],
        )

        result_df = mock_df.ads.warning()
        assert pd.DataFrame.equals(expected_df, result_df)
