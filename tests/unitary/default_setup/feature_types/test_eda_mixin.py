#!/usr/bin/env python
# -*- coding: utf-8 -*--

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
from tests.ads_unit_tests.utils import get_test_dataset_path


class TestEDAMixin(TestCase):
    def setUp(self):
        """Sets up the test case."""
        super().setUp()
        FeatureTypeManager.feature_type_reset()

    df = pd.read_csv(get_test_dataset_path("vor_titanic.csv"))
    continuous_cols = ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_cols = ["Cabin", "Embarked", "Name", "Sex", "Survived", "Ticket"]

    df_weather = pd.read_csv(get_test_dataset_path("vor_delhi_weather.csv"))
    continuous_cols_weather = [
        " _dewptm",
        " _heatindexm",
        " _hum",
        " _pressurem",
        " _tempm",
        " _vism",
        " _wdird",
        " _wspdm",
    ]
    categorical_cols_weather = [
        " _conds",
        " _fog",
        " _hail",
        " _rain",
        " _thunder",
        " _tornado",
        " _wdire",
    ]

    def test_cont_vs_cont_titanic(self):
        df = cont_vs_cont(self.df[self.continuous_cols])
        for col in self.continuous_cols:
            assert (
                (df["Column 1"] == col) & (df["Column 2"] == col) & (df["Value"] == 1.0)
            ).any()
        assert (
            (df["Column 1"] == "PassengerId")
            & (df["Column 2"] == "Pclass")
            & (df["Value"] == -0.0351)
        ).any()
        assert (
            (df["Column 1"] == "Pclass")
            & (df["Column 2"] == "PassengerId")
            & (df["Value"] == -0.0351)
        ).any()
        assert not ((df["Column 1"] == "Name") & (df["Column 2"] == "Cabin")).any()

    def test_cont_vs_cont_weather(self):
        df = cont_vs_cont(self.df_weather[self.continuous_cols_weather])
        for col in self.continuous_cols_weather:
            assert (
                (df["Column 1"] == col) & (df["Column 2"] == col) & (df["Value"] == 1.0)
            ).any()
        assert (
            (df["Column 1"] == " _heatindexm")
            & (df["Column 2"] == " _dewptm")
            & (df["Value"] == 0.4476)
        ).any()
        assert (
            (df["Column 1"] == " _wdird")
            & (df["Column 2"] == " _wspdm")
            & (df["Value"] == 0.2473)
        ).any()

    def test_cat_vs_cat(self):
        df = cat_vs_cat(self.df[self.categorical_cols])
        for col in self.categorical_cols:
            assert (
                (df["Column 1"] == col) & (df["Column 2"] == col) & (df["Value"] == 1.0)
            ).any()
        assert (
            (df["Column 1"] == "Cabin")
            & (df["Column 2"] == "Sex")
            & (df["Value"] == 0.1372)
        ).any()
        assert (
            (df["Column 1"] == "Sex")
            & (df["Column 2"] == "Cabin")
            & (df["Value"] == 0.1372)
        ).any()
        assert not ((df["Column 1"] == "Name") & (df["Column 2"] == "Cabin")).any()

    def test_cat_vs_cat_weather(self):
        df = cat_vs_cat(self.df_weather[self.categorical_cols_weather])
        for col in self.categorical_cols_weather:
            assert (
                (df["Column 1"] == col) & (df["Column 2"] == col) & (df["Value"] == 1.0)
            ).any()

    def test_cat_vs_cont(self):
        df = cat_vs_cont(self.df, self.categorical_cols, self.continuous_cols)
        for col in self.df.columns:
            assert (
                (df["Column 1"] == col) & (df["Column 2"] == col) & (df["Value"] == 1.0)
            ).any()
        assert (
            (df["Column 1"] == "Embarked")
            & (df["Column 2"] == "Age")
            & (df["Value"] == 0.0423)
        ).any()
        assert (
            (df["Column 1"] == "Age")
            & (df["Column 2"] == "Embarked")
            & (df["Value"] == 0.0423)
        ).any()
        assert not ((df["Column 1"] == "Fare") & (df["Column 2"] == "Age")).any()

    def test_cat_vs_cont_weather(self):
        df = cat_vs_cont(
            self.df_weather, self.categorical_cols_weather, self.continuous_cols_weather
        )
        for col in self.categorical_cols_weather + self.continuous_cols_weather:
            assert (
                (df["Column 1"] == col) & (df["Column 2"] == col) & (df["Value"] == 1.0)
            ).any()
        assert (
            (df["Column 1"] == " _dewptm")
            & (df["Column 2"] == " _conds")
            & (df["Value"] == 0.4991)
        ).any()
        assert (
            (df["Column 1"] == " _thunder")
            & (df["Column 2"] == " _wspdm")
            & (df["Value"] == 0.0426)
        ).any()
        assert not ((df["Column 1"] == " _wdire") & (df["Column 2"] == " _hail")).any()

    def test_correlation_value_range(self):
        for df in [
            cat_vs_cat(self.df_weather[self.categorical_cols_weather]),
            cat_vs_cat(self.df[self.categorical_cols]),
            cat_vs_cont(
                self.df_weather,
                self.categorical_cols_weather,
                self.continuous_cols_weather,
            ),
            cat_vs_cont(self.df, self.categorical_cols, self.continuous_cols),
            cont_vs_cont(self.df_weather[self.continuous_cols_weather]),
            cont_vs_cont(self.df[self.continuous_cols]),
        ]:
            assert not (df["Value"] > 1).any() and not (df["Value"] < -1).any()

    cols_tuples = [("a", "b"), ("a", "c")]
    corr_list = [1, 2]

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
        assert isinstance(pearson_plt, mpl.axes._subplots.Axes)

        cramersv_plt = self.df.ads.cramersv_plot()
        assert isinstance(cramersv_plt, mpl.axes._subplots.Axes)

        corr_plt = self.df.ads.correlation_ratio_plot()
        assert isinstance(corr_plt, mpl.axes._subplots.Axes)

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
