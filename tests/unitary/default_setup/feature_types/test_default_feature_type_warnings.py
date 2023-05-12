#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import random

import numpy as np
import pandas as pd

from ads.feature_engineering.feature_type.handler.warnings import (
    missing_values_handler,
    skew_handler,
    high_cardinality_handler,
    zeros_handler,
)


class TestFeatureTypesDefaultWarnings:
    empty_warning_df = pd.DataFrame(
        [], columns=["Warning", "Message", "Metric", "Value"]
    )

    nums = pd.Series(
        [
            1,
            2,
            3,
            4,
            5,
            np.nan,
            6,
            7,
            np.nan,
            np.nan,
            8,
            9,
            10,
            np.nan,
            3,
            5,
            8,
            2,
            1,
            5,
            1,
            1,
        ]
    )
    letters = pd.Series(
        [
            np.nan,
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            np.nan,
            "l",
            "m",
            "n",
            "o",
        ]
    )
    skewed = pd.Series([0, 0, 0, np.nan, 0, 0, 1, 1, np.nan, 0, 0, 0, 0, 0])

    def test_missing_values_nums_feature(self):
        warning_df = missing_values_handler(self.nums)
        assert warning_df.loc[0]["Value"] == 4
        assert warning_df.loc[0]["Metric"] == "count"
        assert abs(warning_df.loc[1]["Value"] - 18.2) < 0.1
        assert warning_df.loc[1]["Metric"] == "percentage"

    def test_missing_values_letters_feature(self):
        warning_df = missing_values_handler(self.letters)
        assert warning_df.loc[0]["Value"] == 2
        assert warning_df.loc[0]["Metric"] == "count"
        assert abs(warning_df.loc[1]["Value"] - 11.8) < 0.1
        assert warning_df.loc[1]["Metric"] == "percentage"

    def test_skew_empty(self):
        warning_df = skew_handler(self.nums)
        assert len(warning_df) == 0

    def test_skew_skewed_feature(self):
        warning_df = skew_handler(self.skewed)
        assert len(warning_df) == 1
        assert list(warning_df.columns) == ["Warning", "Message", "Metric", "Value"]
        assert abs(warning_df.loc[0]["Value"] - 2.055) < 0.1
        assert warning_df.loc[0]["Metric"] == "skew"

    def test_high_cardinality_letters_feature(self):
        warning_df = high_cardinality_handler(self.letters)
        assert warning_df.loc[0]["Value"] == 16
        assert warning_df.loc[0]["Metric"] == "count"

    def test_high_cardinality_every_value_distinct(self):
        s = self.letters.dropna(inplace=False)
        warning_df = high_cardinality_handler(s)
        assert warning_df.loc[0]["Value"] == 15
        assert warning_df.loc[0]["Metric"] == "count"
        assert "every value is distinct" in warning_df.loc[0]["Message"]

    def test_zeros_skewed_feature(self):
        warning_df = zeros_handler(self.skewed)
        assert warning_df.loc[0]["Value"] == 10
        assert warning_df.loc[0]["Metric"] == "count"
        assert abs(warning_df.loc[1]["Value"] - 71.4) < 0.1
        assert warning_df.loc[1]["Metric"] == "percentage"
