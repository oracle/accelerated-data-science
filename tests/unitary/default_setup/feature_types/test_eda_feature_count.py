#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import pandas as pd
from unittest import TestCase

from ads.feature_engineering.feature_type_manager import FeatureTypeManager


class TestEDAFeatureCount(TestCase):
    def setUp(self):
        """Sets up the test case."""
        super().setUp()
        FeatureTypeManager.feature_type_reset()

    data = {
        "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "col2": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        "col3": [1, 3, 1, 1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
        "col4": [
            "ab",
            "te",
            "ui",
            "kl",
            "tr",
            "uy",
            "iu",
            "mi",
            "uy",
            "pa",
            "pp",
            "we",
            "ii",
            "bt",
            "gg",
            "tl",
            "mn",
            "lo",
            "qa",
            "by",
        ],
        "col5": [
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
        "col6": np.random.default_rng().uniform(low=1, high=98, size=20),
        "col7": np.random.randint(10, size=20),
        "col8": 100 * np.random.rand(20) + 10,
        "col9": [
            "A10",
            "K19",
            "M89",
            "L55",
            "O99",
            "S34",
            "G77",
            "S34",
            "M65",
            "S34",
            "P87",
            "V47",
            "L82",
            "S34",
            "S34",
            "K19",
            "A10",
            "L38",
            "Q88",
            "A10",
        ],
    }
    df = pd.DataFrame(data=data)

    def test_feature_count_0(self):
        feature_count_df = self.df.ads.feature_count()
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "continuous"][
                "Count"
            ].iloc[0]
            == 2
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "continuous"][
                "Primary"
            ].iloc[0]
            == 2
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "string"][
                "Count"
            ].iloc[0]
            == 3
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "string"][
                "Primary"
            ].iloc[0]
            == 3
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "integer"][
                "Count"
            ].iloc[0]
            == 4
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "integer"][
                "Primary"
            ].iloc[0]
            == 4
        )

    def test_feature_count_1(self):
        self.df["col6"].ads.feature_type = ["continuous", "integer"]
        feature_count_df = self.df.ads.feature_count()
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "continuous"][
                "Count"
            ].iloc[0]
            == 2
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "continuous"][
                "Primary"
            ].iloc[0]
            == 2
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "string"][
                "Count"
            ].iloc[0]
            == 3
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "string"][
                "Primary"
            ].iloc[0]
            == 3
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "integer"][
                "Count"
            ].iloc[0]
            == 5
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "integer"][
                "Primary"
            ].iloc[0]
            == 4
        )

    def test_feature_count_2(self):
        self.df["col2"].ads.feature_type = ["category", "integer"]
        feature_count_df = self.df.ads.feature_count()
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "continuous"][
                "Count"
            ].iloc[0]
            == 2
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "continuous"][
                "Primary"
            ].iloc[0]
            == 2
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "category"][
                "Count"
            ].iloc[0]
            == 1
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "string"][
                "Count"
            ].iloc[0]
            == 3
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "category"][
                "Primary"
            ].iloc[0]
            == 1
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "integer"][
                "Count"
            ].iloc[0]
            == 5
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "integer"][
                "Primary"
            ].iloc[0]
            == 3
        )
