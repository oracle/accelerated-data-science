#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
from unittest import TestCase
from tests.ads_unit_tests.utils import get_test_dataset_path
from ads.feature_engineering.feature_type_manager import FeatureTypeManager


class TestEDAFeatureCount(TestCase):
    def setUp(self):
        """Sets up the test case."""
        super().setUp()
        FeatureTypeManager.feature_type_reset()

    titanic = pd.read_csv(get_test_dataset_path("vor_titanic.csv"))

    def test_feature_count_titanic_0(self):
        feature_count_df = self.titanic.ads.feature_count()
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
            == 5
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "string"][
                "Primary"
            ].iloc[0]
            == 5
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
            == 5
        )

    def test_feature_count_titanic_1(self):
        self.titanic["Age"].ads.feature_type = ["continuous", "integer"]
        feature_count_df = self.titanic.ads.feature_count()
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
            == 5
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "string"][
                "Primary"
            ].iloc[0]
            == 5
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "integer"][
                "Count"
            ].iloc[0]
            == 6
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "integer"][
                "Primary"
            ].iloc[0]
            == 5
        )

    def test_feature_count_titanic_2(self):
        self.titanic["Survived"].ads.feature_type = ["category", "integer"]
        feature_count_df = self.titanic.ads.feature_count()
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
            == 5
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
            == 6
        )
        assert (
            feature_count_df[feature_count_df["Feature Type"] == "integer"][
                "Primary"
            ].iloc[0]
            == 4
        )
