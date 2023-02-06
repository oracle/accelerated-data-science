#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.feature_engineering.feature_type.handler.warnings import (
    missing_values_handler,
    skew_handler,
    high_cardinality_handler,
    zeros_handler,
)
from tests.ads_unit_tests.utils import get_test_dataset_path
import pandas as pd


class TestFeatureTypesDefaultWarnings:
    empty_warning_df = pd.DataFrame(
        [], columns=["Warning", "Message", "Metric", "Value"]
    )
    df_titanic = pd.read_csv(get_test_dataset_path("vor_titanic.csv"))
    df_flights = pd.read_csv(get_test_dataset_path("vor_flights.csv"))

    def test_missing_values_titanic_cabin(self):
        warning_df = missing_values_handler(self.df_titanic["Cabin"])
        assert warning_df.loc[0]["Value"] == 687
        assert warning_df.loc[0]["Metric"] == "count"
        assert abs(warning_df.loc[1]["Value"] - 77.1) < 0.1
        assert warning_df.loc[1]["Metric"] == "percentage"

    def test_missing_values_flights_cancellation_code(self):
        warning_df = missing_values_handler(self.df_flights["CancellationCode"])
        assert warning_df.loc[0]["Value"] == 98858
        assert warning_df.loc[0]["Metric"] == "count"
        assert abs(warning_df.loc[1]["Value"] - 98.9) < 0.1
        assert warning_df.loc[1]["Metric"] == "percentage"

    def test_skew_empty(self):
        warning_df = skew_handler(self.df_titanic["Fare"])
        assert len(warning_df) == 1
        assert list(warning_df.columns) == ["Warning", "Message", "Metric", "Value"]
        assert abs(warning_df.loc[0]["Value"] - 4.787) < 0.1

    def test_skew_flights_diverted(self):
        warning_df = skew_handler(self.df_flights["Diverted"])
        assert abs(warning_df.loc[0]["Value"] - 24.940) < 0.1
        assert warning_df.loc[0]["Metric"] == "skew"

    def test_high_cardinality_titanic_ticket(self):
        warning_df = high_cardinality_handler(self.df_titanic["Ticket"])
        assert warning_df.loc[0]["Value"] == 681
        assert warning_df.loc[0]["Metric"] == "count"

    def test_high_cardinality_flights_tailnum(self):
        warning_df = high_cardinality_handler(self.df_flights["TailNum"])
        assert warning_df.loc[0]["Value"] == 624
        assert warning_df.loc[0]["Metric"] == "count"

    def test_zeros_titanic_survived(self):
        warning_df = zeros_handler(self.df_titanic["Survived"])
        assert warning_df.loc[0]["Value"] == 549
        assert warning_df.loc[0]["Metric"] == "count"
        assert abs(warning_df.loc[1]["Value"] - 61.62) < 0.1
        assert warning_df.loc[1]["Metric"] == "percentage"

    def test_zeros_titanic_sibsp(self):
        warning_df = zeros_handler(self.df_titanic["SibSp"])
        assert warning_df.loc[0]["Value"] == 608
        assert warning_df.loc[0]["Metric"] == "count"
        assert abs(warning_df.loc[1]["Value"] - 68.24) < 0.1
        assert warning_df.loc[1]["Metric"] == "percentage"

    def test_zeros_flights_cancelled(self):
        warning_df = zeros_handler(self.df_flights["Cancelled"])
        assert warning_df.loc[0]["Value"] == 98858
        assert warning_df.loc[0]["Metric"] == "count"
        assert abs(warning_df.loc[1]["Value"] - 98.86) < 0.1
        assert warning_df.loc[1]["Metric"] == "percentage"
