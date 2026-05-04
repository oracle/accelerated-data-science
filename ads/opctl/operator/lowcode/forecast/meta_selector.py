#!/usr/bin/env python

# Copyright (c) 2023, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import pandas as pd


class MetaSelector:
    """
    A class to select the best forecasting model for each series based on pre-learned meta-rules.
    The rules are based on the meta-features calculated by the FFORMS approach.
    """

    def __init__(self):
        """Initialize the MetaSelector with pre-learned meta rules"""
        # Pre-learned rules based on meta-features
        self._meta_rules = {
            "ets_0": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", "<=", 0.5027925670146942),
                    ("curvature", "<=", 24.33253288269043),
                    ("diff1y_acf1", "<=", -0.22750446200370789),
                    ("stability", "<=", 179344.421875),
                    ("stability", "<=", 19081.6865234375),
                ],
                "model": "ets",
                "priority": 1,
            },
            "xgbforecast_1": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", "<=", 0.5027925670146942),
                    ("curvature", "<=", 24.33253288269043),
                    ("diff1y_acf1", "<=", -0.22750446200370789),
                    ("stability", "<=", 179344.421875),
                    ("stability", ">", 19081.6865234375),
                ],
                "model": "xgbforecast",
                "priority": 2,
            },
            "arima_2": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", "<=", 0.5027925670146942),
                    ("curvature", "<=", 24.33253288269043),
                    ("diff1y_acf1", "<=", -0.22750446200370789),
                    ("stability", ">", 179344.421875),
                    ("y_acf5", "<=", 1.8753584623336792),
                ],
                "model": "arima",
                "priority": 3,
            },
            "xgbforecast_3": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", "<=", 0.5027925670146942),
                    ("curvature", "<=", 24.33253288269043),
                    ("diff1y_acf1", "<=", -0.22750446200370789),
                    ("stability", ">", 179344.421875),
                    ("y_acf5", ">", 1.8753584623336792),
                ],
                "model": "xgbforecast",
                "priority": 4,
            },
            "xgbforecast_4": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", "<=", 0.5027925670146942),
                    ("curvature", "<=", 24.33253288269043),
                    ("diff1y_acf1", ">", -0.22750446200370789),
                    ("diff2y_pacf5", "<=", 0.7345715165138245),
                    ("diff1y_acf1", "<=", -0.059953220188617706),
                ],
                "model": "xgbforecast",
                "priority": 5,
            },
            "arima_5": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", "<=", 0.5027925670146942),
                    ("curvature", "<=", 24.33253288269043),
                    ("diff1y_acf1", ">", -0.22750446200370789),
                    ("diff2y_pacf5", "<=", 0.7345715165138245),
                    ("diff1y_acf1", ">", -0.059953220188617706),
                ],
                "model": "arima",
                "priority": 6,
            },
            "arima_6": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", "<=", 0.5027925670146942),
                    ("curvature", "<=", 24.33253288269043),
                    ("diff1y_acf1", ">", -0.22750446200370789),
                    ("diff2y_pacf5", ">", 0.7345715165138245),
                    ("diff2y_acf5", "<=", 0.5539001524448395),
                ],
                "model": "arima",
                "priority": 7,
            },
            "arima_7": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", "<=", 0.5027925670146942),
                    ("curvature", "<=", 24.33253288269043),
                    ("diff1y_acf1", ">", -0.22750446200370789),
                    ("diff2y_pacf5", ">", 0.7345715165138245),
                    ("diff2y_acf5", ">", 0.5539001524448395),
                ],
                "model": "arima",
                "priority": 8,
            },
            "xgbforecast_8": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", "<=", 0.5027925670146942),
                    ("curvature", ">", 24.33253288269043),
                ],
                "model": "xgbforecast",
                "priority": 9,
            },
            "arima_9": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", ">", 0.5027925670146942),
                    ("entropy", "<=", 0.5807604193687439),
                    ("diff2y_pacf5", "<=", 0.6616644561290741),
                    ("ur_pp", "<=", -1.4911885261535645),
                ],
                "model": "arima",
                "priority": 10,
            },
            "arima_10": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", ">", 0.5027925670146942),
                    ("entropy", "<=", 0.5807604193687439),
                    ("diff2y_pacf5", "<=", 0.6616644561290741),
                    ("ur_pp", ">", -1.4911885261535645),
                    ("diff1y_pacf5", "<=", 0.7248013317584991),
                ],
                "model": "arima",
                "priority": 11,
            },
            "arima_11": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", ">", 0.5027925670146942),
                    ("entropy", "<=", 0.5807604193687439),
                    ("diff2y_pacf5", "<=", 0.6616644561290741),
                    ("ur_pp", ">", -1.4911885261535645),
                    ("diff1y_pacf5", ">", 0.7248013317584991),
                ],
                "model": "arima",
                "priority": 12,
            },
            "arima_12": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", ">", 0.5027925670146942),
                    ("entropy", "<=", 0.5807604193687439),
                    ("diff2y_pacf5", ">", 0.6616644561290741),
                ],
                "model": "arima",
                "priority": 13,
            },
            "arima_13": {
                "conditions": [
                    ("horizon", "<=", 9.0),
                    ("diff1y_acf1", ">", 0.5027925670146942),
                    ("entropy", ">", 0.5807604193687439),
                ],
                "model": "arima",
                "priority": 14,
            },
            "ets_14": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", "<=", 1650.5),
                    ("e_acf1", "<=", 0.8155759274959564),
                    ("length", "<=", 108.5),
                    ("seasonality_7", "<=", 0.9141938090324402),
                    ("seas_pacf", "<=", 0.21581197530031204),
                ],
                "model": "ets",
                "priority": 15,
            },
            "theta_15": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", "<=", 1650.5),
                    ("e_acf1", "<=", 0.8155759274959564),
                    ("length", "<=", 108.5),
                    ("seasonality_7", "<=", 0.9141938090324402),
                    ("seas_pacf", ">", 0.21581197530031204),
                ],
                "model": "theta",
                "priority": 16,
            },
            "lgbforecast_16": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", "<=", 1650.5),
                    ("e_acf1", "<=", 0.8155759274959564),
                    ("length", "<=", 108.5),
                    ("seasonality_7", ">", 0.9141938090324402),
                ],
                "model": "lgbforecast",
                "priority": 17,
            },
            "theta_17": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", "<=", 1650.5),
                    ("e_acf1", "<=", 0.8155759274959564),
                    ("length", ">", 108.5),
                    ("diff1y_pacf5", "<=", 0.3320583403110504),
                    ("seas_pacf", "<=", 0.5277020633220673),
                ],
                "model": "theta",
                "priority": 18,
            },
            "prophet_18": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", "<=", 1650.5),
                    ("e_acf1", "<=", 0.8155759274959564),
                    ("length", ">", 108.5),
                    ("diff1y_pacf5", "<=", 0.3320583403110504),
                    ("seas_pacf", ">", 0.5277020633220673),
                ],
                "model": "prophet",
                "priority": 19,
            },
            "ets_19": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", "<=", 1650.5),
                    ("e_acf1", "<=", 0.8155759274959564),
                    ("length", ">", 108.5),
                    ("diff1y_pacf5", ">", 0.3320583403110504),
                    ("length", "<=", 894.0),
                ],
                "model": "ets",
                "priority": 20,
            },
            "ets_20": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", "<=", 1650.5),
                    ("e_acf1", "<=", 0.8155759274959564),
                    ("length", ">", 108.5),
                    ("diff1y_pacf5", ">", 0.3320583403110504),
                    ("length", ">", 894.0),
                ],
                "model": "ets",
                "priority": 21,
            },
            "prophet_21": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", "<=", 1650.5),
                    ("e_acf1", ">", 0.8155759274959564),
                    ("skew", "<=", 1.1987826228141785),
                    ("exog_last_abs_mean", "<=", 1188.0742797851562),
                    ("max", "<=", 3457.5),
                ],
                "model": "prophet",
                "priority": 22,
            },
            "theta_22": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", "<=", 1650.5),
                    ("e_acf1", ">", 0.8155759274959564),
                    ("skew", "<=", 1.1987826228141785),
                    ("exog_last_abs_mean", "<=", 1188.0742797851562),
                    ("max", ">", 3457.5),
                ],
                "model": "theta",
                "priority": 23,
            },
            "prophet_23": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", "<=", 1650.5),
                    ("e_acf1", ">", 0.8155759274959564),
                    ("skew", "<=", 1.1987826228141785),
                    ("exog_last_abs_mean", ">", 1188.0742797851562),
                    ("seasonality_m", "<=", 0.9896882474422455),
                ],
                "model": "prophet",
                "priority": 24,
            },
            "lgbforecast_24": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", "<=", 1650.5),
                    ("e_acf1", ">", 0.8155759274959564),
                    ("skew", "<=", 1.1987826228141785),
                    ("exog_last_abs_mean", ">", 1188.0742797851562),
                    ("seasonality_m", ">", 0.9896882474422455),
                ],
                "model": "lgbforecast",
                "priority": 25,
            },
            "xgbforecast_25": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", "<=", 1650.5),
                    ("e_acf1", ">", 0.8155759274959564),
                    ("skew", ">", 1.1987826228141785),
                ],
                "model": "xgbforecast",
                "priority": 26,
            },
            "ets_26": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", ">", 1650.5),
                    ("e_acf1", "<=", 0.7245055437088013),
                    ("diff1y_acf1", "<=", -0.4505922943353653),
                ],
                "model": "ets",
                "priority": 27,
            },
            "ets_27": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", ">", 1650.5),
                    ("e_acf1", "<=", 0.7245055437088013),
                    ("diff1y_acf1", ">", -0.4505922943353653),
                    ("entropy", "<=", 0.44683755934238434),
                    ("diff2y_acf5", "<=", 0.4195839762687683),
                ],
                "model": "ets",
                "priority": 28,
            },
            "ets_28": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", ">", 1650.5),
                    ("e_acf1", "<=", 0.7245055437088013),
                    ("diff1y_acf1", ">", -0.4505922943353653),
                    ("entropy", "<=", 0.44683755934238434),
                    ("diff2y_acf5", ">", 0.4195839762687683),
                ],
                "model": "ets",
                "priority": 29,
            },
            "ets_29": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", ">", 1650.5),
                    ("e_acf1", "<=", 0.7245055437088013),
                    ("diff1y_acf1", ">", -0.4505922943353653),
                    ("entropy", ">", 0.44683755934238434),
                ],
                "model": "ets",
                "priority": 30,
            },
            "ets_30": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", ">", 1650.5),
                    ("e_acf1", ">", 0.7245055437088013),
                    ("y_acf5", "<=", 3.8231289386749268),
                    ("cv", "<=", 0.35646331310272217),
                    ("adf_pvalue", "<=", 0.4134090393781662),
                ],
                "model": "ets",
                "priority": 31,
            },
            "theta_31": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", ">", 1650.5),
                    ("e_acf1", ">", 0.7245055437088013),
                    ("y_acf5", "<=", 3.8231289386749268),
                    ("cv", "<=", 0.35646331310272217),
                    ("adf_pvalue", ">", 0.4134090393781662),
                ],
                "model": "theta",
                "priority": 32,
            },
            "ets_32": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", ">", 1650.5),
                    ("e_acf1", ">", 0.7245055437088013),
                    ("y_acf5", "<=", 3.8231289386749268),
                    ("cv", ">", 0.35646331310272217),
                    ("diff1y_acf1", "<=", -0.4530174732208252),
                ],
                "model": "ets",
                "priority": 33,
            },
            "ets_33": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", ">", 1650.5),
                    ("e_acf1", ">", 0.7245055437088013),
                    ("y_acf5", "<=", 3.8231289386749268),
                    ("cv", ">", 0.35646331310272217),
                    ("diff1y_acf1", ">", -0.4530174732208252),
                ],
                "model": "ets",
                "priority": 34,
            },
            "prophet_34": {
                "conditions": [
                    ("horizon", ">", 9.0),
                    ("length", ">", 1650.5),
                    ("e_acf1", ">", 0.7245055437088013),
                    ("y_acf5", ">", 3.8231289386749268),
                ],
                "model": "prophet",
                "priority": 35,
            },
            "ets_default": {
                "conditions": [],
                "model": "ets",
                "priority": 36,
            },
        }

    def _evaluate_condition(self, value, operator, threshold):
        """Evaluate a single condition based on pre-defined operators"""
        if pd.isna(value):
            return False

        if operator == ">=":
            return value >= threshold
        elif operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "abs>=":
            return abs(value) >= threshold
        elif operator == "abs<":
            return abs(value) < threshold
        return False

    def _check_model_conditions(self, features, model_rules):
        """Check if a series meets all conditions for a model"""
        for feature, operator, threshold in model_rules["conditions"]:
            if feature not in features:
                return False
            if not self._evaluate_condition(features[feature], operator, threshold):
                return False
        return True

    def select_best_model(self, meta_features_df):
        """
        Select the best model for each series based on pre-learned rules.

        Parameters
        ----------
        meta_features_df : pandas.DataFrame
            DataFrame containing meta-features for each series

        Returns
        -------
        pandas.DataFrame
            DataFrame with series identifiers, selected model names, and matching rule info
        """
        results = []

        # Process each series
        for _, row in meta_features_df.iterrows():
            series_info = {}

            # Preserve group columns if they exist
            group_cols = [col for col in row.index if not col.startswith("ts_")]
            for col in group_cols:
                series_info[col] = row[col]

            # Find matching models
            matching_models = []
            matched_features = {}
            for rule_name, rules in self._meta_rules.items():
                if self._check_model_conditions(row, rules):
                    matching_models.append((rule_name, rules["priority"]))
                    # Store which features triggered this rule
                    matched_features[rule_name] = [
                        (feature, row[feature]) for feature, _, _ in rules["conditions"]
                    ]

            # Select best model based on priority
            if matching_models:
                best_rule = min(matching_models, key=lambda x: x[1])[0]
                best_model = self._meta_rules[best_rule]["model"]
                series_info["matched_features"] = matched_features[best_rule]
            else:
                best_rule = "default"
                best_model = "prophet"  # Default to prophet if no rules match
                series_info["matched_features"] = []

            series_info["selected_model"] = best_model
            series_info["rule_matched"] = best_rule
            results.append(series_info)

        return pd.DataFrame(results)

    def get_model_conditions(self):
        """
        Get the pre-learned conditions for each model.
        This is read-only and cannot be modified at runtime.

        Returns
        -------
        dict
            Dictionary containing the conditions for each model
        """
        return self._meta_rules.copy()
