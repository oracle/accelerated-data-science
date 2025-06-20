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
            # Rule 1: Strong trend, weak seasonality → ARIMA
            "arima_0": {
                "conditions": [
                    ("ts_trend", "abs>=", 0.65),  # Strong trend strength
                    ("ts_seasonal_strength", "<", 0.20),  # Weak seasonality
                ],
                "model": "arima",
                "priority": 1,
            },
            # Rule 2: Strong seasonality, long series → Prophet
            "prophet_0": {
                "conditions": [
                    ("ts_seasonal_strength", ">=", 0.50),  # Strong seasonality
                    ("ts_n_obs", ">=", 200),  # Long series
                ],
                "model": "prophet",
                "priority": 2,
            },
            # Rule 3: High entropy, low autocorrelation → AutoMLX
            "automlx_0": {
                "conditions": [
                    ("ts_entropy", ">=", 4.0),  # High entropy
                    ("ts_acf1", "<=", 0.30),  # Low autocorrelation
                ],
                "model": "automlx",
                "priority": 3,
            },
            # Rule 4: Strong seasonality with trend and changing patterns → Prophet
            "prophet_1": {
                "conditions": [
                    ("ts_seasonal_strength", ">=", 0.3),  # Strong seasonality
                    ("ts_trend", "abs>=", 0.1),  # Clear trend
                    ("ts_turning_points_rate", ">=", 0.2),  # Multiple change points
                    ("ts_n_obs", ">=", 50),  # Sufficient data
                    ("ts_step_max", ">=", 100),  # Significant steps
                    ("ts_diff1_variance", ">=", 10),  # Variable differences
                ],
                "model": "prophet",
                "priority": 4,
            },
            # Rule 5: Multiple seasonality with nonlinear patterns → Prophet
            "prophet_2": {
                "conditions": [
                    ("ts_seasonal_peak_strength", ">=", 0.4),  # Strong peak seasonality
                    ("ts_seasonal_strength", ">=", 0.2),  # Overall seasonality
                    ("ts_acf10", ">=", 0.2),  # Long-term correlation
                    ("ts_entropy", ">=", 0.5),  # Complex patterns
                    ("ts_crossing_rate", ">=", 0.3),  # Frequent mean crossings
                ],
                "model": "prophet",
                "priority": 5,
            },
            # Rule 6: Strong autocorrelation with stationary behavior → ARIMA
            "arima_1": {
                "conditions": [
                    ("ts_acf1", ">=", 0.7),  # Strong lag-1 correlation
                    ("ts_acf2", ">=", 0.5),  # Strong lag-2 correlation
                    ("ts_seasonal_strength", "<", 0.3),  # Weak seasonality
                    ("ts_std_residuals", "<", 500),  # Stable residuals
                    ("ts_diff1_variance", "<", 100),  # Stable first differences
                    ("ts_hurst", ">", -0.1),  # Some persistence
                ],
                "model": "arima",
                "priority": 6,
            },
            # Rule 7: Linear trend with moderate noise → ARIMA
            "arima_2": {
                "conditions": [
                    ("ts_trend", "abs>=", 0.15),  # Clear trend
                    ("ts_trend_change", "<", 100),  # Stable trend
                    ("ts_cv", "<", 0.4),  # Low variation
                    ("ts_kurtosis", "<", 5),  # Normal-like distribution
                    ("ts_nonlinearity", "<", 1e5),  # Linear relationships
                ],
                "model": "arima",
                "priority": 7,
            },
            # Rule 8: Complex seasonality with high nonlinearity → NeuralProphet
            "neuralprophet_1": {
                "conditions": [
                    ("ts_seasonal_peak_strength", ">=", 0.5),  # Strong seasonal peaks
                    ("ts_nonlinearity", ">=", 1e6),  # Nonlinear patterns
                    ("ts_n_obs", ">=", 200),  # Long series
                    ("ts_entropy", ">=", 0.6),  # Complex patterns
                    ("ts_diff2_variance", ">=", 50),  # Variable acceleration
                ],
                "model": "neuralprophet",
                "priority": 8,
            },
            # Rule 9: Multiple seasonal patterns with changing behavior → NeuralProphet
            "neuralprophet_2": {
                "conditions": [
                    ("ts_seasonal_strength", ">=", 0.4),  # Strong seasonality
                    ("ts_turning_points_rate", ">=", 0.3),  # Many turning points
                    ("ts_skewness", "abs>=", 1),  # Skewed distribution
                    ("ts_diff1_mean", ">=", 10),  # Large changes
                    ("ts_crossing_rate", ">=", 0.4),  # Frequent crossings
                ],
                "model": "neuralprophet",
                "priority": 9,
            },
            # Rule 10: High volatility with complex patterns → AutoMLX
            "automlx_1": {
                "conditions": [
                    ("ts_cv", ">=", 0.6),  # High variation
                    ("ts_nonlinearity", ">=", 1e7),  # Strong nonlinearity
                    ("ts_spikes_rate", ">=", 0.1),  # Frequent spikes
                    ("ts_entropy", ">=", 0.7),  # Very complex
                    ("ts_std_residuals", ">=", 1000),  # Large residuals
                ],
                "model": "automlx",
                "priority": 10,
            },
            # Rule 11: Unstable patterns with regime changes → AutoMLX
            "automlx_2": {
                "conditions": [
                    ("ts_trend_change", ">=", 200),  # Changing trend
                    ("ts_turning_points_rate", ">=", 0.4),  # Many turning points
                    ("ts_diff2_variance", ">=", 100),  # Variable acceleration
                    ("ts_hurst", "<", -0.2),  # Anti-persistent
                    ("ts_step_max", ">=", 1000),  # Large steps
                ],
                "model": "automlx",
                "priority": 11,
            },
            # Rule 12: Long series with stable seasonality → AutoTS
            "autots_1": {
                "conditions": [
                    ("ts_n_obs", ">=", 150),  # Long series
                    ("ts_seasonal_strength", ">=", 0.2),  # Moderate seasonality
                    ("ts_cv", "<", 0.5),  # Moderate variation
                    ("ts_entropy", "<", 0.5),  # Not too complex
                    ("ts_acf1", ">=", 0.3),  # Some autocorrelation
                ],
                "model": "autots",
                "priority": 12,
            },
            # Rule 13: Stable patterns with low noise → Prophet
            "prophet_3": {
                "conditions": [
                    ("ts_cv", "<", 0.3),  # Low variation
                    ("ts_kurtosis", "<", 4),  # Normal-like
                    ("ts_turning_points_rate", "<", 0.25),  # Few turning points
                    ("ts_diff1_variance", "<", 50),  # Stable changes
                    ("ts_seasonal_strength", ">=", 0.1),  # Some seasonality
                ],
                "model": "prophet",
                "priority": 13,
            },
            # Rule 14: Short series with strong linear patterns → ARIMA
            "arima_3": {
                "conditions": [
                    ("ts_n_obs", "<", 100),  # Short series
                    ("ts_trend", "abs>=", 0.2),  # Strong trend
                    ("ts_entropy", "<", 0.4),  # Simple patterns
                    ("ts_nonlinearity", "<", 1e5),  # Linear
                    ("ts_seasonal_strength", "<", 0.2),  # Weak seasonality
                ],
                "model": "arima",
                "priority": 14,
            },
            # Rule 15: Complex seasonal patterns with long memory → NeuralProphet
            "neuralprophet_3": {
                "conditions": [
                    ("ts_n_obs", ">=", 300),  # Very long series
                    ("ts_seasonal_strength", ">=", 0.3),  # Clear seasonality
                    ("ts_acf10", ">=", 0.3),  # Long memory
                    ("ts_hurst", ">", 0),  # Persistent
                    ("ts_nonlinearity", ">=", 5e5),  # Some nonlinearity
                ],
                "model": "neuralprophet",
                "priority": 15,
            },
            # Rule 16: High complexity with non-normal distribution → AutoMLX
            "automlx_3": {
                "conditions": [
                    ("ts_kurtosis", ">=", 5),  # Heavy tails
                    ("ts_skewness", "abs>=", 2),  # Highly skewed
                    ("ts_entropy", ">=", 0.6),  # Complex
                    ("ts_spikes_rate", ">=", 0.05),  # Some spikes
                    ("ts_diff2_mean", ">=", 5),  # Changing acceleration
                ],
                "model": "automlx",
                "priority": 16,
            },
            # Rule 17: Simple patterns with weak seasonality → AutoTS
            "autots_2": {
                "conditions": [
                    ("ts_entropy", "<", 0.3),  # Simple patterns
                    ("ts_seasonal_strength", "<", 0.3),  # Weak seasonality
                    ("ts_cv", "<", 0.4),  # Low variation
                    ("ts_nonlinearity", "<", 1e5),  # Nearly linear
                    ("ts_diff1_mean", "<", 10),  # Small changes
                ],
                "model": "autots",
                "priority": 17,
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
            DataFrame containing meta-features for each series, as returned by
            build_fforms_meta_features

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
