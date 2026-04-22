#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import math

from ads.opctl.operator.lowcode.regression.const import SupportedMetrics
from sklearn.ensemble import RandomForestRegressor

from .shared_model import SharedRegressionOperatorModel


class RandomForestRegressionOperatorModel(SharedRegressionOperatorModel):
    @classmethod
    def get_model_display_name(cls):
        return "Random Forest"

    @classmethod
    def get_model_description(cls):
        return (
            "A random forest regressor combines many decision trees trained on different "
            "samples of the data and subsets of features. By averaging predictions across "
            "those trees, it can capture non-linear relationships and interaction effects "
            "while being more stable and less prone to overfitting than a single decision "
            "tree. It is typically a strong choice for structured tabular data when model "
            "flexibility is more important than simple coefficient-based interpretability."
        )

    def _build_estimator(self):
        params = self._build_default_params()
        params.update(self.spec.model_kwargs or {})
        return RandomForestRegressor(**params)

    def _build_default_params(self):
        row_count = max(len(self.datasets.training_data.index), 1)
        feature_count = max(len(self.feature_columns), 1)

        min_samples_leaf = self._default_min_samples_leaf(row_count)
        params = {
            "n_estimators": self._default_n_estimators(row_count),
            "criterion": self._default_criterion(),
            "max_depth": self._default_max_depth(row_count, feature_count),
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": max(4, min_samples_leaf * 2),
            "max_features": self._default_max_features(feature_count),
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
        }

        if row_count >= 1000:
            params["max_samples"] = 0.85

        if row_count >= max(200, feature_count * 5):
            params["oob_score"] = True

        return params

    def _default_criterion(self):
        if self.spec.metric in (SupportedMetrics.MAE, SupportedMetrics.MAPE):
            return "absolute_error"
        return "squared_error"

    @staticmethod
    def _default_n_estimators(row_count: int) -> int:
        if row_count < 500:
            return 500
        if row_count < 5000:
            return 400
        if row_count < 50000:
            return 300
        return 200

    @staticmethod
    def _default_min_samples_leaf(row_count: int) -> int:
        if row_count < 200:
            return 1
        if row_count < 1000:
            return 2
        if row_count < 5000:
            return 3
        return min(20, max(4, int(math.ceil(row_count * 0.001))))

    @staticmethod
    def _default_max_features(feature_count: int):
        if feature_count <= 4:
            return 1.0
        if feature_count <= 16:
            return "sqrt"
        if feature_count <= 64:
            return 0.5
        return 0.3

    @staticmethod
    def _default_max_depth(row_count: int, feature_count: int):
        if row_count >= 50000 and feature_count >= 100:
            return 20
        if row_count >= 10000 and feature_count >= 50:
            return 24
        return None
