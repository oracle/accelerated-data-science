#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from sklearn.neighbors import KNeighborsRegressor

from .shared_model import SharedRegressionOperatorModel


class KNNRegressionOperatorModel(SharedRegressionOperatorModel):
    @classmethod
    def get_model_display_name(cls):
        return "KNN"

    @classmethod
    def get_model_description(cls):
        return (
            "A k-nearest neighbors regressor predicts each target value from the outcomes "
            "of the most similar training rows in feature space. It is a simple, instance-"
            "based method that can work well when nearby observations are expected to have "
            "similar targets. Its performance depends heavily on feature scaling, distance "
            "behavior, and the quality of the local neighborhoods in the dataset."
        )

    def _build_estimator(self):
        params = self._build_default_params()
        params.update(self.spec.model_kwargs or {})
        return KNeighborsRegressor(**params)

    def _build_default_params(self):
        row_count = max(len(self.datasets.training_data.index), 1)
        feature_count = max(len(self.feature_columns), 1)

        return {
            "n_neighbors": self._default_n_neighbors(row_count),
            "weights": "uniform",
            "algorithm": "auto",
            "p": self._default_p(feature_count),
            "n_jobs": -1,
        }

    @staticmethod
    def _default_n_neighbors(row_count: int) -> int:
        if row_count < 50:
            return min(3, row_count)
        if row_count < 200:
            return min(5, row_count)
        if row_count < 1000:
            return min(9, row_count)
        if row_count < 5000:
            return min(15, row_count)
        return min(25, row_count)

    @staticmethod
    def _default_p(feature_count: int) -> int:
        if feature_count <= 20:
            return 2
        return 1
