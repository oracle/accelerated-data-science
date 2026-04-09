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
            "A k-nearest neighbors regressor predicts each target value using the "
            "nearest training examples in feature space, which can work well when "
            "similar rows are expected to have similar outcomes."
        )

    def _build_estimator(self):
        params = {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "p": 2,
        }
        params.update(self.spec.model.params or {})
        params.update(self.spec.model_kwargs or {})
        return KNeighborsRegressor(**params)
