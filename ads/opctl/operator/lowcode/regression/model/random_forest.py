#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from sklearn.ensemble import RandomForestRegressor

from .shared_model import SharedRegressionOperatorModel


class RandomForestRegressionOperatorModel(SharedRegressionOperatorModel):
    @classmethod
    def get_model_display_name(cls):
        return "Random Forest"

    @classmethod
    def get_model_description(cls):
        return (
            "A random forest regressor combines many decision trees trained on sampled "
            "subsets of the data, which helps capture non-linear relationships while "
            "reducing overfitting compared with a single tree."
        )

    def _build_estimator(self):
        params = {
            "n_estimators": 200,
            "max_depth": None,
            "random_state": 42,
            "n_jobs": -1,
        }
        params.update(self.spec.model.params or {})
        params.update(self.spec.model_kwargs or {})
        return RandomForestRegressor(**params)
