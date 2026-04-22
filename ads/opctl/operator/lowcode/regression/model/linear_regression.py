#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from .shared_model import SharedRegressionOperatorModel


class LinearRegressionOperatorModel(SharedRegressionOperatorModel):
    @classmethod
    def get_model_display_name(cls):
        return "Linear Regression"

    @classmethod
    def get_model_description(cls):
        return (
            "Linear regression models a continuous target as a weighted combination of "
            "the input features. It is fast, interpretable, and often performs well when "
            "the underlying relationships are close to linear. In this operator, it can "
            "also be regularized with ridge, lasso, or elastic net penalties to improve "
            "stability and reduce overfitting when features are noisy or correlated."
        )

    def _build_estimator(self):
        params = {
            "fit_intercept": True,
            "copy_X": True,
            "n_jobs": None,
        }
        params.update(self.spec.model_kwargs or {})

        penalty = (params.pop("penalty", "none") or "none").lower()
        alpha = params.pop("alpha", 1.0)
        l1_ratio = params.pop("l1_ratio", 0.5)

        if penalty in {"ridge", "l2"}:
            return Ridge(alpha=alpha, **params)
        if penalty in {"lasso", "l1"}:
            return Lasso(alpha=alpha, **params)
        if penalty in {"elasticnet", "elastic"}:
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **params)

        return LinearRegression(**params)
