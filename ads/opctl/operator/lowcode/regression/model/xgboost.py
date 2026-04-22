#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.decorator.runtime_dependency import OptionalDependency, runtime_dependency
from ads.opctl.operator.lowcode.regression.const import SupportedMetrics

from .shared_model import SharedRegressionOperatorModel


class XGBoostRegressionOperatorModel(SharedRegressionOperatorModel):
    @classmethod
    def get_model_display_name(cls):
        return "XGBoost"

    @classmethod
    def get_model_description(cls):
        return (
            "An XGBoost regressor builds decision trees sequentially, with each new tree "
            "focused on correcting the residual errors left by the previous ensemble. "
            "This gradient boosting approach is often one of the strongest options for "
            "accurate regression on structured tabular data, especially when relationships "
            "are non-linear and feature interactions matter. It usually offers strong "
            "predictive performance, but it also introduces more tuning complexity than "
            "simpler linear models."
        )

    @runtime_dependency(
        module="xgboost",
        install_from=OptionalDependency.OPCTL,
        err_msg="Please run `python3 -m pip install xgboost` to use the `xgboost` regression model.",
    )
    def _build_estimator(self):
        from xgboost import XGBRegressor

        params = self._build_default_params()
        params.update(self.spec.model_kwargs or {})
        return XGBRegressor(**params)

    def _build_default_params(self):
        row_count = max(len(self.datasets.training_data.index), 1)
        feature_count = max(len(self.feature_columns), 1)

        return {
            "n_estimators": self._default_n_estimators(row_count),
            "max_depth": self._default_max_depth(feature_count),
            "learning_rate": self._default_learning_rate(row_count),
            "min_child_weight": self._default_min_child_weight(row_count),
            "subsample": 0.9 if row_count < 1000 else 0.85,
            "colsample_bytree": self._default_colsample_bytree(feature_count),
            "reg_lambda": 1.0 if row_count < 1000 else 1.5,
            "objective": "reg:squarederror",
            "eval_metric": self._default_eval_metric(),
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }

    def _default_eval_metric(self):
        if self.spec.metric in (SupportedMetrics.MAE, SupportedMetrics.MAPE):
            return "mae"
        return "rmse"

    @staticmethod
    def _default_n_estimators(row_count: int) -> int:
        if row_count < 1000:
            return 500
        if row_count < 10000:
            return 400
        if row_count < 50000:
            return 300
        return 200

    @staticmethod
    def _default_learning_rate(row_count: int) -> float:
        if row_count < 10000:
            return 0.05
        return 0.075

    @staticmethod
    def _default_max_depth(feature_count: int) -> int:
        if feature_count <= 10:
            return 6
        if feature_count <= 50:
            return 5
        return 4

    @staticmethod
    def _default_min_child_weight(row_count: int) -> int:
        if row_count < 1000:
            return 1
        if row_count < 10000:
            return 3
        if row_count < 50000:
            return 5
        return 8

    @staticmethod
    def _default_colsample_bytree(feature_count: int) -> float:
        if feature_count <= 20:
            return 0.9
        if feature_count <= 100:
            return 0.8
        return 0.7
