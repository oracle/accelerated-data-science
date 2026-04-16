#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.decorator.runtime_dependency import OptionalDependency, runtime_dependency

from .shared_model import SharedRegressionOperatorModel


class XGBoostRegressionOperatorModel(SharedRegressionOperatorModel):
    @classmethod
    def get_model_display_name(cls):
        return "XGBoost"

    @classmethod
    def get_model_description(cls):
        return (
            "An XGBoost regressor builds trees sequentially, with each new tree focused "
            "on correcting the residual errors of the previous ones, making it a strong "
            "choice for accurate tabular regression on structured data."
        )

    @runtime_dependency(
        module="xgboost",
        install_from=OptionalDependency.OPCTL,
        err_msg="Please run `python3 -m pip install xgboost` to use the `xgboost` regression model.",
    )
    def _build_estimator(self):
        from xgboost import XGBRegressor

        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        params.update(self.spec.model_kwargs or {})
        return XGBRegressor(**params)
