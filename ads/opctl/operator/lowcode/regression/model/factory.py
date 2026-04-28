#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator.lowcode.regression.const import SupportedModels, TROUBLESHOOTING_GUIDE
from ads.opctl.operator.lowcode.regression.model.auto import AutoRegressionOperatorModel
from ads.opctl.operator.lowcode.regression.model.base_model import RegressionOperatorBaseModel
from ads.opctl.operator.lowcode.regression.model.knn import KNNRegressionOperatorModel
from ads.opctl.operator.lowcode.regression.model.linear_regression import (
    LinearRegressionOperatorModel,
)
from ads.opctl.operator.lowcode.regression.model.random_forest import (
    RandomForestRegressionOperatorModel,
)
from ads.opctl.operator.lowcode.regression.model.regression_dataset import RegressionDatasets
from ads.opctl.operator.lowcode.regression.model.xgboost import XGBoostRegressionOperatorModel
from ads.opctl.operator.lowcode.regression.operator_config import RegressionOperatorConfig


class UnSupportedModelError(Exception):
    def __init__(self, model_type: str):
        super().__init__(
            f"Model: `{model_type}` is not supported. Supported models: {SupportedModels.values()}"
            f"\nPlease refer to the troubleshooting guide at {TROUBLESHOOTING_GUIDE} for resolution steps."
        )


class RegressionOperatorModelFactory:
    """Factory to instantiate proper regression model operator by model name."""

    _MAP = {
        SupportedModels.LINEAR_REGRESSION: LinearRegressionOperatorModel,
        SupportedModels.RANDOM_FOREST: RandomForestRegressionOperatorModel,
        SupportedModels.KNN: KNNRegressionOperatorModel,
        SupportedModels.XGBOOST: XGBoostRegressionOperatorModel,
    }

    @classmethod
    def get_model(
        cls,
        operator_config: RegressionOperatorConfig,
        datasets: RegressionDatasets,
    ) -> RegressionOperatorBaseModel:
        model_type = (operator_config.spec.model or "").lower()

        if model_type == SupportedModels.AUTO:
            return AutoRegressionOperatorModel(config=operator_config, datasets=datasets)

        if model_type not in cls._MAP:
            raise UnSupportedModelError(model_type)

        return cls._MAP[model_type](config=operator_config, datasets=datasets)
