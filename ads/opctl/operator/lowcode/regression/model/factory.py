#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy

from ads.opctl import logger
from ads.opctl.operator.lowcode.regression.const import (
    SupportedMetrics,
    SupportedModels,
    TROUBLESHOOTING_GUIDE,
)
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
            model_type = cls._auto_select_model(operator_config, datasets)
            operator_config.spec.model = model_type
            operator_config.spec.model_kwargs = {}
            logger.info(f"Auto-selected model: {model_type}")

        if model_type not in cls._MAP:
            raise UnSupportedModelError(model_type)

        return cls._MAP[model_type](config=operator_config, datasets=datasets)

    @classmethod
    def _auto_select_model(
        cls,
        operator_config: RegressionOperatorConfig,
        datasets: RegressionDatasets,
    ):
        from sklearn.model_selection import train_test_split

        x = datasets.training_data[datasets.feature_columns]
        y = datasets.training_data[operator_config.spec.target_column]
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        maximize = operator_config.spec.metric == SupportedMetrics.R2
        best_model = None
        best_score = None

        for model_name, model_cls in cls._MAP.items():
            try:
                test_cfg = copy.deepcopy(operator_config)
                test_cfg.spec.model = model_name
                candidate = model_cls(config=test_cfg, datasets=datasets)
                preprocessor = candidate._build_preprocessor(x_train)
                estimator = candidate._build_estimator()
                x_train_processed = preprocessor.preprocess_for_training(x_train)
                estimator.fit(x_train_processed, y_train)
                candidate.preprocessor = preprocessor
                candidate.regressor = estimator
                inference_model = candidate._create_inference_model()
                yhat = inference_model.predict(x_valid)
                metrics = candidate._compute_metrics(y_valid, yhat)
                score = metrics.get(operator_config.spec.metric)

                if best_score is None:
                    best_model = model_name
                    best_score = score
                else:
                    is_better = score > best_score if maximize else score < best_score
                    if is_better:
                        best_model = model_name
                        best_score = score
            except Exception as e:
                logger.warning(f"Skipping model `{model_name}` during auto-select. Error: {e}")

        return best_model or SupportedModels.RANDOM_FOREST
