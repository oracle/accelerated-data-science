#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ads.opctl import logger
from ads.opctl.operator.lowcode.regression.const import SupportedMetrics, SupportedModels

from .base_model import RegressionOperatorBaseModel
from .knn import KNNRegressionOperatorModel
from .linear_regression import LinearRegressionOperatorModel
from .random_forest import RandomForestRegressionOperatorModel
from .xgboost import XGBoostRegressionOperatorModel


class AutoRegressionOperatorModel(RegressionOperatorBaseModel):
    _CANDIDATES = {
        SupportedModels.LINEAR_REGRESSION: LinearRegressionOperatorModel,
        SupportedModels.RANDOM_FOREST: RandomForestRegressionOperatorModel,
        SupportedModels.KNN: KNNRegressionOperatorModel,
        SupportedModels.XGBOOST: XGBoostRegressionOperatorModel,
    }

    def __init__(self, config, datasets):
        super().__init__(config, datasets)
        self.selected_model_name = None
        self.selected_model_display_name = None
        self.cv_results_df = None

    @classmethod
    def get_model_display_name(cls):
        return "Auto"

    @classmethod
    def get_model_description(cls):
        return (
            "Auto regression evaluates the available regression models using cross-validation "
            "on the training dataset, selects the model that performs best for the configured "
            "metric, and then fits that selected model on the full training data before "
            "generating predictions and artifacts."
        )

    def _report_model_display_name(self):
        return self.get_model_display_name()

    def _report_model_description(self):
        if not self.selected_model_display_name:
            return self.get_model_description()
        return (
            f"Auto model selection evaluated the available regression models using "
            f"cross-validation on the training dataset and found "
            f"{self.selected_model_display_name} as the best model for the configured "
            f"`{self.spec.metric}` metric. "
            f"{self._CANDIDATES[self.selected_model_name].get_model_description()}"
        )

    def _build_estimator(self):
        raise NotImplementedError(
            "Auto regression does not build a single estimator directly."
        )

    def _train_and_predict(self, x_train, y_train):
        selected_model_name, cv_results_df = self._select_best_model(x_train, y_train)
        self.selected_model_name = selected_model_name
        self.cv_results_df = cv_results_df

        selected_model_cls = self._CANDIDATES[selected_model_name]
        self.selected_model_display_name = selected_model_cls.get_model_display_name()

        final_config = copy.deepcopy(self.config)
        final_config.spec.model = selected_model_name
        final_config.spec.model_kwargs = {"tuning_n_trials": 0}

        final_model = selected_model_cls(config=final_config, datasets=self.datasets)
        x_proc = final_model._train_and_predict(x_train, y_train)

        self.preprocessor = final_model.preprocessor
        self.regressor = final_model.regressor
        self.model_obj = final_model.model_obj
        self.feature_names_out = final_model.feature_names_out
        self.train_predictions = final_model.train_predictions
        self.test_predictions = final_model.test_predictions
        self.train_metrics = final_model.train_metrics
        self.test_metrics = final_model.test_metrics
        self.global_explanations_df = final_model.global_explanations_df

        self.spec.model = selected_model_name
        self.config.spec.model = selected_model_name
        self.spec.model_kwargs = {"tuning_n_trials": 0}
        self.config.spec.model_kwargs = {"tuning_n_trials": 0}

        logger.info(
            f"Auto regression selected `{selected_model_name}` using `{self.spec.metric}`."
        )
        return x_proc

    def _select_best_model(
        self, x_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[str, pd.DataFrame]:
        row_count = len(x_train)
        n_splits = min(5, row_count)
        if n_splits < 2:
            return SupportedModels.LINEAR_REGRESSION, pd.DataFrame()

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        maximize = self.spec.metric == SupportedMetrics.R2
        cv_rows = []

        for model_name, model_cls in self._CANDIDATES.items():
            fold_scores = []
            try:
                for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(x_train), start=1):
                    fold_config = copy.deepcopy(self.config)
                    fold_config.spec.model = model_name
                    fold_config.spec.model_kwargs = {}

                    candidate = model_cls(config=fold_config, datasets=self.datasets)

                    x_fold_train = x_train.iloc[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    x_fold_valid = x_train.iloc[valid_idx]
                    y_fold_valid = y_train.iloc[valid_idx]

                    preprocessor = candidate._build_preprocessor(x_fold_train)
                    estimator = candidate._build_estimator()
                    x_fold_train_processed = preprocessor.preprocess_for_training(
                        x_fold_train
                    )
                    estimator.fit(x_fold_train_processed, y_fold_train)

                    candidate.preprocessor = preprocessor
                    candidate.regressor = estimator
                    inference_model = candidate._create_inference_model()
                    predictions = inference_model.predict(x_fold_valid)
                    fold_metric = candidate._compute_metrics(y_fold_valid, predictions)[
                        self.spec.metric
                    ]
                    fold_scores.append(fold_metric)
                    cv_rows.append(
                        {
                            "model": model_name,
                            "fold": fold_index,
                            "metric": self.spec.metric,
                            "score": fold_metric,
                        }
                    )
            except Exception as e:
                logger.warning(
                    f"Skipping model `{model_name}` during auto regression selection. Error: {e}"
                )
                continue

        cv_results_df = pd.DataFrame(cv_rows)
        if cv_results_df.empty:
            return SupportedModels.RANDOM_FOREST, cv_results_df

        summary_df = (
            cv_results_df.groupby("model", as_index=False)["score"]
            .mean()
            .rename(columns={"score": "mean_score"})
        )
        summary_df = summary_df.sort_values(
            "mean_score", ascending=not maximize
        ).reset_index(drop=True)

        best_model_name = summary_df.iloc[0]["model"]
        return best_model_name, cv_results_df
