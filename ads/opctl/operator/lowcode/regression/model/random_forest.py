#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import math
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ads.common.decorator.runtime_dependency import OptionalDependency, runtime_dependency
from ads.opctl import logger
from ads.opctl.operator.lowcode.regression.const import SupportedMetrics
from sklearn.ensemble import RandomForestRegressor

from .shared_model import SharedRegressionOperatorModel


class RandomForestRegressionOperatorModel(SharedRegressionOperatorModel):
    DEFAULT_TUNING_TRIALS = 20

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
        return self._build_estimator_from_params(self.spec.model_kwargs or {})

    def _build_estimator_from_params(self, model_kwargs):
        params = self._build_default_params()
        params.update(model_kwargs or {})
        params.pop("tuning_n_trials", None)
        return RandomForestRegressor(**params)

    @staticmethod
    def _slice_rows(data, indices):
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        return data[indices]

    def _extract_tuning_n_trials(self, model_kwargs):
        value = model_kwargs.pop("tuning_n_trials", self.DEFAULT_TUNING_TRIALS)
        if value is None:
            return self.DEFAULT_TUNING_TRIALS
        try:
            return int(value)
        except (TypeError, ValueError):
            return self.DEFAULT_TUNING_TRIALS

    def _evaluate_with_cv(self, x_train_processed, y_train, params):
        n_splits = min(5, len(y_train))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []

        for train_idx, valid_idx in splitter.split(range(len(y_train))):
            estimator = self._build_estimator_from_params(params)
            x_fold_train = self._slice_rows(x_train_processed, train_idx)
            x_fold_valid = self._slice_rows(x_train_processed, valid_idx)
            y_fold_train = self._slice_rows(y_train, train_idx)
            y_fold_valid = self._slice_rows(y_train, valid_idx)

            estimator.fit(x_fold_train, y_fold_train)
            predictions = estimator.predict(x_fold_valid)
            scores.append(
                self._compute_metrics(y_fold_valid, predictions)[self.spec.metric]
            )

        return float(np.mean(scores))

    @runtime_dependency(
        module="optuna",
        install_from=OptionalDependency.OPTUNA,
        err_msg=(
            "Please run `python3 -m pip install optuna` to use regression "
            "hyperparameter tuning."
        ),
    )
    def _tune_estimator(self, x_train_processed, y_train):
        base_params = self._build_default_params()
        user_params = dict(self.spec.model_kwargs or {})
        n_trials = self._extract_tuning_n_trials(user_params)
        fallback_params = dict(base_params)
        fallback_params.update(user_params)
        self.tuning_results_df = self.tuning_results_df.iloc[0:0]
        self.best_tuned_params = dict(fallback_params)

        if n_trials <= 0 or len(y_train) < 2:
            return self._build_estimator_from_params(fallback_params)

        max_features_choices = [
            self._default_max_features(max(len(self.feature_columns), 1)),
            "sqrt",
            0.5,
            1.0,
        ]
        max_features_choices = list(dict.fromkeys(max_features_choices))

        def merge_params(params):
            merged = dict(base_params)
            merged.update(params)
            merged.update(user_params)
            if merged.get("min_samples_split", 2) < merged.get("min_samples_leaf", 1) * 2:
                merged["min_samples_split"] = max(
                    2, merged["min_samples_leaf"] * 2
                )
            return merged

        import optuna
        from optuna.trial import TrialState

        def objective(trial):
            params = {}
            if "n_estimators" not in user_params:
                params["n_estimators"] = trial.suggest_int("n_estimators", 100, 600, step=50)
            if "max_depth" not in user_params:
                params["max_depth"] = trial.suggest_categorical(
                    "max_depth", [None, 8, 12, 16, 24]
                )
            if "min_samples_leaf" not in user_params:
                params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 6)
            if "min_samples_split" not in user_params:
                params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 12)
            if "max_features" not in user_params:
                params["max_features"] = trial.suggest_categorical(
                    "max_features", max_features_choices
                )
            if "max_samples" not in user_params and "max_samples" in base_params:
                params["max_samples"] = trial.suggest_float("max_samples", 0.7, 0.95)

            merged = merge_params(params)
            score = self._evaluate_with_cv(x_train_processed, y_train, merged)
            if np.isnan(score):
                raise ValueError("Tuning score is NaN.")
            return score

        study = optuna.create_study(
            direction="maximize" if self.spec.metric == "r2" else "minimize"
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=(Exception,))

        self.tuning_results_df = pd.DataFrame(
            [
                {
                    "candidate": trial.number + 1,
                    "metric": self.spec.metric,
                    "score": trial.value,
                    "state": trial.state.name,
                    "params": json.dumps(
                        self._sanitize_report_value(merge_params(dict(trial.params))),
                        sort_keys=True,
                    ),
                }
                for trial in study.trials
            ]
        )

        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed_trials:
            logger.warning(
                "Random forest tuning produced no completed trials. "
                "Falling back to default parameters."
            )
            return self._build_estimator_from_params(fallback_params)

        self.best_tuned_params = merge_params(dict(study.best_params))
        return self._build_estimator_from_params(self.best_tuned_params)

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
