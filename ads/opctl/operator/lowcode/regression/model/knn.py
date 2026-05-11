#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

from ads.common.decorator.runtime_dependency import OptionalDependency, runtime_dependency
from ads.opctl import logger
from .shared_model import SharedRegressionOperatorModel


class KNNRegressionOperatorModel(SharedRegressionOperatorModel):
    DEFAULT_TUNING_TRIALS = 20

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
        return self._build_estimator_from_params(self.spec.model_kwargs or {})

    def _build_estimator_from_params(self, model_kwargs):
        params = self._build_default_params()
        params.update(model_kwargs or {})
        params.pop("tuning_n_trials", None)
        return KNeighborsRegressor(**params)

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
        row_count = max(len(y_train), 1)
        fallback_params = dict(base_params)
        fallback_params.update(user_params)
        self.tuning_results_df = self.tuning_results_df.iloc[0:0]
        self.best_tuned_params = dict(fallback_params)

        if n_trials <= 0 or len(y_train) < 2:
            return self._build_estimator_from_params(fallback_params)

        def merge_params(params):
            merged = dict(base_params)
            merged.update(params)
            merged.update(user_params)
            merged["n_neighbors"] = max(1, min(row_count, int(merged["n_neighbors"])))
            return merged

        import optuna
        from optuna.trial import TrialState

        def objective(trial):
            upper_neighbors = max(3, min(row_count, 30))
            params = {}
            if "n_neighbors" not in user_params:
                params["n_neighbors"] = trial.suggest_int(
                    "n_neighbors", 1, upper_neighbors
                )
            if "weights" not in user_params:
                params["weights"] = trial.suggest_categorical(
                    "weights", ["uniform", "distance"]
                )
            if "p" not in user_params:
                params["p"] = trial.suggest_categorical("p", [1, 2])

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
                "KNN tuning produced no completed trials. Falling back to default parameters."
            )
            return self._build_estimator_from_params(fallback_params)

        self.best_tuned_params = merge_params(dict(study.best_params))
        return self._build_estimator_from_params(self.best_tuned_params)

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
