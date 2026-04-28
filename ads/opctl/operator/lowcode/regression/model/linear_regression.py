#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from ads.common.decorator.runtime_dependency import OptionalDependency, runtime_dependency
from ads.opctl import logger

from .shared_model import SharedRegressionOperatorModel


class LinearRegressionOperatorModel(SharedRegressionOperatorModel):
    DEFAULT_TUNING_TRIALS = 20

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
        return self._build_estimator_from_params(self.spec.model_kwargs or {})

    def _build_estimator_from_params(self, model_kwargs):
        params = {
            "fit_intercept": True,
            "copy_X": True,
        }
        params.update(model_kwargs or {})
        params.pop("tuning_n_trials", None)
        linear_regression_n_jobs = params.pop("n_jobs", None)

        penalty = (params.pop("penalty", "none") or "none").lower()
        alpha = params.pop("alpha", 1.0)
        l1_ratio = params.pop("l1_ratio", 0.5)

        if penalty in {"ridge", "l2"}:
            return Ridge(alpha=alpha, **params)
        if penalty in {"lasso", "l1"}:
            return Lasso(alpha=alpha, **params)
        if penalty in {"elasticnet", "elastic"}:
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **params)

        if linear_regression_n_jobs is not None:
            params["n_jobs"] = linear_regression_n_jobs
        return LinearRegression(**params)

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
        user_params = copy.deepcopy(self.spec.model_kwargs or {})
        n_trials = self._extract_tuning_n_trials(user_params)
        fallback_params = dict(user_params)
        fallback_params.setdefault("penalty", "none")
        self.tuning_results_df = self.tuning_results_df.iloc[0:0]
        self.best_tuned_params = dict(fallback_params)

        if n_trials <= 0 or len(y_train) < 2:
            return self._build_estimator_from_params(fallback_params)

        penalty_aliases = {
            "none": "none",
            "ridge": "ridge",
            "l2": "ridge",
            "lasso": "lasso",
            "l1": "lasso",
            "elasticnet": "elasticnet",
            "elastic": "elasticnet",
        }

        def merge_params(params):
            merged = dict(params)
            merged.update(user_params)
            if "penalty" in merged:
                merged["penalty"] = penalty_aliases.get(
                    merged["penalty"].lower(), merged["penalty"].lower()
                )
            return merged

        import optuna
        from optuna.trial import TrialState

        def objective(trial):
            params = {}
            if "penalty" in user_params:
                params["penalty"] = user_params["penalty"]
            else:
                params["penalty"] = trial.suggest_categorical(
                    "penalty", ["none", "ridge", "lasso", "elasticnet"]
                )

            penalty = penalty_aliases.get(params["penalty"].lower(), "none")
            if penalty in {"ridge", "lasso", "elasticnet"} and "alpha" not in user_params:
                params["alpha"] = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
            if penalty == "elasticnet" and "l1_ratio" not in user_params:
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)

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
                "Linear regression tuning produced no completed trials. "
                "Falling back to default parameters."
            )
            return self._build_estimator_from_params(fallback_params)

        self.best_tuned_params = merge_params(dict(study.best_params))
        return self._build_estimator_from_params(self.best_tuned_params)
