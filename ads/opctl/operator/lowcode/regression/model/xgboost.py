#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ads.common.decorator.runtime_dependency import OptionalDependency, runtime_dependency
from ads.opctl import logger
from ads.opctl.operator.lowcode.regression.const import SupportedMetrics

from .shared_model import SharedRegressionOperatorModel


class XGBoostRegressionOperatorModel(SharedRegressionOperatorModel):
    DEFAULT_TUNING_TRIALS = 20

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
        return self._build_estimator_from_params(self.spec.model_kwargs or {})

    @runtime_dependency(
        module="xgboost",
        install_from=OptionalDependency.OPCTL,
        err_msg="Please run `python3 -m pip install xgboost` to use the `xgboost` regression model.",
    )
    def _build_estimator_from_params(self, model_kwargs):
        from xgboost import XGBRegressor

        params = self._build_default_params()
        params.update(model_kwargs or {})
        params.pop("tuning_n_trials", None)
        return XGBRegressor(**params)

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
        module="xgboost",
        install_from=OptionalDependency.OPCTL,
        err_msg="Please run `python3 -m pip install xgboost` to use the `xgboost` regression model.",
    )
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

        def merge_params(params):
            merged = dict(base_params)
            merged.update(params)
            merged.update(user_params)
            return merged

        import optuna
        from optuna.trial import TrialState

        def objective(trial):
            params = {}
            if "n_estimators" not in user_params:
                params["n_estimators"] = trial.suggest_int(
                    "n_estimators", 100, 600, step=50
                )
            if "max_depth" not in user_params:
                params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            if "learning_rate" not in user_params:
                params["learning_rate"] = trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                )
            if "min_child_weight" not in user_params:
                params["min_child_weight"] = trial.suggest_int(
                    "min_child_weight", 1, 10
                )
            if "subsample" not in user_params:
                params["subsample"] = trial.suggest_float("subsample", 0.7, 1.0)
            if "colsample_bytree" not in user_params:
                params["colsample_bytree"] = trial.suggest_float(
                    "colsample_bytree", 0.7, 1.0
                )
            if "reg_lambda" not in user_params:
                params["reg_lambda"] = trial.suggest_float(
                    "reg_lambda", 0.1, 10.0, log=True
                )

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
                "XGBoost tuning produced no completed trials. "
                "Falling back to default parameters."
            )
            return self._build_estimator_from_params(fallback_params)

        self.best_tuned_params = merge_params(dict(study.best_params))
        return self._build_estimator_from_params(self.best_tuned_params)

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
