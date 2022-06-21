#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from numbers import Integral, Number
from time import time

import numpy as np

import sklearn
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import BaseCrossValidator  # NOQA
from sklearn.model_selection import check_cv, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils import check_random_state
from sklearn.utils.metaestimators import _safe_split

try:
    from sklearn.utils import _safe_indexing as sklearn_safe_indexing
except:
    from sklearn.utils import safe_indexing as sklearn_safe_indexing

try:
    from sklearn.metrics import check_scoring
except:
    from sklearn.metrics.scorer import check_scoring
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class _Objective(object):
    """
    Callable that implements objective function.


    Parameters
    ----------
        model:
            Object to use to fit the data. This is assumed to implement the
            scikit-learn estimator interface. Either this needs to provide
            ``score``, or ``scoring`` must be passed.
        X:
            Training data.
        y:
            Target variable.
        cv:
            Cross-validation strategy.
        enable_pruning:
            If :obj:`True`, pruning is performed in the case where the
            underlying estimator supports ``partial_fit``.
        error_score:
            Value to assign to the score if an error occurs in fitting. If
            'raise', the error is raised. If numeric,
            ``sklearn.exceptions.FitFailedWarning`` is raised. This does not
            affect the refit step, which will always raise the error.
        fit_params:
            Parameters passed to ``fit`` one the estimator.
        groups:
            Group labels for the samples used while splitting the dataset into
            train/validation set.
        max_iter,
            max number of iteration for ``partial_fit``
        return_train_score:
            If :obj:`True`, training scores will be included. Computing
            training scores is used to get insights on how different
            hyperparameter settings impact the overfitting/underfitting
            trade-off. However computing training scores can be
            computationally expensive and is not strictly required to select
            the hyperparameters that yield the best generalization
            performance. Hence, we default it to be False and do not expose
            this parameter to the users.
        scoring:
            Scorer function.
        scoring_name:
            name of the Scorer function.
        step_name:
            step name of the estimator in a pipeline

        Returns
        -------
            the objective score

    """

    def __init__(
        self,
        model,  # type: Union[BaseEstimator, Pipeline]
        param_distributions,  # type: Mapping[str, distributions.BaseDistribution]
        cv,  # type: BaseCrossValidator
        enable_pruning,  # type: bool
        error_score,  # type: Union[Number, str]
        fit_params,  # type: Dict[str, Any]
        groups,  # type: Optional[OneDimArrayLikeType]
        max_iter,  # type: int
        return_train_score,  # type: bool
        scoring,  # type: Callable[..., Number]
        scoring_name,  # type: str
        step_name,  # type: str
    ):
        # type: (...) -> None
        self.model = model
        self.param_distributions = param_distributions
        self.cv = cv
        self.enable_pruning = enable_pruning
        self.error_score = error_score
        self.fit_params = fit_params
        self.groups = groups
        self.max_iter = max_iter
        self.return_train_score = return_train_score
        self.scoring = scoring
        self.scoring_name = scoring_name
        self.step_name = step_name

    def __call__(self, X, y, trial):
        # type: (trial_module.Trial) -> float

        estimator = clone(self.model)
        params = self._get_params(trial, self.param_distributions)
        params = self._extract_max_iter(params)
        estimator.set_params(**params)

        if self.enable_pruning:
            scores = self._cross_validate_with_pruning(X, y, trial, estimator)
        else:
            scores = cross_validate(
                estimator,
                X,
                y,
                cv=self.cv,
                error_score=self.error_score,
                fit_params=self.fit_params,
                groups=self.groups,
                return_train_score=self.return_train_score,
                scoring=self.scoring,
            )

        self._store_scores(trial, scores, self.scoring_name)

        return trial.user_attrs["mean_test_score"]

    def _extract_max_iter(self, params):
        if self.enable_pruning:
            max_iter_name = "max_iter"
            if self.step_name:
                max_iter_name = self.step_name + "__" + max_iter_name
            if max_iter_name in params:
                self.max_iter = params.pop(max_iter_name)
        return params

    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def _cross_validate_with_pruning(
        self,
        X,
        y,
        trial,  # type: trial_module.Trial
        estimator,  # type: BaseEstimator
    ):
        # type: (...) -> Dict[str, OneDimArrayLikeType]

        if is_classifier(estimator):
            partial_fit_params = self.fit_params.copy()
            classes = np.unique(y)

            partial_fit_params.setdefault("classes", classes)

        else:
            partial_fit_params = self.fit_params.copy()

        n_splits = self.cv.get_n_splits(X, y, groups=self.groups)

        estimators = [clone(estimator) for _ in range(n_splits)]
        scores = {
            "fit_time": np.zeros(n_splits),
            "score_time": np.zeros(n_splits),
            "test_score": np.empty(n_splits),
        }

        if self.return_train_score:
            scores["train_score"] = np.empty(n_splits)

        for step in range(self.max_iter):
            for i, (train, test) in enumerate(self.cv.split(X, y, groups=self.groups)):

                out = self._partial_fit_and_score(
                    X, y, estimators[i], train, test, partial_fit_params
                )

                if self.return_train_score:
                    scores["train_score"][i] = out.pop(0)

                scores["test_score"][i] = out[0]
                scores["fit_time"][i] += out[1]
                scores["score_time"][i] += out[2]

            intermediate_value = np.nanmean(scores["test_score"])

            trial.report(intermediate_value, step=step)

            if trial.should_prune():

                self._store_scores(trial, scores, self.scoring_name)

                raise optuna.TrialPruned(f"trial was pruned at iteration {step}.")
        return scores

    def _get_params(self, trial, param_distributions):
        # type: (trial_module.Trial) -> Dict[str, Any]

        return {
            name: trial._suggest(name, distribution.get_distribution())
            for name, distribution in param_distributions.items()
        }

    def _partial_fit_and_score(
        self,
        X,
        y,
        estimator,  # type: BaseEstimator
        train,  # type: List[int]
        test,  # type: List[int]
        partial_fit_params,  # type: Dict[str, Any]
    ):
        # type: (...) -> List[Number]

        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train_indices=train)

        start_time = time()

        try:
            estimator.partial_fit(X_train, y_train, **partial_fit_params)

        except Exception as e:
            if self.error_score == "raise":
                raise e

            elif isinstance(self.error_score, Number):
                fit_time = time() - start_time
                test_score = self.error_score
                score_time = 0.0

                if self.return_train_score:
                    train_score = self.error_score

            else:
                raise ValueError("error_score must be 'raise' or numeric.")

        else:
            fit_time = time() - start_time
            # Required for type checking but is never expected to fail.
            assert isinstance(fit_time, Number)
            scoring_start_time = time()
            test_score = self.scoring(estimator, X_test, y_test)
            score_time = time() - scoring_start_time
            # Required for type checking but is never expected to fail.
            assert isinstance(score_time, Number)

            if self.return_train_score:
                train_score = self.scoring(estimator, X_train, y_train)

        ret = [test_score, fit_time, score_time]

        if self.return_train_score:
            ret.insert(0, train_score)

        return ret

    def _store_scores(self, trial, scores, scoring_name):
        # type: (trial_module.Trial, Dict[str, OneDimArrayLikeType]) -> None

        trial.set_user_attr("metric", scoring_name)

        for name, array in scores.items():
            if name in ["test_score", "train_score"]:
                for i, score in enumerate(array):
                    trial.set_user_attr("split{}_{}".format(i, name), score)
            trial.set_user_attr("mean_{}".format(name), np.nanmean(array))
            trial.set_user_attr("std_{}".format(name), np.nanstd(array))
