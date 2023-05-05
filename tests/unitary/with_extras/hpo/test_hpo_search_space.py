#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Contains tests for ads.hpo.search_space
"""

import unittest
import lightgbm
import pytest
import sklearn
import xgboost
import sys, mock

from ads.hpo.stopping_criterion import *
from ads.hpo.distributions import *
from ads.hpo.search_cv import ADSTuner

from sklearn.datasets import load_iris, make_regression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split


class TestADSTunerDistributions:
    """Contains test cases for ads.hpo.search_space"""

    iris_dataset = load_iris(return_X_y=True)
    synthetic_dataset = make_regression(
        n_samples=10000, n_features=10, n_informative=2, random_state=42
    )

    strategy_list = ["perfunctory", "detailed"]
    regression_list = [
        sklearn.linear_model._ridge.Ridge(),
        sklearn.linear_model._coordinate_descent.Lasso(),
        sklearn.linear_model._coordinate_descent.ElasticNet(),
        sklearn.svm._classes.SVR(),
        sklearn.svm._classes.LinearSVR(),
        sklearn.tree._classes.DecisionTreeRegressor(),
        sklearn.ensemble._forest.RandomForestRegressor(),
        sklearn.ensemble._gb.GradientBoostingRegressor(),
        xgboost.sklearn.XGBRegressor(n_jobs=1),
        sklearn.ensemble._forest.ExtraTreesRegressor(),
        lightgbm.sklearn.LGBMRegressor(n_jobs=1),
        sklearn.linear_model._stochastic_gradient.SGDRegressor(),
    ]

    classification_list = [
        sklearn.linear_model._ridge.RidgeClassifier(),
        sklearn.linear_model._logistic.LogisticRegression(),
        sklearn.svm._classes.SVC(),
        sklearn.svm._classes.LinearSVC(),
        sklearn.tree._classes.DecisionTreeClassifier(),
        sklearn.ensemble._forest.RandomForestClassifier(),
        sklearn.ensemble._gb.GradientBoostingClassifier(),
        xgboost.sklearn.XGBClassifier(n_jobs=1),
        sklearn.ensemble._forest.ExtraTreesClassifier(),
        lightgbm.sklearn.LGBMClassifier(n_jobs=1),
        sklearn.linear_model._stochastic_gradient.SGDClassifier(),
    ]

    def get_adstuner(self, strategy, model, dataset):
        """Initializes an ADSTuner instance"""
        X, y = dataset
        X_train, X_valid, y_train, y_valid = train_test_split(X, y)
        ads_search = ADSTuner(model, random_state=42, strategy=strategy)
        ads_search.tune(X_train, y_train, exit_criterion=[NTrials(2)], synchronous=True)
        return ads_search

    def test_not_supported_search_space(self):
        clf_not_supported = AdaBoostClassifier()
        with pytest.raises(NotImplementedError) as execinfo:
            ads_search = self.get_adstuner(
                "perfunctory", clf_not_supported, self.iris_dataset
            )
        ads_search = self.get_adstuner(
            strategy={"n_estimators": IntUniformDistribution(50, 100, 10)},
            model=clf_not_supported,
            dataset=self.iris_dataset,
        )

    @pytest.mark.parametrize("strategy", strategy_list)
    @pytest.mark.parametrize("model", regression_list)
    def test_hpo_search_space_regression(self, strategy, model):
        tuner = self.get_adstuner(
            strategy=strategy, model=model, dataset=self.synthetic_dataset
        )
        assert tuner.plot_best_scores() is None
        assert isinstance(tuner.scoring_name, str) and isinstance(
            tuner.best_score, float
        )
        assert tuner.best_index >= 0 and len(tuner.trials) == 2
        assert isinstance(tuner.sklearn_steps, dict) and isinstance(
            tuner.best_params, dict
        )
        assert tuner.random_state is not None and isinstance(tuner.n_trials, int)

    @pytest.mark.parametrize("strategy", strategy_list)
    @pytest.mark.parametrize("model", classification_list)
    def test_hpo_search_space_classification(self, strategy, model):
        tuner = self.get_adstuner(
            strategy=strategy, model=model, dataset=self.iris_dataset
        )
        assert tuner.plot_best_scores() is None
        assert isinstance(tuner.scoring_name, str) and isinstance(
            tuner.best_score, float
        )
        assert tuner.best_index >= 0 and len(tuner.trials) == 2
        assert isinstance(tuner.sklearn_steps, dict) and isinstance(
            tuner.best_params, dict
        )
        assert tuner.random_state is not None and isinstance(tuner.n_trials, int)
