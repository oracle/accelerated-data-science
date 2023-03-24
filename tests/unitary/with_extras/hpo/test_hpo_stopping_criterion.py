#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Contains tests for ads.hpo.stopping_criterion
"""

import unittest
import sklearn
from ads.hpo.distributions import *
from ads.hpo.search_cv import ADSTuner
from ads.hpo.stopping_criterion import *
from sklearn.datasets import load_iris

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


class ADSTunerStoppingCriterionTest(unittest.TestCase):
    """Contains test cases for ads.hpo.stopping_criterion"""

    iris_dataset = load_iris(return_X_y=True)

    @staticmethod
    def get_adstuner(dataset, exit_criterion, synchronous=True):
        """Initializes an ADSTuner instance"""
        X, y = dataset
        X_train, X_valid, y_train, y_valid = train_test_split(X, y)
        clf = XGBClassifier(n_jobs=1)
        ads_search = ADSTuner(clf, random_state=42, scoring="f1_weighted")
        ads_search.tune(
            X_train, y_train, exit_criterion=exit_criterion, synchronous=synchronous
        )
        return ads_search

    def test_hpo_n_trials(self):
        ads_search = self.get_adstuner(self.iris_dataset, [NTrials(2)])
        assert (
            len(ads_search.trials) == 2
        ), "<code>NTrials</code> is not working properly."

    def test_hpo_score_value(self):
        ads_search = self.get_adstuner(self.iris_dataset, [ScoreValue(0.50)])
        assert (
            len(ads_search.trials) > 0 and ads_search.best_score >= 0.5
        ), "<code>ScoreValue</code> is not working properly."

    def test_hpo_time_budget(self):
        ads_search = self.get_adstuner(self.iris_dataset, [TimeBudget(5)])
        assert (
            len(ads_search.trials) > 0
        ), "<code>TimeBudget</code> is not working properly."
