#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Contains tests for ads.hpo.distributions
"""

import unittest
import sklearn
from ads.hpo.distributions import *
from ads.hpo.search_cv import ADSTuner
from ads.hpo.stopping_criterion import *
from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class ADSTunerDistributionsTest(unittest.TestCase):
    """Contains test cases for ads.hpo.distributions"""

    iris_dataset = load_iris(return_X_y=True)

    @staticmethod
    def get_adstuner(dataset, strategy):
        """Initializes an ADSTuner instance"""
        X, y = dataset
        X_train, X_valid, y_train, y_valid = train_test_split(X, y)
        clf = sklearn.linear_model.LogisticRegression()
        ads_search = ADSTuner(clf, random_state=42, strategy=strategy)
        ads_search.tune(X_train, y_train, exit_criterion=[NTrials(2)], synchronous=True)
        return ads_search

    def test_categorical_distribution_list(self):
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset, strategy={"fit_intercept": [True, False]}
        )
        assert (
            "fit_intercept" in ads_search.best_params.keys()
        ), "fit_intercept is not tuned."

    def test_categorical_distribution_single_value(self):
        ads_search = self.get_adstuner(dataset=self.iris_dataset, strategy={"C": 1})
        assert ads_search.best_params["C"] == 1, "C is not fix."

    def test_IntLogUniformDistribution(self):
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset, strategy={"C": IntLogUniformDistribution(1, 5)}
        )
        assert (
            ads_search.best_params["C"] <= 5
            and ads_search.best_params["C"] >= 1
            and isinstance(ads_search.best_params["C"], int)
        ), "C is beyond the range or not int"

    def test_UniformDistribution(self):
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset, strategy={"C": UniformDistribution(1, 2)}
        )
        assert (
            ads_search.best_params["C"] <= 2
            and ads_search.best_params["C"] >= 1
            and isinstance(ads_search.best_params["C"], float)
        )

    def test_DiscreteUniformDistribution(self):
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset,
            strategy={"C": DiscreteUniformDistribution(1, 2, 0.5)},
        )
        assert ads_search.best_params["C"] in [1, 1.5, 2]
