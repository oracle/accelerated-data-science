#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Contains tests for ads.hpo.tuner_artifact
"""

import os

import lightgbm
import sklearn
import xgboost

from ads.hpo.distributions import *
from ads.hpo.search_cv import ADSTuner
from ads.hpo.stopping_criterion import *

from sklearn.datasets import load_iris, make_regression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import sys, mock, pytest
from ads.common import auth
from ads.common import oci_client as oc
from tests.integration.config import secrets


class TestADSTunerTunerArtifact:
    """Contains test cases for ads.hpo.tuner_artifact
    switch to use mocking test later
    """

    iris_dataset = load_iris(return_X_y=True)
    synthetic_dataset = make_regression(
        n_samples=10000, n_features=10, n_informative=2, random_state=42
    )

    auth = auth.default_signer(client_kwargs={"timeout": 6000})
    client = oc.OCIClientFactory(**auth).object_storage
    bucket_name = secrets.other.BUCKET_3
    name_space = secrets.common.NAMESPACE

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

    def get_adstuner(
        self,
        dataset,
        model=None,
        scoring=None,
        study_name=None,
        storage=None,
        loglevel=20,
        cv=5,
        n=2,
        strategy="perfunctory",
    ):
        """Initializes an ADSTuner instance"""
        X, y = dataset
        if model is None:
            model = sklearn.linear_model.LogisticRegression()
        X_train, X_valid, y_train, y_valid = train_test_split(X, y)
        ads_search = ADSTuner(
            model,
            scoring=scoring,
            study_name=study_name,
            storage=storage,
            random_state=42,
            strategy=strategy,
            loglevel=loglevel,
            cv=cv,
        )
        ads_search.tune(X_train, y_train, exit_criterion=[NTrials(n)], synchronous=True)
        return ads_search

    @staticmethod
    def compare_two_distributions(dist1, dist2):
        assert isinstance(dist1, type(dist2))
        assert dist1.__str__() == dist2.__str__()

    @staticmethod
    def compare_two_strategies(strategy1, strategy2):
        for param, dist in strategy1.items():
            assert TestADSTunerTunerArtifact.compare_two_distributions(
                strategy1[param], strategy2[param]
            )

    @staticmethod
    def compare_two_tuner(tuner1, tuner2):
        """Compare two tuner instances"""
        assert tuner1.storage == tuner2.storage
        assert tuner1.study_name == tuner2.study_name
        # assert ADSTunerTunerArtifact.compare_two_strategies(tuner1.strategy,
        #                                                     tuner2.strategy)
        assert tuner1.model.__class__.__name__ == tuner2.model.__class__.__name__
        assert tuner1.loglevel == tuner2.loglevel
        assert tuner1.metadata == tuner2.metadata
        assert tuner1.scoring == tuner2.scoring
        assert tuner1.cv == tuner2.cv

    @pytest.mark.parametrize("model", regression_list)
    def test_hpo_moving_state_supported_regression_models(self, model):
        file_uri = f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/tuner/{model.__class__.__name__}.zip"
        ads_search = self.get_adstuner(model=model, dataset=self.synthetic_dataset)
        ads_search.trials_export(file_uri, metadata="my metadata for testing purpose")
        tuner = ADSTuner.trials_import(file_uri)
        self.compare_two_tuner(ads_search, tuner)
        self.client.delete_object(
            object_name=f"tuner/{model.__class__.__name__}.zip",
            bucket_name=self.bucket_name,
            namespace_name=self.name_space,
        )

    @pytest.mark.parametrize("model", classification_list)
    def test_hpo_moving_state_supported_classification_models(self, model):
        file_uri = f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/tuner/{model.__class__.__name__}.zip"
        ads_search = self.get_adstuner(model=model, dataset=self.iris_dataset)
        ads_search.trials_export(file_uri, metadata="my metadata for testing purpose")
        tuner = ADSTuner.trials_import(file_uri)
        self.compare_two_tuner(ads_search, tuner)
        self.client.delete_object(
            object_name=f"tuner/{model.__class__.__name__}.zip",
            bucket_name=self.bucket_name,
            namespace_name=self.name_space,
        )

    def test_hpo_moving_state_unsupported_model(self):
        model = AdaBoostClassifier()
        file_uri = f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/tuner/{model.__class__.__name__}.zip"
        ads_search = self.get_adstuner(
            model=model,
            strategy={"learning_rate": LogUniformDistribution(low=0.3, high=0.4)},
            dataset=self.iris_dataset,
        )
        ads_search.trials_export(file_uri, metadata="my metadata for testing purpose")
        tuner = ADSTuner.trials_import(file_uri)
        self.compare_two_tuner(ads_search, tuner)
        self.client.delete_object(
            object_name=f"tuner/{model.__class__.__name__}.zip",
            bucket_name=self.bucket_name,
            namespace_name=self.name_space,
        )

    def test_hpo_moving_state_script(self):
        model = AdaBoostClassifier()
        file_uri = f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/tuner/{model.__class__.__name__}.zip"
        ads_search = self.get_adstuner(
            model=model,
            strategy={"learning_rate": LogUniformDistribution(low=0.3, high=0.4)},
            dataset=self.iris_dataset,
        )
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        ads_search.trials_export(
            file_uri,
            metadata="my metadata for testing purpose",
            script_dict={
                "model": os.path.join(
                    cur_dir, "hpo_tuner_test_files", "customized_model.py"
                )
            },
        )
        tuner = ADSTuner.trials_import(file_uri)
        self.compare_two_tuner(ads_search, tuner)
        self.client.delete_object(
            object_name=f"tuner/{model.__class__.__name__}.zip",
            bucket_name=self.bucket_name,
            namespace_name=self.name_space,
        )

    def test_hpo_moving_state_scoring_script(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        file_uri = f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/tuner/cust_scoring.zip"
        ads_search = self.get_adstuner(
            strategy={"learning_rate": LogUniformDistribution(low=0.3, high=0.4)},
            dataset=self.iris_dataset,
        )
        ads_search.trials_export(
            file_uri,
            metadata="my metadata for testing purpose",
            script_dict={
                "scoring": os.path.join(
                    cur_dir, "hpo_tuner_test_files", "customized_scoring.py"
                )
            },
        )
        tuner = ADSTuner.trials_import(file_uri)
        self.compare_two_tuner(ads_search, tuner)
        self.client.delete_object(
            object_name=f"tuner/cust_scoring.zip",
            bucket_name=self.bucket_name,
            namespace_name=self.name_space,
        )

    def test_hpo_with_optuna_uninstalled(self):
        with mock.patch.dict(sys.modules, {"optuna": None}):
            with pytest.raises(ModuleNotFoundError):
                clf = XGBClassifier(n_jobs=1)
                ads_search = ADSTuner(clf, scoring="f1_weighted")
