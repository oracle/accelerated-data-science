#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Contains tests for ads.hpo.search_cv
"""

import os
import unittest

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sklearn
from ads.dataset.factory import DatasetFactory
from ads.hpo.distributions import *
from ads.hpo.search_cv import ADSTuner
from ads.hpo.stopping_criterion import *
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris, load_breast_cancer, make_regression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier


class ADSTunerTest(unittest.TestCase):
    """Contains test cases for ads.hpo.search_cv"""

    iris_dataset = load_iris(return_X_y=True)
    breast_cancer_dataset = (
        load_breast_cancer(as_frame=True).data,
        load_breast_cancer(as_frame=True).target,
    )

    @staticmethod
    def get_adstuner(
        dataset, strategy, model=None, synchronous=True, n=2, t=None, score=None
    ):
        """Initializes an ADSTuner instance"""
        X, y = dataset
        if model is None:
            model = sklearn.linear_model.LogisticRegression()
        X_train, X_valid, y_train, y_valid = train_test_split(X, y)
        ads_search = ADSTuner(model, random_state=42, strategy=strategy)
        if t is not None:
            ads_search.tune(
                X_train,
                y_train,
                exit_criterion=[TimeBudget(t)],
                synchronous=synchronous,
            )
        elif score is not None:
            ads_search.tune(
                X_train,
                y_train,
                exit_criterion=[ScoreValue(score)],
                synchronous=synchronous,
            )
        else:
            ads_search.tune(
                X_train, y_train, exit_criterion=[NTrials(n)], synchronous=synchronous
            )
        return ads_search

    @staticmethod
    def get_adstuner_data_in_constructor(
        dataset, strategy, model=None, synchronous=True, n=2
    ):
        """Initializes an ADSTuner instance"""
        X, y = dataset
        if model is None:
            model = sklearn.linear_model.LogisticRegression()
        X_train, X_valid, y_train, y_valid = train_test_split(X, y)
        ads_search = ADSTuner(model, random_state=42, strategy=strategy, X=X, y=y)
        ads_search.tune(exit_criterion=[NTrials(n)], synchronous=synchronous)
        return ads_search

    def test_hpo_async_behavior(self):
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset, strategy="perfunctory", synchronous=False, n=2
        )
        assert ads_search.status.name == "RUNNING"
        ads_search.halt()
        assert ads_search.status.name == "HALTED"
        ads_search.resume()
        assert ads_search.status.name == "RUNNING"
        ads_search.terminate()
        assert ads_search.status.name == "TERMINATED"
        X, y = self.iris_dataset
        ads_search.tune(X, y, exit_criterion=[NTrials(2)])
        assert ads_search.status.name == "RUNNING"
        ads_search.wait()
        assert ads_search.status.name == "COMPLETED"

    @pytest.mark.skip(
        reason="pytest get stuck due to visualization window does not close automatically."
    )
    def test_hpo_visualization_sync(self):
        model = SGDClassifier()
        ads_search = self.get_adstuner_data_in_constructor(
            dataset=self.iris_dataset,
            model=model,
            strategy="perfunctory",
            synchronous=True,
            n=10,
        )
        assert (
            ads_search.plot_parallel_coordinate_scores() is None
        ), "<code>plot_param_importance</code> does not work."
        assert (
            ads_search.plot_contour_scores() is None
        ), "<code>plot_contour_scores</code> does not work."
        assert (
            ads_search.plot_best_scores() is None
        ), "<code>plot_best_scores</code> does not work."
        assert (
            ads_search.plot_edf_scores() is None
        ), "<code>plot_edf_scores</code> does not work."
        assert (
            ads_search.plot_intermediate_scores() is None
        ), "<code>plot_intermediate_scores</code> does not work."
        assert (
            ads_search.plot_param_importance() is None
        ), "<code>plot_param_importance</code> does not work."

    def test_hpo_visualization_async(self):
        model = LogisticRegression()
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset,
            model=model,
            strategy="detailed",
            synchronous=False,
            n=10,
        )
        assert (
            ads_search.plot_parallel_coordinate_scores() is None
        ), "<code>plot_param_importance</code> does not work."

        ads_search = self.get_adstuner(
            dataset=self.iris_dataset,
            model=model,
            strategy="perfunctory",
            synchronous=False,
            n=10,
        )

        assert (
            ads_search.plot_contour_scores() is None
        ), "<code>plot_contour_scores</code> does not work."
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset,
            model=model,
            strategy="perfunctory",
            synchronous=False,
            n=2,
        )
        assert (
            ads_search.plot_best_scores() is None
        ), "<code>plot_best_scores</code> does not work."
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset,
            model=model,
            strategy="perfunctory",
            synchronous=False,
            n=2,
        )
        assert (
            ads_search.plot_edf_scores() is None
        ), "<code>plot_edf_scores</code> does not work."
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset,
            model=model,
            strategy="perfunctory",
            synchronous=False,
            n=2,
        )
        assert (
            ads_search.plot_intermediate_scores() is None
        ), "<code>plot_intermediate_scores</code> does not work."
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset,
            model=model,
            strategy="perfunctory",
            synchronous=False,
            n=2,
        )
        assert (
            ads_search.plot_param_importance() is None
        ), "<code>plot_param_importance</code> does not work."

    def test_hpo_visualization_exception(self):
        model = SGDClassifier()
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset,
            model=model,
            strategy="perfunctory",
            synchronous=False,
            n=2,
        )
        with pytest.raises(
            ValueError, match="Not all the params are in the search space."
        ):
            ads_search.plot_parallel_coordinate_scores(params=["fake_one", "fake_two"])
        with pytest.raises(
            ValueError, match="Not all the params are in the search space."
        ):
            ads_search.plot_contour_scores(params=["fake_one", "fake_two"])

    def test_hpo_adsdata(self):
        _X, _y = make_regression(
            n_samples=10000, n_features=10, n_informative=2, random_state=42
        )
        df = pd.DataFrame(_X, columns=["F{}".format(x) for x in range(10)])
        df["target"] = pd.Series(_y)
        synthetic_dataset = DatasetFactory.open(df).set_target("target")
        train, test = synthetic_dataset.train_test_split()
        clf = Lasso()
        ads_search = ADSTuner(clf)
        ads_search.tune(train, exit_criterion=[NTrials(2)], synchronous=True)
        assert len(ads_search.trials) > 0

    def build_pipeline(self):
        X, y = self.breast_cancer_dataset
        y = preprocessing.LabelEncoder().fit_transform(y)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        numeric_features = X.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns
        categorical_features = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        num_features = len(numeric_features) + len(categorical_features)
        numeric_transformer = Pipeline(
            steps=[
                ("num_imputer", SimpleImputer(strategy="median")),
                ("num_scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "cat_imputer",
                    SimpleImputer(strategy="constant", fill_value="missing"),
                ),
                ("cat_encoder", ce.woe.WOEEncoder()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        steps = [
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(f_classif, k=int(0.9 * num_features))),
            ("classifier", XGBClassifier(n_estimators=250, n_jobs=1)),
        ]

        pipe = Pipeline(steps=steps)
        return X_train, y_train, pipe, steps

    @staticmethod
    def customerize_score(y_true, y_pred, sample_weight=None):
        score = y_true == y_pred
        return np.average(score, weights=sample_weight)

    def test_hpo_sklearn_pipe(self):
        score = make_scorer(self.customerize_score)
        X_train, y_train, pipe, steps = self.build_pipeline()
        ads_search = ADSTuner(pipe, scoring=score, random_state=42)
        ads_search.tune(
            X=X_train, y=y_train, exit_criterion=[TimeBudget(1)], synchronous=True
        )
        assert len(ads_search.trials) > 0
        assert isinstance(ads_search.best_score, float)
        assert all(
            param.startswith("classifier__")
            for param, _ in ads_search.sklearn_steps.items()
        )
        assert all(
            not param.startswith("classifier__")
            for param, _ in ads_search.search_space().items()
        )
        assert all(
            not param.startswith("classifier__")
            for param, _ in ads_search.best_params.items()
        )

    def test_hpo_sklearn_steps(self):
        score = make_scorer(self.customerize_score)
        X_train, y_train, pipe, steps = self.build_pipeline()
        ads_search_steps = ADSTuner(steps, scoring=score, random_state=42)
        ads_search_steps.tune(
            X=X_train, y=y_train, exit_criterion=[NTrials(1)], synchronous=True
        )
        assert len(ads_search_steps.trials) == 1
        ads_search_steps = ADSTuner(
            steps,
            strategy={"colsample_bytree": UniformDistribution(0.3, 0.7)},
            scoring=score,
            random_state=42,
        )
        ads_search_steps.tune(
            X=X_train, y=y_train, exit_criterion=[NTrials(1)], synchronous=True
        )
        assert len(ads_search_steps.trials) == 1

    def test_hpo_add_remove_param(self):
        model = SGDClassifier()
        ads_search = self.get_adstuner_data_in_constructor(
            dataset=self.iris_dataset,
            strategy={"alpha": LogUniformDistribution(low=0.0001, high=0.1)},
            model=model,
            synchronous=False,
            n=5,
        )
        import time

        ads_search.halt()
        time.sleep(0.5)
        assert len(ads_search.trials) >= 0

        # add params
        ads_search.search_space(
            strategy={
                "alpha": LogUniformDistribution(low=0.0001, high=0.1),
                "penalty": CategoricalDistribution(choices=["l1", "l2", "none"]),
            },
            overwrite=True,
        )
        ads_search.resume()
        time.sleep(0.5)
        from ads.hpo.search_cv import State

        if ads_search.status == State.RUNNING:
            ads_search._tune_process.join()
        assert len(ads_search.trials) == 5
        assert (
            "params_penalty" not in ads_search.trials.columns
            and "params_alpha" in ads_search.trials.columns
        )
        X, y = self.iris_dataset
        # remove params
        ads_search.tune(X=X, y=y, exit_criterion=[NTrials(3)])
        ads_search.search_space(
            strategy={"penalty": CategoricalDistribution(choices=["l1", "l2", "none"])},
            overwrite=True,
        )
        ads_search.terminate()
        ads_search.tune(X=X, y=y, exit_criterion=[NTrials(1)])
        ads_search._tune_process.join()
        assert pd.isnull(ads_search.trials["params_alpha"].iloc[-1])

    @pytest.mark.skip(reason="freeze in python 3.8 and 3.9")
    def test_hpo_score_remaining(self):
        model = SGDClassifier()
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset,
            strategy={"alpha": LogUniformDistribution(low=0.0001, high=0.1)},
            model=model,
            synchronous=False,
            score=0.5,
        )
        import time

        while not ads_search._is_tuning_finished():
            time.sleep(1)

        score_remaining = ads_search.score_remaining
        best_score = ads_search.best_score

        # Score remaining should be negative since the tuning exited on a scoring condition
        assert score_remaining < 0, (
            f"Invalid diff value (remaining, optimal, best) :: {score_remaining} "
            "{ads_search._optimal_score}, {best_score}"
        )
        if not ads_search._is_tuning_finished():
            ads_search.terminate()

    @pytest.mark.skip(reason="freeze in python 3.8 and 3.9")
    def test_hpo_best_scores(self):
        model = SGDClassifier()
        n = 5
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset,
            strategy={"alpha": LogUniformDistribution(low=0.0001, high=0.1)},
            model=model,
            synchronous=False,
            n=n,
        )

        import time

        time.sleep(5)
        assert len(ads_search.trials) >= 1, "No trials have completed"
        start_time = time.time()
        while not ads_search._is_tuning_finished():
            time.sleep(1)
            assert (time.time() - start_time) < 60, "Tuner timed out after 60 seconds"

        best_scores = ads_search.best_scores(n=3)
        assert (
            len(best_scores) == 3
        ), f"Best scores doesn't have enough entries ({best_scores})"
        assert best_scores == sorted(
            best_scores, reverse=True
        ), "Best scores not sorted correctly"

    @pytest.mark.skip(reason="freeze in python 3.8 and 3.9")
    def test_hpo_time_analytics(self):
        model = SGDClassifier()
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset,
            strategy={"alpha": LogUniformDistribution(low=0.0001, high=0.1)},
            model=model,
            synchronous=False,
            n=None,
            t=10,
        )

        import time

        time.sleep(2)
        time_elapsed = ads_search.time_elapsed
        assert time_elapsed > 1, f"More time should have elapsed than {time_elapsed}"
        time_remaining = ads_search.time_remaining
        assert (
            time_remaining < 10
        ), f"Time remaining should be less than {time_remaininig}"

        ads_search.halt()
        assert ads_search.time_since_resume == 0, f"Time since restart should be 0"

        check1 = ads_search.time_elapsed
        time.sleep(0.5)
        check2 = ads_search.time_elapsed
        assert check1 == check2, "Elapsed time should not be changing during halt"

        ads_search.resume()
        time.sleep(1)
        assert (
            ads_search.time_since_resume < 2
        ), "Time elapsed since halt should be less than 2 seconds"

        ads_search.terminate()
        assert ads_search.time_elapsed > time_elapsed, "Elapsed time should increase"
        assert (
            ads_search.time_remaining < time_remaining
        ), "Time remaining should decrease"

    @pytest.mark.skip(reason="freeze in python 3.8 and 3.9")
    def test_hpo_trials_remaining_on_halt(self):
        model = SGDClassifier()
        n = 10
        ads_search = self.get_adstuner(
            dataset=self.iris_dataset,
            strategy={"alpha": LogUniformDistribution(low=0.0001, high=0.1)},
            model=model,
            synchronous=False,
            n=n,
        )

        import time

        time.sleep(5)
        ads_search.halt()
        time.sleep(0.5)

        c = ads_search.trial_count
        r = ads_search.trials_remaining

        assert c > 0, f"No trials have run in time alotted"
        assert r < n, f"Trials should remain"
        assert c + r == n, f"Count ({c}) + remaining ({r}) is wrong"

        ads_search.resume()
        ads_search.terminate()
