#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Contains tests for ads.evaluations.evaluator
"""
import numpy as np
import pandas as pd
import unittest

from ads.evaluations.evaluator import ADSEvaluator
from ads.dataset.dataset_browser import DatasetBrowser
from ads.dataset.factory import DatasetFactory
from ads.common.data import ADSData
from ads.common.model import ADSModel
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris

#
# run with:
#  python -m pytest -v -p no:warnings --cov-config=.coveragerc --cov=./ --cov-report html /home/datascience/advanced-ds/tests/unitary/test_evaluations_evaluator.py
#


class EvaluationMetricsTest(unittest.TestCase):
    """
    Contains test cases for ads.evaluations.evaluator
    """

    model_type = linear_model.LogisticRegression()
    fit_models = []
    ds = DatasetBrowser.sklearn().open("iris")
    train, test = ds.train_test_split(test_size=0.15)
    X = train.X.values
    y = train.y.values
    clf = model_type.fit(X, y)
    fit_models.append(ADSModel.from_estimator(clf))

    def test_show_in_notebook_pretty_lable_inplace(self):
        """
        Test show_in_notebook() with copy = False
        Should change ev_test & ev_train
        """
        evaluator = ADSEvaluator(self.test, self.fit_models)
        evaluation_metrics = evaluator.metrics
        ev_test = evaluation_metrics.ev_test
        assert "accuracy" in ev_test.index

        evaluation_metrics.show_in_notebook()

        assert "Accuracy" in ev_test.index

    def test_show_in_notebook_pretty_lable_input_label(self):
        """
        Test show_in_notebook() with input labels
        Should change ev_test & ev_train according to given labels
        """
        evaluator = ADSEvaluator(self.test, self.fit_models)
        evaluation_metrics = evaluator.metrics
        ev_test = evaluation_metrics.ev_test
        assert "accuracy" in ev_test.index

        my_labels = {"accuracy": "My Accuracy"}
        evaluation_metrics.show_in_notebook(my_labels)

        assert "Accuracy" not in ev_test.index
        assert "My Accuracy" in ev_test.index

    def test_EvaluationMetrics_set_precision(self):
        """
        Test precision with
        non-negative/negative input
        float/integer input
        """
        evaluator = ADSEvaluator(self.test, self.fit_models)
        evaluation_metrics = evaluator.metrics
        assert evaluation_metrics.precision == 4

        evaluation_metrics.precision = 0
        assert evaluation_metrics.precision == 0

        evaluation_metrics.precision = 2
        assert evaluation_metrics.precision == 2

        evaluation_metrics.precision = 3.0
        assert evaluation_metrics.precision == 3

        try:
            evaluation_metrics.precision == 2.5
        except:
            pass
        assert evaluation_metrics.precision == 3

        try:
            evaluation_metrics.precision == -1
        except:
            pass
        assert evaluation_metrics.precision == 3

    def test_general_metrics(self):
        X_train, y_train = load_iris(return_X_y=True)
        lr_clf = LogisticRegression(
            random_state=0, solver="lbfgs", multi_class="multinomial"
        ).fit(X_train, y_train)
        rf_clf = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

        lr_model = ADSModel.from_estimator(lr_clf, classes=[0, 1, 2])
        rf_model = ADSModel.from_estimator(rf_clf, classes=[0, 1, 2])
        ads_val = ADSData.build(X=X_train, y=y_train)
        multi_evaluator = ADSEvaluator(ads_val, models=[lr_model, rf_model])
        assert isinstance(multi_evaluator, ADSEvaluator)
        assert isinstance(multi_evaluator.metrics, ADSEvaluator.EvaluationMetrics)

    def test_auc_against_sklearn(self):
        data = {
            "col1": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
            ],
            "col2": np.random.randint(10, size=20),
            "col3": np.random.default_rng().uniform(low=1, high=100, size=20),
            "col4": 100 * np.random.rand(20) + 10,
            "target": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        }
        df = pd.DataFrame(data=data)
        binary_fk = DatasetFactory.open(df, target="target")

        train, test = binary_fk.train_test_split(test_size=0.15)
        X_train = train.X.values
        y_train = train.y.values

        lr_clf = LogisticRegression(
            random_state=0, solver="lbfgs", multi_class="multinomial"
        ).fit(X_train, y_train)

        rf_clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(
            X_train, y_train
        )
        svc_clf = SVC(kernel="linear", C=1.0, random_state=0, probability=True).fit(X_train, y_train)

        bin_lr_model = ADSModel.from_estimator(lr_clf, classes=[0, 1])
        bin_rf_model = ADSModel.from_estimator(rf_clf, classes=[0, 1])
        svc_model = ADSModel.from_estimator(svc_clf, classes=[0, 1])

        bin_evaluator = ADSEvaluator(
            test, models=[bin_lr_model, bin_rf_model, svc_model], training_data=train
        )

        from sklearn.metrics import roc_curve, auc

        sklearn_metrics = {}

        for model_type, model in [
            ("lr", bin_lr_model),
            ("rf", bin_rf_model),
            ("svc", svc_model),
        ]:
            pos_label_idx = model.classes_.index(1)
            fpr, tpr, _ = roc_curve(test.y, model.est.predict_proba(test.X)[:,pos_label_idx], pos_label=1)
            sklearn_metrics[model_type] = round(auc(fpr, tpr), 4)

        assert (
            bin_evaluator.raw_metrics["LogisticRegression"]["auc"]
            - sklearn_metrics["lr"]
            < 0.01
        )
        assert (
            bin_evaluator.raw_metrics["RandomForestClassifier"]["auc"]
            - sklearn_metrics["rf"]
            < 0.01
        )
        assert bin_evaluator.raw_metrics["SVC"]["auc"] - sklearn_metrics["svc"] < 0.01

        # test add_metrics and del_metrics
        from sklearn.metrics import fbeta_score

        def func1(y_true, y_pred):
            return sum(y_true == y_pred)

        def func2(y_true, y_pred):
            return fbeta_score(y_true, y_pred, beta=2)

        bin_evaluator.add_metrics([func1, func2], ["Total True", "F2 Score"])
        bin_evaluator.del_metrics(["Total True", "F2 Score"])

        # test calculate_cost
        bin_evaluator.calculate_cost(0, 1, 1, 0)

        # test add_models and del_models
        from sklearn import tree

        train, _ = binary_fk.train_test_split(test_size=0.15)
        X_train = train.X.values
        y_train = train.y.values

        tree_mod = tree.DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)

        bin_tree_model = ADSModel.from_estimator(tree_mod, classes=[0, 1])
        bin_evaluator.add_models([bin_tree_model])
        bin_evaluator.del_models(["DecisionTreeClassifier"])
