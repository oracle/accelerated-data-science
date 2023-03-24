#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import pandas as pd
import pytest
from ads.common.data import ADSData
from ads.evaluations.evaluator import ADSEvaluator
from ads.dataset.dataset_browser import DatasetBrowser
from sklearn.linear_model import LogisticRegression
from ads.common.model import ADSModel

multi_met = {"accuracy": 1.0}
#                     'hamming_loss': 0.25,
#                     'precision': 0.9375,
#                     'recall': 0.75,
#                     'f1': 0.8333333333333334,
#                     'specificity': 0.75,
#                     # 'auc': 0.8125,
#                     'roc_curve': None,
#                     'pr_curve': None,
#                     'gain_chart': None,
#                     'lift_chart': None
#                     }
bin_met = {"accuracy": 1.0, "hamming_loss": 0}

reg_met = {"r2_score": 0.628}

data = pd.DataFrame(
    {
        "sepal_length": [
            5.0,
            5.0,
            4.4,
            5.5,
            5.5,
            5.1,
            6.9,
            6.5,
            5.2,
            6.1,
            5.4,
            6.3,
            7.3,
            6.7,
        ],
        "sepal_width": [
            3.6,
            3.4,
            2.9,
            4.2,
            3.5,
            3.8,
            3.1,
            2.8,
            2.7,
            2.8,
            3,
            2.9,
            2.9,
            2.5,
        ],
        "petal_width": [
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            1.5,
            1.5,
            1.4,
            1.2,
            1.5,
            1.8,
            1.8,
            1.8,
        ],
        "class": [
            "setosa",
            "setosa",
            "setosa",
            "setosa",
            "setosa",
            "setosa",
            "versicolor",
            "versicolor",
            "versicolor",
            "versicolor",
            "versicolor",
            "virginica",
            "virginica",
            "virginica",
        ],
        "petal_length": [
            1.4,
            1.5,
            1.4,
            1.4,
            1.3,
            1.6,
            4.9,
            4.6,
            3.9,
            4.7,
            4.5,
            5.6,
            6.3,
            5.8,
        ],
    }
)


class FakeADSModel:
    def __init__(self, test, prob_type):
        self.prob_type = prob_type
        self.name = test.name
        self.positive_class = "virginica"
        self.is_classifier = prob_type in ["multi", "bin"]
        self.est = self.fake_est()
        if self.is_classifier:
            self.classes_ = list(np.unique(test.y))

    class fake_est:
        def __init__(self):
            self.predict_proba = True

    def predict(self, X):
        y_pred = []
        if self.prob_type == "bin":
            y_pred = [
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "virginica",
                "virginica",
                "virginica",
                "virginica",
                "virginica",
                "virginica",
                "virginica",
                "virginica",
            ]
        elif self.prob_type == "multi":
            y_pred = [
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "versicolor",
                "versicolor",
                "versicolor",
                "versicolor",
                "versicolor",
                "virginica",
                "virginica",
                "virginica",
            ]
        elif self.prob_type == "reg":
            y_pred = [
                3.40291004,
                3.37632733,
                3.14622087,
                3.61681767,
                3.64340038,
                3.39252615,
                3.05672424,
                2.96534625,
                2.54620361,
                2.62045221,
                2.52133216,
                2.76114134,
                3.00287767,
                2.87910204,
            ]
        return y_pred

    def predict_proba(self, X):
        y_proba = None
        if self.prob_type == "bin":
            y_proba = np.asarray(
                [
                    [
                        1.00000000e00,
                        1.00000000e00,
                        1.00000000e00,
                        1.00000000e00,
                        1.00000000e00,
                        1.00000000e00,
                        2.22044605e-16,
                        2.22044605e-16,
                        2.22044605e-16,
                        2.22044605e-16,
                        2.22044605e-16,
                        2.22044605e-16,
                        2.22044605e-16,
                        2.22044605e-16,
                    ],
                    [
                        1.00000001e00,
                        1.00000000e00,
                        1.00000000e00,
                        1.00000000e00,
                        1.00000000e00,
                        1.00000000e00,
                        2.22044605e-16,
                        2.22044605e-16,
                        -2.22044605e-16,
                        2.22044605e-16,
                        2.22044605e-16,
                        2.22044605e-16,
                        2.22044605e-16,
                        2.22044605e-16,
                    ],
                ]
            ).T
        elif self.prob_type == "multi":
            y_proba = [
                [0.99860479, 0.44450374, 0.01035789],
                [0.9989662, 0.45105536, 0.00735547],
                [0.99524763, 0.45261666, 0.01221877],
                [0.99716314, 0.4351522, 0.01810729],
                [0.99690429, 0.45972853, 0.00761727],
                [0.9976732, 0.44349028, 0.0123232],
                [0.00345411, 0.98567229, 0.54221004],
                [0.00152712, 0.98961966, 0.53807905],
                [0.00415428, 0.99502731, 0.48881342],
                [0.0027179, 1.0000001, 0.46357281],
                [0.00582648, 0.99127605, 0.50484923],
                [-1.22044605e-15, 0.49490621, 0.99831679],
                [0.00780772, 0.44797438, 0.99937418],
                [0.00601517, 0.47024469, 0.99618613],
            ]
        elif self.prob_type == "reg":
            y_proba = None
        return y_proba


def test_reg_simple():
    y = data["sepal_width"]
    X = data.drop(["sepal_width"], axis=1)
    test = ADSData(X, y, name="test_model")

    model = FakeADSModel(test, "reg")
    evaluator = ADSEvaluator(test, models=[model])
    evaluation = evaluator.test_evaluations
    evaluator.show_in_notebook()

    for k, v in reg_met.items():
        assert pytest.approx(evaluation.loc[k][0], 0.01) == v


def test_multiclass_simple():
    y = data["class"]
    X = data.drop(["class"], axis=1)
    test = ADSData(X, y, name="test_model")

    model = FakeADSModel(test, "multi")
    evaluator = ADSEvaluator(test, models=[model])
    evaluation = evaluator.test_evaluations
    evaluator.show_in_notebook()

    for k, v in multi_met.items():
        assert evaluation.loc[k][0] == v


@pytest.mark.skip(reason="fails - come back later to reproduce and fix")
def test_bin_simple():
    y = data["class"]
    y_bin = y.replace({"versicolor": "virginica"})
    X = data.drop(["class"], axis=1)
    test = ADSData(X, y_bin, name="test_model")

    model = FakeADSModel(test, "bin")
    evaluator = ADSEvaluator(test, models=[model])
    evaluation = evaluator.test_evaluations
    evaluator.show_in_notebook()
    evaluator = ADSEvaluator(test, models=[model])
    evaluator.show_in_notebook(["roc_curve"])
    evaluator.show_in_notebook(["gain_chart"])

    for k, v in bin_met.items():
        assert evaluation.loc[k][0] == v


def test_same_name():
    y = data["class"]
    X = data.drop(["class"], axis=1)
    test = ADSData(X, y, name="test_model")

    model = FakeADSModel(test, "multi")
    model2 = FakeADSModel(test, "multi")
    evaluator = ADSEvaluator(test, models=[model, model2])
    evaluation = evaluator.test_evaluations
    evaluator.show_in_notebook()

    for k, v in multi_met.items():
        assert evaluation["test_model"][k] == v
        assert evaluation["test_model_1"][k] == v


@pytest.mark.xfail(reason="Bad test")
def test_wrong_class():
    multi_ds = DatasetBrowser.sklearn().open("wine").set_target("target")
    train, test = multi_ds.train_test_split(test_size=0.15)
    X_train = train.X.values
    y_train = train.y.values
    lr_clf = LogisticRegression(
        random_state=0, solver="lbfgs", multi_class="multinomial"
    ).fit(X_train, y_train)
    # this should fail, because the classes are wrong
    lr_model = ADSModel.from_estimator(lr_clf, classes=[0, 1, 2])
    multi_evaluator = ADSEvaluator(test, models=[lr_model])
    assert all(
        np.isnan(
            multi_evaluator.test_evaluations["LogisticRegression"]["tpr_by_label"][0]
        )
    )

    lr_model2 = ADSModel.from_estimator(lr_clf)
    multi_evaluator2 = ADSEvaluator(test, models=[lr_model2])
    assert all(
        multi_evaluator2.test_evaluations["LogisticRegression"]["tpr_by_label"][0]
        == np.asarray([0, 1 / 11, 10 / 11, 10 / 11, 1, 1])
    )
