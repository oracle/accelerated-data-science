#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Contains tests for ads.evaluations.evaluator
"""
import pytest
import unittest
import tempfile
from ads.evaluations.evaluator import Evaluator

from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd
from ads.common.model_metadata import UseCaseType
from ads.model.framework.lightgbm_model import LightGBMModel
from ads.model.framework.sklearn_model import SklearnModel


def test_model_types():
    with pytest.raises(ValueError):
        Evaluator(models=["a"], X=[[[1]]], y=[[1]])


def test_pandas_input():
    X, y = make_classification(n_samples=1000, n_informative=5, n_classes=5)
    X = pd.DataFrame(X, columns=[f"col{x}" for x in range(20)])
    y = pd.Series(y, name="target").map(
        {0: "Red", 1: "Orange", 2: "Yellow", 3: "Green", 4: "Blue"}
    )
    est = LGBMClassifier().fit(X, y)
    model = LightGBMModel(estimator=est, artifact_dir=tempfile.mkdtemp())
    model.prepare(
        inference_conda_env="generalml_p38_cpu_v1",
        training_conda_env="generalml_p38_cpu_v1",
        X_sample=X,
        y_sample=y,
        use_case_type=UseCaseType.MULTINOMIAL_CLASSIFICATION,
    )

    report = Evaluator(models=[model], X=X, y=y)
    report.evaluation
    report.display()
    raw_html = report.html()

    y_score = None
    y_pred = model.verify(X)["prediction"]
    report2 = Evaluator([model], X=X, y=y, y_preds=[y_pred], y_scores=[y_score])
    report2.evaluation
    report2.display()
    raw_html2 = report.html()
    assert raw_html != raw_html2
    model.evaluate(X=X, y=y)


class EvaluatorTest(unittest.TestCase):
    """
    Contains test cases for ads.evaluations.evaluator
    """

    inference_conda_env = "generalml_p38_cpu_v1"
    training_conda_env = "generalml_p38_cpu_v1"

    X, y = make_classification(n_samples=1000)
    y = pd.Series(y).map({0: "No", 1: "Yes"}).values
    bin_class_data = train_test_split(X, y, test_size=0.3)

    X, y = make_classification(n_samples=1000, n_informative=5, n_classes=5)
    y = (
        pd.Series(y)
        .map({0: "Red", 1: "Orange", 2: "Yellow", 3: "Green", 4: "Blue"})
        .values
    )
    multi_class_data = train_test_split(X, y, test_size=0.3)

    X, y = make_regression(n_samples=1000)
    reg_data = train_test_split(X, y, test_size=0.3)

    class_models = [
        (LGBMClassifier, LightGBMModel),
        (DecisionTreeClassifier, SklearnModel),
    ]
    reg_models = [
        (LGBMRegressor, LightGBMModel),
        (LinearRegression, SklearnModel),
    ]

    def train_eval_model(self, data, model, use_case):
        X_train, X_test, y_train, y_test = data
        est_class, model_class = model
        est = est_class().fit(X_train, y_train)
        artifact_dir = tempfile.mkdtemp()
        my_model = model_class(estimator=est, artifact_dir=artifact_dir)
        my_model.prepare(
            inference_conda_env="generalml_p38_cpu_v1",
            training_conda_env="generalml_p38_cpu_v1",
            X_sample=X_test,
            y_sample=y_test,
            use_case_type=use_case,
        )
        report = Evaluator([my_model], X=X_test, y=y_test)
        report.evaluation
        report.display()
        raw_html = report.html()
        report.save("report.html")

    def test_regression(self):
        for m in self.reg_models:
            self.train_eval_model(self.reg_data, m, UseCaseType.REGRESSION)
            self.train_eval_model(self.reg_data, m, None)

    def test_multiclass(self):
        for m in self.class_models:
            self.train_eval_model(
                self.multi_class_data, m, UseCaseType.MULTINOMIAL_CLASSIFICATION
            )
            self.train_eval_model(self.multi_class_data, m, None)

    def test_binclass(self):
        for m in self.class_models:
            self.train_eval_model(
                self.bin_class_data, m, UseCaseType.BINARY_CLASSIFICATION
            )
            self.train_eval_model(self.bin_class_data, m, None)
