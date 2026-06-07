#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Contains tests for ads.evaluations.evaluator
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ads.common.model_metadata import UseCaseType
from ads.evaluations.evaluator import Evaluator
from ads.model.framework.lightgbm_model import LightGBMModel
from ads.model.framework.sklearn_model import SklearnModel

DEFAULT_PYTHON_VERSION = "3.12"
N_SAMPLES = 200
RANDOM_STATE = 7
LIGHTGBM_PARAMS = {
    "n_estimators": 5,
    "n_jobs": 1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}


def test_model_types():
    with pytest.raises(ValueError):
        Evaluator(models=["a"], X=[[[1]]], y=[[1]])


def test_pandas_input():
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_informative=5,
        n_classes=5,
        random_state=RANDOM_STATE,
    )
    X = pd.DataFrame(X, columns=[f"col{x}" for x in range(20)])
    y = pd.Series(y, name="target").map(
        {0: "Red", 1: "Orange", 2: "Yellow", 3: "Green", 4: "Blue"}
    )
    est = LGBMClassifier(**LIGHTGBM_PARAMS).fit(X, y)
    model = LightGBMModel(estimator=est, artifact_dir=tempfile.mkdtemp())
    model.prepare(
        inference_conda_env="generalml_p38_cpu_v1",
        training_conda_env="generalml_p38_cpu_v1",
        inference_python_version=DEFAULT_PYTHON_VERSION,
        X_sample=X,
        y_sample=y,
        use_case_type=UseCaseType.MULTINOMIAL_CLASSIFICATION,
    )

    report = Evaluator(models=[model], X=X, y=y)
    report.evaluation
    report.display(plots=["roc_curve"])
    raw_html = report.html(plots=["roc_curve"])

    y_score = None
    y_pred = model.verify(X)["prediction"]
    report2 = Evaluator([model], X=X, y=y, y_preds=[y_pred], y_scores=[y_score])
    report2.evaluation
    report2.display(plots=["roc_curve"])
    raw_html2 = report2.html(plots=["roc_curve"])
    assert raw_html != raw_html2
    with patch.object(Evaluator, "display") as mock_display:
        model.evaluate(X=X, y=y)
    mock_display.assert_called_once()


class EvaluatorTest(unittest.TestCase):
    """
    Contains test cases for ads.evaluations.evaluator
    """

    inference_conda_env = "generalml_p38_cpu_v1"
    training_conda_env = "generalml_p38_cpu_v1"

    X, y = make_classification(n_samples=N_SAMPLES, random_state=RANDOM_STATE)
    y = pd.Series(y).map({0: "No", 1: "Yes"}).values
    bin_class_data = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_informative=5,
        n_classes=5,
        random_state=RANDOM_STATE,
    )
    y = (
        pd.Series(y)
        .map({0: "Red", 1: "Orange", 2: "Yellow", 3: "Green", 4: "Blue"})
        .values
    )
    multi_class_data = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    X, y = make_regression(n_samples=N_SAMPLES, random_state=RANDOM_STATE)
    reg_data = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

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
        estimator_kwargs = (
            LIGHTGBM_PARAMS if est_class in [LGBMClassifier, LGBMRegressor] else {}
        )
        est = est_class(**estimator_kwargs).fit(X_train, y_train)
        artifact_dir = tempfile.mkdtemp()
        my_model = model_class(estimator=est, artifact_dir=artifact_dir)
        my_model.prepare(
            inference_conda_env="generalml_p38_cpu_v1",
            training_conda_env="generalml_p38_cpu_v1",
            inference_python_version=DEFAULT_PYTHON_VERSION,
            X_sample=X_test,
            y_sample=y_test,
            use_case_type=use_case,
        )
        report = Evaluator([my_model], X=X_test, y=y_test)
        report.evaluation
        report.display(plots=[])
        raw_html = report.html(plots=[])
        assert raw_html
        report.save(os.path.join(artifact_dir, "report.html"), plots=[])

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
