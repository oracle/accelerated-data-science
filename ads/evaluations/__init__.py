#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import List
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


class EvaluatorMixin:
    def evaluate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike = None,
        y_score: ArrayLike = None,
        X_train: ArrayLike = None,
        y_train: ArrayLike = None,
        classes: List = None,
        positive_class: str = None,
        legend_labels: dict = None,
        perfect: bool = True,
        filename: str = None,
        use_case_type: str = None,
    ):
        """Creates an ads evaluation report.

        Parameters
        ----------
        X : DataFrame-like
            The data used to make a prediction.
            Can be set to None if `y_preds` is given. (And `y_scores` for more thorough analysis).
        y : array-like
            The true values corresponding to the input data
        y_pred : array-like, optional
            The predictions from each model in the same order as the models
        y_score : array-like, optional
            The predict_probas from each model in the same order as the models
        X_train : DataFrame-like, optional
            The data used to train the model
        y_train : array-like, optional
            The true values corresponding to the input training data
        classes : List or None, optional
            A List of the possible labels for y, when evaluating a classification use case
        positive_class : str or int, optional
            The class to report metrics for binary dataset. If the target classes is True or False,
            positive_class will be set to True by default. If the dataset is multiclass or multilabel,
            this will be ignored.
        legend_labels : dict, optional
            List of legend labels. Defaults to `None`.
            If legend_labels not specified class names will be used for plots.
        use_case_type : str, optional
            The type of problem this model is solving. This can be set during `prepare()`.
            Examples: "binary_classification", "regression", "multinomial_classification"
            Full list of supported types can be found here: `ads.common.model_metadata.UseCaseType`
        filename: str, optional
            If filename is given, the html report will be saved to the location specified.

        Examples
        --------
        >>> import tempfile
        >>> from ads.evaluations.evaluator import Evaluator
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from ads.model.framework.sklearn_model import SklearnModel
        >>> from ads.common.model_metadata import UseCaseType
        >>>
        >>> X, y = make_classification(n_samples=1000)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        >>> est = DecisionTreeClassifier().fit(X_train, y_train)
        >>> model = SklearnModel(estimator=est, artifact_dir=tempfile.mkdtemp())
        >>> model.prepare(
                inference_conda_env="generalml_p38_cpu_v1",
                training_conda_env="generalml_p38_cpu_v1",
                X_sample=X_test,
                y_sample=y_test,
                use_case_type=UseCaseType.BINARY_CLASSIFICATION,
            )
        >>> model.evaluate(X_test, y_test, filename="report.html")
        """

        from ads.evaluations.evaluator import Evaluator

        y_preds, y_scores = (
            ([y_pred], [y_score]) if y_pred is not None else (None, None)
        )
        report = Evaluator(
            models=[self],
            X=X,
            y=y,
            y_preds=y_preds,
            y_scores=y_scores,
            X_train=X_train,
            y_train=y_train,
            classes=classes,
            positive_class=positive_class,
            legend_labels=legend_labels,
            use_case_type=use_case_type,
        )
        if filename is None:
            report.display(perfect=perfect)
        else:
            report.save(filename=filename, perfect=perfect)
        return report
