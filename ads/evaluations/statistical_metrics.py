#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from __future__ import print_function, absolute_import, division

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from ads.common import logger
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)

__all__ = ["ModelEvaluator"]

DEFAULT_BIN_CLASS_METRICS = [
    "accuracy",
    "hamming_loss",
    "precision",
    "recall",
    "f1",
    "auc",
]
DEFAULT_MULTI_CLASS_METRICS = [
    "accuracy",
    "hamming_loss",
    "precision_weighted",
    "precision_micro",
    "recall_weighted",
    "recall_micro",
    "f1_weighted",
    "f1_micro",
]
DEFAULT_REG_METRICS = ["r2_score", "mse", "mae"]
DEFAULT_BIN_CLASS_LABELS_MAP = {
    "accuracy": "Accuracy",
    "hamming_loss": "Hamming distance",
    "kappa_score": "Cohen's kappa coefficient",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "auc": "ROC AUC",
}
DEFAULT_MULTI_CLASS_LABELS_MAP = {
    "accuracy": "Accuracy",
    "hamming_loss": "Hamming distance",
    "precision_weighted": "Precision Weighted Average",
    "precision_micro": "Precision Micro Average",
    "recall_weighted": "Recall Weighted Average",
    "recall_micro": "Recall Micro Average",
    "f1_weighted": "F1 Weighted Average",
    "f1_micro": "F1 Micro Average",
}
DEFAULT_REG_LABELS_MAP = {
    "r2_score": "r-Squared Score",
    "root_mean_squared_error": "Root Mean Squared Error",
    "median_absolute_error": "Median Absolute Error",
}


class ModelEvaluator:
    """
    ModelEvaluator takes in the true and predicted values and returns a pandas dataframe

    Attributes
    ----------
    y_true : array-like object holding the true values for the model
    y_pred : array-like object holding the predicted values for the model
    model_name (str): the name of the model
    classes (list): list of target classes
    positive_class (str): label for positive outcome from model
    y_score : array-like object holding the scores for true values for the model
    metrics (dict): dictionary object holding model data

    Methods
    -------
    get_metrics()
        Gets the metrics information in a dataframe based on the number of classes
    safe_metrics_call(scoring_functions, *args)
        Applies sklearn scoring functions to parameters in args

    """

    def __init__(
        self,
        y_true,
        y_pred,
        model_name,
        classes=None,
        positive_class=None,
        y_score=None,
    ):
        self.y_true = y_true
        self.y_pred = np.squeeze(
            y_pred
        )  # This is a bug where y_pred is shape (1,n) from AutoML, rather than (n)
        self.y_score = y_score
        self.model_name = model_name
        self.classes = classes
        self.positive_class = positive_class
        self.metrics = {
            "model_name": model_name,
            "classes": classes,
            "positive_class": positive_class,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
        }

    def get_metrics(self):
        """
        Gets the metrics information in a dataframe based on the number of classes

        Parameters
        ----------
            self: (`ModelEvaluator` instance)
                The `ModelEvaluator` instance with the metrics.

        Returns
        -------
        :class:`pandas.DataFrame`
            Pandas dataframe containing the metrics
        """

        if self.classes is None:
            self._get_regression_metrics()
            return pd.DataFrame.from_dict(
                self.metrics, orient="index", columns=[self.model_name]
            )
        else:
            if len(self.classes) == 2:  # 'binary'
                self.positive_class = (
                    self.positive_class if self.positive_class else self.classes[0]
                )
                self._get_binary_metrics()
                return pd.DataFrame.from_dict(
                    self.metrics, orient="index", columns=[self.model_name]
                )
            else:  # 'multiclass'
                self._get_multiclass_metrics()
                return pd.DataFrame.from_dict(
                    self.metrics, orient="index", columns=[self.model_name]
                )

    def _get_general_metrics(self):
        try:
            args = [self.y_true, self.y_pred]
            kwargs = {"labels": self.classes}

            scoring_functions = {
                "classification_report": ("classification_report", len(args)),
                "kappa_score": ("cohen_kappa_score", len(args)),
                "raw_confusion_matrix": ("confusion_matrix", len(args)),
                "hinge_loss": ("hinge_loss", len(args)),
            }
            self.safe_metrics_call(scoring_functions, *args, **kwargs)

            args = [self.y_true, self.y_pred]
            scoring_functions_without_labs = {
                "accuracy": ("accuracy_score", len(args)),
                "zero_one_loss": ("zero_one_loss", len(args)),
                "hamming_loss": ("hamming_loss", len(args)),
            }
            self.safe_metrics_call(scoring_functions_without_labs, *args)

            cm = self.metrics["raw_confusion_matrix"]
            a = cm.astype("float")
            b = cm.sum(axis=1)[:, np.newaxis]
            normalized_cm = np.divide(a, b, out=np.zeros_like(a), where=b != 0).tolist()
            self.metrics["confusion_matrix"] = normalized_cm

            vc = pd.DataFrame(self.y_true).value_counts()
            self.metrics["class_balance"] = min(vc) / max(vc)
        except:
            raise ValueError(
                f"Errors arose when attempting to compute metrics. Metrics broke in the state: {self.metrics}"
            )

    def _get_binary_metrics(self):
        self._get_general_metrics()
        # add a metric to determine imbalance
        self.metrics["balanced accuracy"] = metrics.balanced_accuracy_score(
            self.y_true, self.y_pred
        )
        self.metrics["precision"] = metrics.precision_score(
            self.y_true, self.y_pred, pos_label=self.positive_class
        )
        self.metrics["recall"] = metrics.recall_score(
            self.y_true, self.y_pred, pos_label=self.positive_class
        )
        self.metrics["f1"] = metrics.f1_score(
            self.y_true, self.y_pred, pos_label=self.positive_class, average="binary"
        )

        if self.y_score is not None:
            if not all(0 >= x >= 1 for x in self.y_score):
                self.y_score = np.asarray(
                    [0 if x < 0 else 1 if x > 1 else x for x in self.y_score]
                )
            if len(np.asarray(self.y_score).shape) > 1: 
                # If the SKLearn classifier doesn't correctly identify the problem as 
                # binary classification, y_score may be of shape (n_rows, 2) 
                # instead of (n_rows,)
                pos_class_idx = self.classes.index(self.positive_class)
                positive_class_scores = self.y_score[:, pos_class_idx]
            else:
                positive_class_scores = self.y_score
            (
                self.metrics["false_positive_rate"],
                self.metrics["true_positive_rate"],
                _,
            ) = metrics.roc_curve(y_true=self.y_true, y_score=positive_class_scores, pos_label=self.positive_class)
            self.metrics["auc"] = metrics.auc(
                self.metrics["false_positive_rate"], self.metrics["true_positive_rate"]
            )
            self.y_score = list(self.y_score)
            self.metrics["youden_j"] = (
                self.metrics["true_positive_rate"] - self.metrics["false_positive_rate"]
            )
            best_idx = np.argmax(self.metrics["youden_j"])
            self.metrics["roc_best_model_score"] = (
                self.metrics["false_positive_rate"][best_idx],
                self.metrics["true_positive_rate"][best_idx],
            )
            (
                self.metrics["precision_values"],
                self.metrics["recall_values"],
                _,
            ) = metrics.precision_recall_curve(
                self.y_true, self.y_score, pos_label=self.positive_class
            )
            pr_best_idx = np.argmax(
                self.metrics["precision_values"] + self.metrics["recall_values"]
            )
            self.metrics["pr_best_idx"] = pr_best_idx
            self.metrics["pr_best_model_score"] = (
                self.metrics["recall_values"][pr_best_idx],
                self.metrics["precision_values"][pr_best_idx],
            )
            self.metrics["average_precision_score"] = metrics.average_precision_score(
                self.y_true, self.y_score, pos_label=self.positive_class
            )

            self.metrics["brier score"] = metrics.brier_score_loss(
                self.y_true, self.y_score
            )
            self._get_lift_and_gain()
            # Compute KS Statistic curves
            self._binary_ks_curve()

    def _get_lift_and_gain(self):
        # make y_true a boolean vector
        y_true, y_score = np.asarray(self.y_true), np.asarray(self.y_score)
        y_true = y_true == self.positive_class

        sorted_indices = np.argsort(y_score)[::-1]
        y_true = y_true[sorted_indices]
        gains = np.cumsum(y_true)

        percentages = np.arange(start=1, stop=len(y_true) + 1)
        tp = sum(y_true)
        perfect = np.append(
            np.arange(start=1, stop=tp + 1) / float(tp), np.ones(len(y_true) - tp)
        )

        gains = gains / float(np.sum(y_true))
        percentages = percentages / float(len(y_true))

        self.metrics["cumulative_gain"] = np.insert(gains, 0, [0]) * 100
        self.metrics["percentages"] = np.insert(percentages, 0, [0]) * 100
        self.metrics["perfect_gain"] = np.insert(perfect, 0, [0]) * 100

        percentages = percentages[1:]
        self.metrics["lift"] = gains[1:] / percentages
        self.metrics["perfect_lift"] = perfect[1:] / percentages

    def _binary_ks_curve(self):
        """This function generates the points necessary to calculate the KS
        Statistic curve.
        Args:
            y_true (array-like, shape (n_samples)): True labels of the data.
            y_probas (array-like, shape (n_samples)): Probability predictions of
                the positive class.
        Returns:
            thresholds (numpy.ndarray): An array containing the X-axis values for
                plotting the KS Statistic plot.
            pct1 (numpy.ndarray): An array containing the Y-axis values for one
                curve of the KS Statistic plot.
            pct2 (numpy.ndarray): An array containing the Y-axis values for one
                curve of the KS Statistic plot.
            ks_statistic (float): The KS Statistic, or the maximum vertical
                distance between the two curves.
            max_distance_at (float): The X-axis value at which the maximum vertical
                distance between the two curves is seen.
            classes (np.ndarray, shape (2)): An array containing the labels of the
                two classes making up `y_true`.
        Raises:
            ValueError: If `y_true` is not composed of 2 classes. The KS Statistic
                is only relevant in binary classification.
        """
        y_true, y_probas = np.asarray(self.y_true), np.asarray(self.y_score)
        lb = LabelEncoder()
        encoded_labels = lb.fit_transform(y_true)
        if len(lb.classes_) != 2:
            raise ValueError(
                "Cannot calculate KS statistic for data with "
                "{} category/ies".format(len(lb.classes_))
            )
        idx = encoded_labels == 0
        data1 = np.sort(y_probas[idx])
        data2 = np.sort(y_probas[np.logical_not(idx)])

        ctr1, ctr2 = 0, 0
        thresholds, pct1, pct2 = [], [], []
        while ctr1 < len(data1) or ctr2 < len(data2):

            # Check if data1 has no more elements
            if ctr1 >= len(data1):
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1

            # Check if data2 has no more elements
            elif ctr2 >= len(data2):
                current = data1[ctr1]
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

            else:
                if data1[ctr1] > data2[ctr2]:
                    current = data2[ctr2]
                    while ctr2 < len(data2) and current == data2[ctr2]:
                        ctr2 += 1

                elif data1[ctr1] < data2[ctr2]:
                    current = data1[ctr1]
                    while ctr1 < len(data1) and current == data1[ctr1]:
                        ctr1 += 1

                else:
                    current = data2[ctr2]
                    while ctr2 < len(data2) and current == data2[ctr2]:
                        ctr2 += 1
                    while ctr1 < len(data1) and current == data1[ctr1]:
                        ctr1 += 1

            thresholds.append(current)
            pct1.append(ctr1)
            pct2.append(ctr2)

        thresholds = np.asarray(thresholds)
        pct1 = np.asarray(pct1) / float(len(data1))
        pct2 = np.asarray(pct2) / float(len(data2))

        if thresholds[0] != 0:
            thresholds = np.insert(thresholds, 0, [0.0])
            pct1 = np.insert(pct1, 0, [0.0])
            pct2 = np.insert(pct2, 0, [0.0])
        if thresholds[-1] != 1:
            thresholds = np.append(thresholds, [1.0])
            pct1 = np.append(pct1, [1.0])
            pct2 = np.append(pct2, [1.0])

        differences = pct1 - pct2
        self.metrics["ks_statistic"], self.metrics["max_distance_at"] = (
            np.max(differences),
            thresholds[np.argmax(differences)],
        )

        (
            self.metrics["ks_thresholds"],
            self.metrics["ks_pct1"],
            self.metrics["ks_pct2"],
            self.metrics["ks_labels"],
        ) = (thresholds, pct1, pct2, lb.classes_)

    def _get_multiclass_metrics(self):
        self._get_general_metrics()
        y_true, y_pred, y_score = (
            np.asarray(self.y_true),
            np.asarray(self.y_pred),
            np.asarray(self.y_score),
        )
        if not all([x in self.classes for x in np.unique(y_true)]):
            logger.warning(
                f"There are classes in the test dataset that are not specified in the `classes` "
                f"attribute of model ({self.model_name}). This may lead to erroneous "
                f"results."
            )

        self.metrics["precision_weighted"] = metrics.precision_score(
            y_true, y_pred, labels=self.classes, average="weighted"
        )
        self.metrics["precision_micro"] = metrics.precision_score(
            y_true, y_pred, labels=self.classes, average="micro"
        )
        self.metrics["precision_by_label"] = metrics.precision_score(
            y_true, y_pred, labels=self.classes, average=None
        ).tolist()
        self.metrics["recall_weighted"] = metrics.recall_score(
            y_true, y_pred, labels=self.classes, average="weighted"
        )
        self.metrics["recall_micro"] = metrics.recall_score(
            y_true, y_pred, labels=self.classes, average="micro"
        )
        self.metrics["recall_by_label"] = metrics.recall_score(
            y_true, y_pred, labels=self.classes, average=None
        ).tolist()
        self.metrics["f1_weighted"] = metrics.f1_score(
            y_true, y_pred, labels=self.classes, average="weighted"
        )
        self.metrics["f1_micro"] = metrics.f1_score(
            y_true, y_pred, labels=self.classes, average="micro"
        )
        self.metrics["f1_by_label"] = metrics.f1_score(
            y_true, y_pred, labels=self.classes, average=None
        ).tolist()
        self.metrics["jaccard_weighted"] = metrics.jaccard_score(
            y_true, y_pred, labels=self.classes, average="weighted"
        )
        self.metrics["jaccard_micro"] = metrics.jaccard_score(
            y_true, y_pred, labels=self.classes, average="micro"
        )
        self.metrics["jaccard_by_label"] = metrics.jaccard_score(
            y_true, y_pred, labels=self.classes, average=None
        ).tolist()

        if self.y_score is not None:
            # Multiclass ROC
            (
                self.metrics["fpr_by_label"],
                self.metrics["tpr_by_label"],
                self.metrics["auc"],
                self.metrics["roc_best_model_score"],
            ) = (dict(), dict(), dict(), dict())
            for i, label in enumerate(self.classes):
                (
                    self.metrics["fpr_by_label"][i],
                    self.metrics["tpr_by_label"][i],
                    _,
                ) = metrics.roc_curve(y_true, y_score[:, i], pos_label=self.classes[i])
                self.metrics["auc"][i] = metrics.auc(
                    self.metrics["fpr_by_label"][i], self.metrics["tpr_by_label"][i]
                )
                youden_j = (
                    self.metrics["tpr_by_label"][i] - self.metrics["fpr_by_label"][i]
                )
                best_idx = np.argmax(youden_j)
                self.metrics["roc_best_model_score"][i] = (
                    self.metrics["fpr_by_label"][i][best_idx],
                    self.metrics["tpr_by_label"][i][best_idx],
                )
            # Multiclass PR
            (
                self.metrics["recall_values"],
                self.metrics["precision_values"],
                self.metrics["pr_best_model_score"],
            ) = (dict(), dict(), dict())
            for i, label in enumerate(self.classes):
                (
                    self.metrics["precision_values"][i],
                    self.metrics["recall_values"][i],
                    _,
                ) = metrics.precision_recall_curve(
                    y_true, y_score[:, i], pos_label=self.classes[i]
                )
                pr_best_idx = np.argmax(
                    self.metrics["precision_values"][i]
                    + self.metrics["recall_values"][i]
                )
                self.metrics["pr_best_model_score"][i] = (
                    self.metrics["recall_values"][i][pr_best_idx],
                    self.metrics["precision_values"][i][pr_best_idx],
                )

    @runtime_dependency(module="scipy", install_from=OptionalDependency.VIZ)
    def _get_regression_metrics(self):
        self.y_true = np.array(self.y_true)
        self.y_pred = np.array(np.squeeze(self.y_pred))
        args = [self.y_true, self.y_pred]
        scoring_functions = {
            "r2_score": ("r2_score", len(args)),
            "explained_variance_score": ("explained_variance_score", len(args)),
            "max_error": ("max_error", len(args)),
            "mae": ("mean_absolute_error", len(args)),
            "mse": ("mean_squared_error", len(args)),
            "median_absolute_error": ("median_absolute_error", len(args)),
        }

        self.safe_metrics_call(scoring_functions, *args)

        self.metrics["root_mean_squared_error"] = np.sqrt(self.metrics["mse"])
        self.metrics["residuals"] = self.y_true - self.y_pred
        self.metrics["mean_residuals"] = np.mean(self.metrics["residuals"])

        # For QQ Plot:
        portions = min(len(self.metrics["residuals"]), 100) + 1
        norm_quantiles = (np.arange(portions) / portions)[1:]
        self.metrics["norm_quantiles"] = scipy.stats.norm.ppf(norm_quantiles)
        resid_quant = [
            np.quantile(self.metrics["residuals"], p) for p in norm_quantiles
        ]
        self.metrics["residual_quantiles"] = scipy.stats.zscore(resid_quant)

    def safe_metrics_call(self, scoring_functions, *args, **kwargs):
        """Applies the sklearn function in `scoring_functions` to parameters in `args`.

        Parameters
        ----------
            scoring_functions: (dict)
                Scoring functions dictionary
            args: (keyword arguments)
                Arguments passed to the sklearn function from metrics

        Returns:
            Nothing

        Raises:
            Exception: If an error is enountered applying the sklearn function fn to arguments.
        """

        for name, (fn, n_params) in scoring_functions.items():
            try:
                if fn == "confusion_matrix":
                    self.metrics[name] = getattr(metrics, fn)(
                        **{
                            "y_true": args[0],
                            "y_pred": args[1],
                            "labels": kwargs["labels"],
                        }
                    )
                else:
                    self.metrics[name] = getattr(metrics, fn)(
                        *(args[:n_params]), **kwargs
                    )
            except Exception as e:
                self.metrics[name] = f"Error unable to compute {fn}, due to: {e}"
