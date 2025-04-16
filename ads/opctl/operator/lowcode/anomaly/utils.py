#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

import numpy as np
import pandas as pd

from ads.opctl import logger

from .const import NonTimeADSupportedModels, SupportedMetrics, SupportedModels
from .operator_config import AnomalyOperatorSpec


def _build_metrics_df(y_true, y_pred, column_name):
    from sklearn.metrics import (
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {}
    np.nan_to_num(y_true, copy=False)
    np.nan_to_num(y_pred, copy=False)
    metrics[SupportedMetrics.RECALL] = recall_score(y_true, y_pred)
    metrics[SupportedMetrics.PRECISION] = precision_score(y_true, y_pred)
    metrics[SupportedMetrics.ACCURACY] = accuracy_score(y_true, y_pred)
    metrics[SupportedMetrics.F1_SCORE] = f1_score(y_true, y_pred)
    tn, *fn_fp_tp = confusion_matrix(y_true, y_pred).ravel()
    fp, fn, tp = fn_fp_tp if fn_fp_tp else (0, 0, 0)
    metrics[SupportedMetrics.FP] = fp
    metrics[SupportedMetrics.FN] = fn
    metrics[SupportedMetrics.TP] = tp
    metrics[SupportedMetrics.TN] = tn
    try:
        # Throws exception if y_true has only one class
        metrics[SupportedMetrics.ROC_AUC] = roc_auc_score(y_true, y_pred)
    except Exception as e:
        logger.warning(f"An exception occurred: {e}")
        metrics[SupportedMetrics.ROC_AUC] = None
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    metrics[SupportedMetrics.PRC_AUC] = auc(recall, precision)
    metrics[SupportedMetrics.MCC] = matthews_corrcoef(y_true, y_pred)
    return pd.DataFrame.from_dict(metrics, orient="index", columns=[column_name])


def get_frequency_of_datetime(data: pd.DataFrame, dataset_info: AnomalyOperatorSpec):
    """
    Function finds the inferred freq from date time column

    Parameters
    ------------
    data:  pd.DataFrame
            primary dataset
    dataset_info:  AnomalyOperatorSpec

    Returns
    --------
    None

    """
    date_column = dataset_info.datetime_column.name
    datetimes = pd.to_datetime(
        data[date_column].drop_duplicates(), format=dataset_info.datetime_column.format
    )
    freq = pd.DatetimeIndex(datetimes).inferred_freq
    return freq


def default_signer(**kwargs):
    os.environ["EXTRA_USER_AGENT_INFO"] = "Anomaly-Detection-Operator"
    from ads.common.auth import default_signer

    return default_signer(**kwargs)


def select_auto_model(operator_config):
    if operator_config.spec.datetime_column is not None:
        return SupportedModels.AutoTS
    return NonTimeADSupportedModels.IsolationForest


import plotly.graph_objects as go
from sklearn.metrics import f1_score


def plot_anomaly_threshold_gain(
    scores, threshold=None, labels=None, title="Threshold Analysis"
):
    """
    Plots:
    - Anomalies detected vs. threshold (always)
    - F1 Score vs. threshold (if labels provided)
    - % of data flagged vs. threshold (if labels not provided)

    Args:
        scores (array-like): Anomaly scores (higher = more anomalous)
        threshold (float): Optional current threshold to highlight
        labels (array-like): Optional true labels (1=anomaly, 0=normal)
        title (str): Chart title
    """
    scores = np.array(scores)
    thresholds = np.linspace(min(scores), max(scores), 100)

    anomalies_found = []
    y_secondary = []
    y_secondary_label = ""

    for t in thresholds:
        predicted = (scores >= t).astype(int)

        # Count anomalies detected
        if labels is not None:
            true_anomalies = np.sum((predicted == 1) & (np.array(labels) == 1))
            # Compute F1 score
            y_secondary.append(f1_score(labels, predicted, zero_division=0))
            y_secondary_label = "F1 Score"
        else:
            true_anomalies = np.sum(predicted)
            # Compute % of data flagged
            y_secondary.append(100 * np.mean(predicted))
            y_secondary_label = "% of Data Flagged"

        anomalies_found.append(true_anomalies)

    # Start building the plot
    fig = go.Figure()

    # Primary Y: Anomalies detected
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=anomalies_found,
            name="Anomalies Detected",
            mode="lines",
            line=dict(color="royalblue"),
            yaxis="y1",
        )
    )

    # Secondary Y: F1 or % flagged
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=y_secondary,
            name=y_secondary_label,
            mode="lines",
            line=dict(color="orange", dash="dot"),
            yaxis="y2",
        )
    )

    # Vertical line for current threshold
    if threshold is not None:
        fig.add_trace(
            go.Scatter(
                x=[threshold, threshold],
                y=[0, max(anomalies_found)],
                mode="lines",
                name=f"Current Threshold ({threshold:.2f})",
                line=dict(color="firebrick", dash="dash"),
                yaxis="y1",
            )
        )

    # Layout
    fig.update_layout(
        title=title,
        xaxis=dict(title="Anomaly Score Threshold"),
        yaxis=dict(title="Anomalies Detected", side="left"),
        yaxis2=dict(
            title=y_secondary_label, overlaying="y", side="right", showgrid=False
        ),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
    )

    fig.show()
