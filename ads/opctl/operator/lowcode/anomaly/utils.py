#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

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
        logger.warn(f"An exception occurred: {e}")
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
