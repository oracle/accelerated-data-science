# !/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import pandas as pd
import fsspec
from .operator_config import AnomalyOperatorSpec
from .const import SupportedMetrics
from ads.opctl import logger

def _build_metrics_df(y_true, y_pred, column_name):
    from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix, \
        roc_auc_score, precision_recall_curve, auc, matthews_corrcoef
    metrics = dict()
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


def _call_pandas_fsspec(pd_fn, filename, storage_options, **kwargs):
    if fsspec.utils.get_protocol(filename) == "file":
        return pd_fn(filename, **kwargs)
    elif fsspec.utils.get_protocol(filename) in ["https", "http"]:
        return pd_fn(filename, **kwargs)
    return pd_fn(filename, storage_options=storage_options, **kwargs)


def _load_data(filename, format, storage_options, columns, **kwargs):
    if not format:
        _, format = os.path.splitext(filename)
        format = format[1:]
    if format in ["json", "clipboard", "excel", "csv", "feather", "hdf"]:
        read_fn = getattr(pd, f"read_{format}")
        data = _call_pandas_fsspec(read_fn, filename, storage_options=storage_options)
        if columns:
            # keep only these columns, done after load because only CSV supports stream filtering
            data = data[columns]
        return data
    raise ValueError(f"Unrecognized format: {format}")


def _write_data(data, filename, format, storage_options, index=False, **kwargs):
    if not format:
        _, format = os.path.splitext(filename)
        format = format[1:]
    if format in ["json", "clipboard", "excel", "csv", "feather", "hdf"]:
        write_fn = getattr(data, f"to_{format}")
        return _call_pandas_fsspec(
            write_fn, filename, index=index, storage_options=storage_options
        )
    raise ValueError(f"Unrecognized format: {format}")

def _merge_category_columns(data, target_category_columns):

    """ Merges target category columns into a single column and returns the column values """

    result = data.apply(
        lambda x: "__".join([str(x[col]) for col in target_category_columns]), axis=1
    )
    return result if not result.empty else pd.Series([], dtype=str)


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
