#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
import os
import sys
from ads.jobs.builders.runtimes.python_runtime import PythonRuntime
import pandas as pd
from urllib.parse import urlparse
import json
import yaml


import ads
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import oci
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score, r2_score, mean_squared_error
import fsspec

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def mape(actual, predicted) -> float:
    if not all([isinstance(actual, np.ndarray),
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), 
        np.array(predicted)
  
    return round(np.mean(np.abs((actual - predicted) / actual)) * 100, 2)

def _call_pandas_fsspec(pd_fn, filename, storage_options, **kwargs):
    if fsspec.utils.get_protocol(filename) == 'file':
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

def _write_data(data, filename, format, storage_options, **kwargs):
    if not format:
        _, format = os.path.splitext(filename)
        format = format[1:]
    if format in ["json", "clipboard", "excel", "csv", "feather", "hdf"]:
        write_fn = getattr(data, f"to_{format}")
        return _call_pandas_fsspec(write_fn, filename, storage_options=storage_options)
    raise ValueError(f"Unrecognized format: {format}")

def _clean_data(data):
    # Todo: KNN Imputer?
    return data.fillna(0)

def evaluate_metrics(target_columns, data, outputs, target_col="yhat"):
    total_metrics = pd.DataFrame()
    for idx, col in enumerate(target_columns):
        metrics = dict()
        y_true = np.asarray(data[col])
        y_pred = np.asarray(outputs[idx][target_col][:len(y_true)])

        metrics["MAPE"] = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
        metrics["RMSE"] = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
        metrics["r2"] = r2_score(y_true=y_true, y_pred=y_pred)
        metrics["Explained Variance"] = explained_variance_score(y_true=y_true, y_pred=y_pred)
        
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=[col])
        total_metrics = pd.concat([total_metrics, metrics_df], axis=1)
    return total_metrics