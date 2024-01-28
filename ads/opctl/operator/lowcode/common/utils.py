#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import argparse
import logging
import os
import sys
import time
from string import Template
from typing import Any, Dict, List, Tuple
import pandas as pd

import fsspec
import yaml

from ads.opctl.operator.lowcode.common.errors import (
    InputDataError,
    InvalidParameterError,
    PermissionsError,
    DataMismatchError,
)


def call_pandas_fsspec(pd_fn, filename, storage_options, **kwargs):
    if fsspec.utils.get_protocol(filename) == "file":
        return pd_fn(filename, **kwargs)
    elif fsspec.utils.get_protocol(filename) in ["http", "https"]:
        return pd_fn(filename, **kwargs)

    storage_options = storage_options or (
        default_signer() if ObjectStorageDetails.is_oci_path(filename) else {}
    )

    return pd_fn(filename, storage_options=storage_options, **kwargs)


def load_data(filename, format, storage_options=None, columns=None, **kwargs):
    if filename is None:
        raise InvalidParameterError(
            f"The provided url was blank. Please include a reference to the data."
        )
    if not format:
        _, format = os.path.splitext(filename)
        format = format[1:]
    if format in ["json", "clipboard", "excel", "csv", "feather", "hdf"]:
        read_fn = getattr(pd, f"read_{format}")
        data = call_pandas_fsspec(read_fn, filename, storage_options=storage_options)
    elif format in ["tsv"]:
        data = call_pandas_fsspec(
            pd.read_csv, filename, storage_options=storage_options, sep="\t"
        )
    else:
        raise InvalidParameterError(
            f"The format {format} is not currently supported for reading data. Please reformat the data source: {filename} ."
        )
    if columns:
        # keep only these columns, done after load because only CSV supports stream filtering
        data = data[columns]
    return data


def write_data(data, filename, format, storage_options, index=False, **kwargs):
    if not format:
        _, format = os.path.splitext(filename)
        format = format[1:]
    if format in ["json", "clipboard", "excel", "csv", "feather", "hdf"]:
        write_fn = getattr(data, f"to_{format}")
        return call_pandas_fsspec(
            write_fn, filename, index=index, storage_options=storage_options, **kwargs
        )
    raise OperatorYamlContentError(
        f"The format {format} is not currently supported for writing data. Please change the format parameter for the data output: {filename} ."
    )


def merge_category_columns(data, target_category_columns):
    result = data.apply(
        lambda x: "__".join([str(x[col]) for col in target_category_columns]), axis=1
    )
    return result if not result.empty else pd.Series([], dtype=str)


def default_signer(**kwargs):
    os.environ["EXTRA_USER_AGENT_INFO"] = "Operator"
    from ads.common.auth import default_signer

    return default_signer(**kwargs)


def get_frequency_in_seconds(s: pd.Series, sample_size=100, ignore_duplicates=True):
    """
    Returns frequency of data in seconds

    Parameters
    ------------
    dt_col:  pd.Series  Datetime column
    ignore_duplicates: bool if True, duplicates will be dropped before computing frequency

    Returns
    --------
    int   Minimum difference in seconds
    """
    s1 = pd.Series(s).drop_duplicates() if ignore_duplicates else s
    return s1.tail(20).diff().min().total_seconds()


def get_frequency_of_datetime(dt_col: pd.Series, ignore_duplicates=True):
    """
    Returns string frequency of data

    Parameters
    ------------
    dt_col:  pd.Series  Datetime column
    ignore_duplicates: bool if True, duplicates will be dropped before computing frequency

    Returns
    --------
    str  Pandas Datetime Frequency
    """
    s = pd.Series(dt_col).drop_duplicates() if ignore_duplicates else dt_col
    return pd.infer_freq(s)


def human_time_friendly(seconds):
    TIME_DURATION_UNITS = (
        ("week", 60 * 60 * 24 * 7),
        ("day", 60 * 60 * 24),
        ("hour", 60 * 60),
        ("min", 60),
    )
    if seconds == 0:
        return "inf"
    accumulator = []
    for unit, div in TIME_DURATION_UNITS:
        amount, seconds = divmod(float(seconds), div)
        if amount > 0:
            accumulator.append(
                "{} {}{}".format(int(amount), unit, "" if amount == 1 else "s")
            )
    accumulator.append("{} secs".format(round(seconds, 2)))
    return ", ".join(accumulator)


def set_log_level(pkg_name: str, level: int):
    pkg_logger = logging.getLogger(pkg_name)
    pkg_logger.addHandler(logging.NullHandler())
    pkg_logger.propagate = False
    pkg_logger.setLevel(level)


# Disable
def disable_print():
    sys.stdout = open(os.devnull, "w")


# Restore
def enable_print():
    sys.stdout = sys.__stdout__
