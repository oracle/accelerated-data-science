#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from string import Template
from typing import Any, Dict, List, Tuple
import pandas as pd
from ads.opctl import logger
import oracledb

import fsspec
import yaml
from typing import Union

from ads.opctl import logger
from ads.opctl.operator.lowcode.common.errors import (
    InputDataError,
    InvalidParameterError,
    PermissionsError,
    DataMismatchError,
)
from ads.opctl.operator.common.operator_config import OutputDirectory
from ads.common.object_storage_details import ObjectStorageDetails
from ads.secrets import ADBSecretKeeper


def call_pandas_fsspec(pd_fn, filename, storage_options, **kwargs):
    if fsspec.utils.get_protocol(filename) == "file":
        return pd_fn(filename, **kwargs)
    elif fsspec.utils.get_protocol(filename) in ["http", "https"]:
        return pd_fn(filename, **kwargs)

    storage_options = storage_options or (
        default_signer() if ObjectStorageDetails.is_oci_path(filename) else {}
    )

    return pd_fn(filename, storage_options=storage_options, **kwargs)


def load_data(data_spec, storage_options=None, **kwargs):
    if data_spec is None:
        raise InvalidParameterError(f"No details provided for this data source.")
    filename = data_spec.url
    format = data_spec.format
    columns = data_spec.columns
    connect_args = data_spec.connect_args
    sql = data_spec.sql
    table_name = data_spec.table_name
    limit = data_spec.limit
    vault_secret_id = data_spec.vault_secret_id
    storage_options = storage_options or (
        default_signer() if ObjectStorageDetails.is_oci_path(filename) else {}
    )
    if vault_secret_id is not None and connect_args is None:
        connect_args = dict()

    if filename is not None:
        if not format:
            _, format = os.path.splitext(filename)
            format = format[1:]
        if format in ["json", "clipboard", "excel", "csv", "feather", "hdf"]:
            read_fn = getattr(pd, f"read_{format}")
            data = call_pandas_fsspec(
                read_fn, filename, storage_options=storage_options
            )
        elif format in ["tsv"]:
            data = call_pandas_fsspec(
                pd.read_csv, filename, storage_options=storage_options, sep="\t"
            )
        else:
            raise InvalidParameterError(
                f"The format {format} is not currently supported for reading data. Please reformat the data source: {filename} ."
            )
    elif connect_args is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            if vault_secret_id is not None:
                try:
                    with ADBSecretKeeper.load_secret(vault_secret_id, wallet_dir=temp_dir) as adwsecret:
                        if 'wallet_location' in adwsecret and 'wallet_location' not in connect_args:
                            shutil.unpack_archive(adwsecret["wallet_location"], temp_dir)
                            connect_args['wallet_location'] = temp_dir
                        if 'user_name' in adwsecret and 'user' not in connect_args:
                            connect_args['user'] = adwsecret['user_name']
                        if 'password' in adwsecret and 'password' not in connect_args:
                            connect_args['password'] = adwsecret['password']
                        if 'service_name' in adwsecret and 'service_name' not in connect_args:
                            connect_args['service_name'] = adwsecret['service_name']

                except Exception as e:
                    raise Exception(f"Could not retrieve database credentials from vault {vault_secret_id}: {e}")

            con = oracledb.connect(**connect_args)
            if table_name is not None:
                data = pd.read_sql(f"SELECT * FROM {table_name}", con)
            elif sql is not None:
                data = pd.read_sql(sql, con)
            else:
                raise InvalidParameterError(
                    f"Database `connect_args` provided without sql query or table name. Please specify either `sql` or `table_name`."
                )
    else:
        raise InvalidParameterError(
            f"No filename/url provided, and no connect_args provided. Please specify one of these if you want to read data from a file or a database respectively."
        )
    if columns:
        # keep only these columns, done after load because only CSV supports stream filtering
        data = data[columns]
    if limit:
        data = data[:limit]
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


def merged_category_column_name(target_category_columns: Union[List, None]):
    if not target_category_columns or len(target_category_columns) == 0:
        return None
    return "__".join([str(x) for x in target_category_columns])


def datetime_to_seconds(s: pd.Series):
    """
    Method converts a datetime column into an integer number of seconds.
    This method has many uses, most notably for enabling libraries like shap
        to read datetime columns
    ------------
    s: pd.Series
        A Series of type datetime
    Returns
    pd.Series of type int
    """
    return s.apply(lambda x: x.timestamp())


def seconds_to_datetime(s: pd.Series, dt_format=None):
    """
    Inverse of `datetime_to_second`
    ------------
    s: pd.Series
        A Series of type int
    Returns
    pd.Series of type datetime
    """
    s = pd.to_datetime(s, unit="s")
    if dt_format is not None:
        return pd.to_datetime(s, format=dt_format)
    return s


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


def find_output_dirname(output_dir: OutputDirectory):
    if output_dir and output_dir.url:
        return output_dir.url
    output_dir = "results"

    # If the directory exists, find the next unique directory name by appending an incrementing suffix
    counter = 1
    unique_output_dir = f"{output_dir}"
    while os.path.exists(unique_output_dir):
        unique_output_dir = f"{output_dir}_{counter}"
        counter += 1
    logger.warn(
        "Since the output directory was not specified, the output will be saved to {} directory.".format(
            unique_output_dir
        )
    )
    return unique_output_dir


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
