#!/usr/bin/env python

# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
import os
import shutil
import sys
import tempfile
from typing import List, Union

import cloudpickle
import fsspec
import oracledb
import pandas as pd

from ads.common.object_storage_details import ObjectStorageDetails
from ads.opctl import logger
from ads.opctl.operator.common.operator_config import OutputDirectory
from ads.opctl.operator.lowcode.common.errors import (
    InvalidParameterError,
)
from ads.secrets import ADBSecretKeeper
from sktime.param_est.seasonality import SeasonalityACF


def call_pandas_fsspec(pd_fn, filename, storage_options, **kwargs):
    if fsspec.utils.get_protocol(filename) == "file" or fsspec.utils.get_protocol(
        filename
    ) in ["http", "https"]:
        return pd_fn(filename, **kwargs)

    storage_options = storage_options or (
        default_signer() if ObjectStorageDetails.is_oci_path(filename) else {}
    )

    return pd_fn(filename, storage_options=storage_options, **kwargs)


def load_data(data_spec, storage_options=None, **kwargs):
    if data_spec is None:
        raise InvalidParameterError("No details provided for this data source.")
    filename = data_spec.url
    data = data_spec.data
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
        connect_args = {}

    if data is not None:
        if format == "spark":
            data = data.toPandas()
    elif filename is not None:
        if not format:
            _, format = os.path.splitext(filename)
            format = format[1:]
        if format in ["json", "clipboard", "excel", "csv", "feather", "hdf", "parquet"]:
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
                    with ADBSecretKeeper.load_secret(
                        vault_secret_id, wallet_dir=temp_dir
                    ) as adwsecret:
                        if (
                            "wallet_location" in adwsecret
                            and "wallet_location" not in connect_args
                        ):
                            shutil.unpack_archive(
                                adwsecret["wallet_location"], temp_dir
                            )
                            connect_args["wallet_location"] = temp_dir
                        if "user_name" in adwsecret and "user" not in connect_args:
                            connect_args["user"] = adwsecret["user_name"]
                        if "password" in adwsecret and "password" not in connect_args:
                            connect_args["password"] = adwsecret["password"]
                        if (
                            "service_name" in adwsecret
                            and "service_name" not in connect_args
                        ):
                            connect_args["service_name"] = adwsecret["service_name"]

                except Exception as e:
                    raise Exception(
                        f"Could not retrieve database credentials from vault {vault_secret_id}: {e}"
                    ) from e

            con = oracledb.connect(**connect_args)
            if table_name is not None:
                data = pd.read_sql(f"SELECT * FROM {table_name}", con)
            elif sql is not None:
                data = pd.read_sql(sql, con)
            else:
                raise InvalidParameterError(
                    "Database `connect_args` provided without sql query or table name. Please specify either `sql` or `table_name`."
                )
    else:
        raise InvalidParameterError(
            "No filename/url provided, and no connect_args provided. Please specify one of these if you want to read data from a file or a database respectively."
        )
    if columns:
        # keep only these columns, done after load because only CSV supports stream filtering
        data = data[columns]
    if limit:
        data = data[:limit]
    # Filtering by subset if provided
    subset = kwargs.get('subset', None)
    if subset is not None:
        target_category_columns = kwargs.get('target_category_columns', None)
        mask = False
        for col in target_category_columns:
            mask = mask | data[col].isin(subset)
            data = data[mask]
    return data


def _safe_write(fn, **kwargs):
    try:
        fn(**kwargs)
    except Exception:
        logger.warning(f'Failed to write file {kwargs.get("filename", "UNKNOWN")}')


def write_data(data, filename, format, storage_options=None, index=False, **kwargs):
    return _safe_write(
        fn=_write_data,
        data=data,
        filename=filename,
        format=format,
        storage_options=storage_options,
        index=index,
        **kwargs,
    )


def _write_data(data, filename, format, storage_options=None, index=False, **kwargs):
    disable_print()
    if not format:
        _, format = os.path.splitext(filename)
        format = format[1:]
    if format in ["json", "clipboard", "excel", "csv", "feather", "hdf"]:
        write_fn = getattr(data, f"to_{format}")
        return call_pandas_fsspec(
            write_fn, filename, index=index, storage_options=storage_options, **kwargs
        )
    enable_print()
    raise InvalidParameterError(
        f"The format {format} is not currently supported for writing data. Please change the format parameter for the data output: {filename} ."
    )


def write_json(json_dict, filename, storage_options=None):
    return _safe_write(
        fn=_write_json,
        json_dict=json_dict,
        filename=filename,
        storage_options=storage_options,
    )


def _write_json(json_dict, filename, storage_options=None):
    with fsspec.open(filename, mode="w", **storage_options) as f:
        f.write(json.dumps(json_dict))


def write_simple_json(data, path):
    return _safe_write(fn=_write_simple_json, data=data, path=path)


def _write_simple_json(data, path):
    if ObjectStorageDetails.is_oci_path(path):
        storage_options = default_signer()
    else:
        storage_options = {}
    with fsspec.open(path, mode="w", **storage_options) as f:
        json.dump(data, f, indent=4)


def write_file(local_filename, remote_filename, storage_options, **kwargs):
    return _safe_write(
        fn=_write_file,
        local_filename=local_filename,
        remote_filename=remote_filename,
        storage_options=storage_options,
        **kwargs,
    )


def _write_file(local_filename, remote_filename, storage_options, **kwargs):
    with open(local_filename) as f1:
        with fsspec.open(
            remote_filename,
            "w",
            **storage_options,
        ) as f2:
            f2.write(f1.read())


def load_pkl(filepath):
    storage_options = {}
    if ObjectStorageDetails.is_oci_path(filepath):
        storage_options = default_signer()

    with fsspec.open(filepath, "rb", **storage_options) as f:
        return cloudpickle.load(f)


def write_pkl(obj, filename, output_dir, storage_options):
    return _safe_write(
        fn=_write_pkl,
        obj=obj,
        filename=filename,
        output_dir=output_dir,
        storage_options=storage_options,
    )


def _write_pkl(obj, filename, output_dir, storage_options):
    pkl_path = os.path.join(output_dir, filename)
    with fsspec.open(
        pkl_path,
        "wb",
        **storage_options,
    ) as f:
        cloudpickle.dump(obj, f)


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
    return pd.infer_freq(s) or pd.infer_freq(s[-5:])


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
    accumulator.append(f"{round(seconds, 2)} secs")
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
    logger.warning(
        f"Since the output directory was not specified, the output will be saved to {unique_output_dir} directory."
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
    try:
        sys.stdout.close()
    except Exception:
        pass
    sys.stdout = sys.__stdout__


def find_seasonal_period_from_dataset(data: pd.DataFrame) -> tuple[int, list]:
    try:
        sp_est = SeasonalityACF()
        sp_est.fit(data)
        sp = sp_est.get_fitted_params()["sp"]
        probable_sps = sp_est.get_fitted_params()["sp_significant"]
        return sp, probable_sps
    except Exception as e:
        logger.warning(f"Unable to find seasonal period: {e}")
        return None, None


def normalize_frequency(freq: str) -> str:
    """
    Normalize pandas frequency strings to sktime/period-compatible formats.

    Args:
        freq: Pandas frequency string

    Returns:
        Normalized frequency string compatible with PeriodIndex
    """
    if freq is None:
        return None

    freq = freq.upper()

    # Handle weekly frequencies with day anchors (W-SUN, W-MON, etc.)
    if freq.startswith("W-"):
        return "W"

    # Handle month start/end frequencies
    freq_mapping = {
        "MS": "M",  # Month Start -> Month End
        "ME": "M",  # Month End -> Month
        "BMS": "M",  # Business Month Start -> Month
        "BME": "M",  # Business Month End -> Month
        "QS": "Q",  # Quarter Start -> Quarter
        "QE": "Q",  # Quarter End -> Quarter
        "BQS": "Q",  # Business Quarter Start -> Quarter
        "BQE": "Q",  # Business Quarter End -> Quarter
        "YS": "Y",  # Year Start -> Year (Alias: A)
        "AS": "Y",  # Year Start -> Year (Alias: A)
        "YE": "Y",  # Year End -> Year
        "AE": "Y",  # Year End -> Year
        "BYS": "Y",  # Business Year Start -> Year
        "BAS": "Y",  # Business Year Start -> Year
        "BYE": "Y",  # Business Year End -> Year
        "BAE": "Y",  # Business Year End -> Year
    }

    # Handle frequencies with prefixes (e.g., "2W", "3M")
    for old_freq, new_freq in freq_mapping.items():
        if freq.endswith(old_freq):
            prefix = freq[:-len(old_freq)]
            return f"{prefix}{new_freq}" if prefix else new_freq

    return freq