#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import sys

import fsspec
import pandas as pd

from ads.common.object_storage_details import ObjectStorageDetails

from .errors import PIIInputDataError


def default_signer(**kwargs):
    os.environ["EXTRA_USER_AGENT_INFO"] = "Pii-Operator"
    from ads.common.auth import default_signer

    return default_signer(**kwargs)


def _call_pandas_fsspec(pd_fn, filename, storage_options, **kwargs):
    if fsspec.utils.get_protocol(filename) == "file":
        return pd_fn(filename, **kwargs)

    storage_options = storage_options or (
        default_signer() if ObjectStorageDetails.is_oci_path(filename) else {}
    )

    return pd_fn(filename, storage_options=storage_options, **kwargs)


def _load_data(filename, format=None, storage_options=None, columns=None, **kwargs):
    if not format:
        _, format = os.path.splitext(filename)
        format = format[1:]
    if format in ["json", "csv"]:
        read_fn = getattr(pd, f"read_{format}")
        data = _call_pandas_fsspec(read_fn, filename, storage_options=storage_options)
    elif format in ["tsv"]:
        data = _call_pandas_fsspec(
            pd.read_csv, filename, storage_options=storage_options, sep="\t"
        )
    else:
        raise PIIInputDataError(f"Unrecognized format: {format}")
    if columns:
        # keep only these columns, done after load because only CSV supports stream filtering
        data = data[columns]
    return data


def _write_data(
    data, filename, format=None, storage_options=None, index=False, **kwargs
):
    if not format:
        _, format = os.path.splitext(filename)
        format = format[1:]
    if format in ["json", "csv"]:
        write_fn = getattr(data, f"to_{format}")
        return _call_pandas_fsspec(
            write_fn, filename, index=index, storage_options=storage_options
        )
    raise PIIInputDataError(f"Unrecognized format: {format}")


def get_output_name(given_name, target_name=None):
    """Add ``-out`` suffix to the src filename."""
    if not target_name:
        basename = os.path.basename(given_name)
        fn, ext = os.path.splitext(basename)
        target_name = fn + "_out" + ext
    return target_name


def construct_filth_cls_name(name: str) -> str:
    """Constructs the filth class name from the given name.
    For example, "name" -> "NameFilth".

    Args:
        name (str): filth class name.

    Returns:
        str: The filth class name.
    """
    return "".join([s.capitalize() for s in name.split("_")]) + "Filth"


################
# Report utils #
################
def compute_rate(elapsed_time, num_unit):
    return elapsed_time / num_unit


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


# Disable
def block_print():
    sys.stdout = open(os.devnull, "w")


# Restore
def enable_print():
    sys.stdout = sys.__stdout__
