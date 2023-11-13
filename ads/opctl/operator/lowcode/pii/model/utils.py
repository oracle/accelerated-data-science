#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
import os
import pandas as pd
from typing import Dict, List

from .constant import YAML_KEYS
from ads.common.object_storage_details import ObjectStorageDetails
import fsspec
from ..errors import PIIInputDataError


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


def _load_data(filename, format, storage_options=None, columns=None, **kwargs):
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


def _write_data(data, filename, format, storage_options, index=False, **kwargs):
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


class ReportContextKey:
    RUN_SUMMARY = "run_summary"
    FILE_SUMMARY = "file_summary"
    REPORT_NAME = "report_name"
    TOTAL_FILES = "total_files"
    ELAPSED_TIME = "elapsed_time"
    DATE = "date"
    OUTPUT_DIR = "output_dir"
    INPUT_DIR = "input_dir"
    INPUT = "input"
    TOTAL_T = "total_tokens"
    INPUT_FILE_NAME = "input_file_name"
    OUTPUT_NAME = "output_name"
    ENTITIES = "entities"
    FILE_NAME = "filename"
    INPUT_BASE = "input_base"


def _safe_get_spec(spec_file, key, default):
    try:
        return spec_file[key]
    except KeyError as e:
        if not key in YAML_KEYS:
            logging.warning(f"key: `{key}` is not supported.")
        return default


def construct_filth_cls_name(name: str) -> str:
    """Constructs the filth class name from the given name.
    For example, "name" -> "NameFilth".

    Args:
        name (str): filth class name.

    Returns:
        str: The filth class name.
    """
    return "".join([s.capitalize() for s in name.split("_")]) + "Filth"


def _write_to_file(s: str, uri: str, **kwargs) -> None:
    """Writes the given string to the given uri.

    Args:
        s (str): The string to be written.
        uri (str): The uri of the file to be written.
        kwargs (dict ): keyword arguments to be passed into open().
    """
    with open(uri, "w", **kwargs) as f:
        f.write(s)


def _count_tokens(file_summary):
    """Counts the total number of tokens in the given file summary.

    Args:
        file_summary (dict): file summary.
        e.g. {
            "root1": [
                {..., "total_t": 10, ...},
                {..., "total_t": 3, ...},
            ],
            ...
            }

    Returns:
        int: total number of tokens.
    """
    total_tokens = 0
    for _, files in file_summary.items():
        for file in files:
            total_tokens += file.get("total_tokens")
    return total_tokens


def _process_pos(entities, text) -> List:
    """Processes the position of the given entities."""
    for entity in entities:
        count_line_delimiter = text[: entity.beg].split("\n")
        entity.pos = len(count_line_delimiter)
        entity.line_beg = len(count_line_delimiter[-1])
    return entities
