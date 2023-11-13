#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
from typing import Dict, List

from .constant import YAML_KEYS


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
