#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import configparser
import os
from typing import List

import urllib.parse
import fsspec
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class OperatorNotFound(Exception):   # pragma: no cover
    pass


class CondaPackInfoNotProvided(Exception):   # pragma: no cover
    pass


class NotSupportedError(Exception):   # pragma: no cover
    pass


class _DefaultNoneDict(dict):
    def __missing__(self, key):
        return ""


def read_from_ini(path: str) -> configparser.ConfigParser:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} is not found.")
    parser = configparser.ConfigParser(default_section=None)
    parser.optionxform = str  # preserve case
    parser.read(path)
    return parser


@runtime_dependency(module="nbformat", install_from=OptionalDependency.OPCTL)
@runtime_dependency(module="nbconvert", install_from=OptionalDependency.OPCTL)
def convert_notebook(
    input_path,
    auth,
    exclude_tags: List = None,
    output_path: str = None,
    overwrite: bool = False,
) -> str:
    with fsspec.open(input_path, **auth) as f:
        nb = nbformat.reads(f.read(), nbformat.NO_CONVERT)

    from nbconvert.preprocessors import TagRemovePreprocessor
    from nbconvert.exporters import PythonExporter

    exporter = PythonExporter()
    if exclude_tags:
        exporter.register_preprocessor(
            TagRemovePreprocessor(remove_cell_tags=exclude_tags), enabled=True
        )

    output, _ = exporter.from_notebook_node(nb)
    output = output.strip("\n")
    output_path = output_path or os.path.splitext(input_path)[0] + ".py"
    file_system_clz = fsspec.get_filesystem_class(
        urllib.parse.urlparse(output_path).scheme or "file"
    )
    file_system = file_system_clz(**auth)
    if not overwrite and file_system.exists(output_path):
        raise FileExistsError(
            f"{output_path} exists. Please rename your notebook or use overwrite option."
        )
    with fsspec.open(output_path, mode="wb", **auth) as f:
        f.write(output.encode("utf-8"))
    return output_path
