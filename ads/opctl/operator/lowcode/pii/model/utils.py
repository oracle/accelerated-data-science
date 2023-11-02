#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os
import yaml
import fsspec
import logging

from typing import Dict, List
from html2text import html2text
from striprtf.striprtf import rtf_to_text

YAML_KEYS = [
    "detectors",
    "custom_detectors",
    "spacy_detectors",
    "anonymization",
    "name",
    "label",
    "patterns",
    "model",
    "named_entities",
    "entities",
]


class SupportInputFormat:
    PLAIN = ".txt"
    HTML = ".html"
    RTF = ".rtf"

    MAPPING_EXT_TO_FORMAT = {HTML: "html", RTF: "rtf"}

    @classmethod
    def get_support_list(cls):
        return [cls.PLAIN, cls.HTML, cls.RTF]

    @classmethod
    def map_ext_to_format(cls, ext):
        return cls.MAPPING_EXT_TO_FORMAT.get(ext)


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


# def convert_to_html(file_ext, input_path, file_name):
#     """Example:
#     pandoc -f rtf -t html <input>.rtf -o <output>.html
#     """
#     html_path = os.path.join(tempfile.mkdtemp(), file_name + ".html")
#     cmd_specify_input_format = (
#         ""
#         if file_ext == SupportInputFormat.PLAIN
#         else f"-f {SupportInputFormat.map_ext_to_format(file_ext)}"
#     )
#     cmd = f"pandoc {cmd_specify_input_format} -t html {input_path} -o {html_path}"
#     os.system(cmd)
#     assert os.path.exists(
#         html_path
#     ), f"Failed to convert {input_path} to html. You can run `{cmd}` in terminal to see the error."
#     return html_path


def load_html(uri: str):
    """Convert the given html file to text.

    Args:
        uri (str): uri of the html file.

    Returns:
        str: plain text of the html file.
    """
    fs = open(uri, "rb")
    html = fs.read().decode("utf-8", errors="ignore")
    return html2text(html)


def load_rtf(uri: str, **kwargs):
    """Convert the given rtf file to text.

    Args:
        uri (str): uri of the rtf file.

    Returns:
        str: plain text of the rtf file.
    """
    fsspec_kwargs = kwargs.pop("fsspec_kwargs", {})
    content = _read_from_file(uri, **fsspec_kwargs)
    return rtf_to_text(content)


def get_files(input_dir: str) -> List:
    """Returns all files in the given directory."""
    files = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        if dirpath.endswith(".ipynb_checkpoints"):
            continue
        for f in filenames:
            if not f.endswith(".DS_Store"):
                files.append(os.path.join(dirpath, f))
    return files


def _read_from_file(uri: str, **kwargs) -> str:
    """Returns contents from a file specified by URI

    Parameters
    ----------
    uri : str
        The URI of the file.

    Returns
    -------
    str
        The content of the file as a string.
    """
    with fsspec.open(uri, "r", **kwargs) as f:
        return f.read()


def from_yaml(
    yaml_string: str = None,
    uri: str = None,
    loader: callable = yaml.SafeLoader,
    **kwargs,
) -> Dict:
    """Loads yaml from given yaml string or uri of the yaml.

    Raises
    ------
    ValueError
        Raised if neither string nor uri is provided
    """
    if yaml_string:
        return yaml.load(yaml_string, Loader=loader)
    if uri:
        return yaml.load(_read_from_file(uri=uri, **kwargs), Loader=loader)

    raise ValueError("Must provide either YAML string or URI location")


def _safe_get_spec(spec_file, key, default):
    try:
        return spec_file[key]
    except KeyError as e:
        if not key in YAML_KEYS:
            logging.warning(f"key: `{key}` is not supported.")
        return default


def default_config() -> str:
    """Returns the default config file which intended to process UMHC notes.

    Returns:
        str: uri of the default config file.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(curr_dir, "config", "umhc2.yaml"))


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
