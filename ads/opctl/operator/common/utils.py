#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import argparse
import importlib
import inspect
import os
import re
from dataclasses import dataclass
from string import Template
from typing import Any, Dict, List, Optional, Tuple

import fsspec
import yaml
from cerberus import Validator
from json2table import convert
from yaml import SafeLoader

from ads.opctl import logger
from ads.opctl.constants import OPERATOR_MODULE_PATH
from ads.opctl.operator import __operators__
from ads.opctl.utils import run_command

CONTAINER_NETWORK = "CONTAINER_NETWORK"


@dataclass
class OperatorInfo:
    """Class to represent short information about the operator."""

    name: str
    short_description: str
    description: str
    version: str
    conda: str

    @classmethod
    def from_init(*args, **kwargs) -> "OperatorInfo":
        """Instantiates class from init config."""
        return OperatorInfo(
            name=kwargs.get("__name__"),
            description=kwargs.get("__description__"),
            short_description=kwargs.get("__short_description__"),
            version=kwargs.get("__version__"),
            conda=kwargs.get("__conda__"),
        )


@dataclass
class YamlGenerator:
    """
    Class for generating example YAML based on a schema.

    Attributes
    ----------
    schema: Dict
        The schema of the template.
    """

    schema: Dict[str, Any] = None

    def generate_example(self, values: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an example YAML based on the schema.

        Properties
        ----------
        values: Optional dictionary containing specific values for attributes.

        Returns
        -------
        The generated example YAML as a string.
        """
        example = self._generate_example(self.schema, values)
        return yaml.dump(example)

    def _check_condition(
        self, condition: Dict[str, Any], example: Dict[str, Any]
    ) -> bool:
        for key, value in condition.items():
            if key not in example or example[key] != value:
                return False
        return True

    def _generate_example(
        self, schema: Dict[str, Any], values: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        example = {}
        for key, value in schema.items():
            # only generate values fro required fields
            if value.get("required", False) or value.get("dependencies", False):
                if not "dependencies" in value or self._check_condition(
                    value["dependencies"], example
                ):
                    data_type = value.get("type")

                    if key in values:
                        example[key] = values[key]
                    elif "default" in value:
                        example[key] = value["default"]
                    elif data_type == "string":
                        example[key] = "value"
                    elif data_type == "number":
                        example[key] = 1
                    elif data_type == "boolean":
                        example[key] = True
                    elif data_type == "array":
                        example[key] = ["item1", "item2"]
                    elif data_type == "dict":
                        example[key] = self._generate_example(
                            schema=value.get("schema", {}), values=values
                        )
        return example


class OperatorValidator(Validator):
    pass


def _extant_file(x: str):
    if not (x.endswith(".yml") or x.endswith(".yaml")):
        raise argparse.ArgumentTypeError(
            f"{x} exists, but must be a yaml file (.yaml/.yml)"
        )
    return x


def _parse_input_args(raw_args) -> Tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=_extant_file,
        required=False,
        help="Path to the operator specification YAML file",
    )
    parser.add_argument(
        "-s", "--spec", type=str, required=False, help="Operator Yaml specification"
    )
    parser.add_argument(
        "-v",
        "--verify",
        type=bool,
        default=False,
        required=False,
        help="Verify operator schema",
    )
    return parser.parse_known_args(raw_args)


def _module_constant_values(module_name: str, module_path: str) -> Dict[str, Any]:
    """Returns the list of constant variables from a given module.

    module_name: str
        The name of the module to be imported.
    module_path: str
        The physical path of the module.
    Returns
    -------
    Dict[str, Any]
        Map of variable names and their values.
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: value for name, value in vars(module).items()}


def _operator_info(path: str) -> OperatorInfo:
    """Extracts operator's details by given path.
    The expectation is that operator has init file where the all details placed.

    Returns
    -------
    OperatorInfo
        The operator details.
    """
    module_name = os.path.basename(path.rstrip("/"))
    module_path = f"{path.rstrip('/')}/__init__.py"
    return OperatorInfo.from_init(**_module_constant_values(module_name, module_path))


def _operator_info_list() -> List[OperatorInfo]:
    """Returns the list of registered operators.

    Returns
    -------
    List[OperatorInfo]
        The list of registered operators.
    """
    return (
        _operator_info(
            os.path.dirname(
                inspect.getfile(
                    importlib.import_module(f"{OPERATOR_MODULE_PATH}.{operator_name}")
                )
            )
        )
        for operator_name in __operators__
    )


def text_clean(txt: str) -> str:
    """Cleans the text from the special characters."""

    txt = re.sub("httpS+s*", " ", txt)  # remove URLs
    txt = re.sub("RT|cc", " ", txt)  # remove RT and cc
    # txt = re.sub("#S+", "", txt)  # remove hashtags
    txt = re.sub("@S+", "  ", txt)  # remove mentions
    txt = re.sub(
        "[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), " ", txt
    )  # remove punctuations
    txt = re.sub(r"[^x00-x7f]", r" ", txt)
    txt = re.sub("s+", " ", txt)  # remove extra whitespace
    return txt


def _load_yaml_from_string(doc: str, **kwargs) -> Dict:
    template_dict = {**os.environ, **kwargs}
    return yaml.safe_load(
        Template(doc).safe_substitute(
            **template_dict,
        )
    )


def _load_multi_document_yaml_from_string(doc: str, **kwargs) -> Dict:
    template_dict = {**os.environ, **kwargs}
    return yaml.load_all(
        Template(doc).substitute(
            **template_dict,
        ),
        Loader=SafeLoader,
    )


def _load_multi_document_yaml_from_uri(uri: str, **kwargs) -> Dict:
    with fsspec.open(uri) as f:
        return _load_multi_document_yaml_from_string(str(f.read(), "UTF-8"), **kwargs)


def _load_yaml_from_uri(uri: str, **kwargs) -> str:
    """Loads YAML from the URI path. Can be OS path."""
    with fsspec.open(uri) as f:
        return _load_yaml_from_string(str(f.read(), "UTF-8"), **kwargs)


def _build_image(
    dockerfile: str,
    image_name: str,
    tag: str = None,
    target: str = None,
    **kwargs: Dict[str, Any],
) -> str:
    """
    Build an image for operator.

    Parameters
    ----------
    dockerfile: str
        Path to the docker file.
    image_name: str
        The name of the image.
    tag: (str, optional)
        The tag of the image.
    target: (str, optional)
        The image target.

    Returns
    -------
    str
        The final image name.

    Raises
    ------
    ValueError
        When dockerfile or image name not provided.
    FileNotFoundError
        When dockerfile doesn't exist.
    RuntimeError
        When docker build operation fails.
    """
    if not (dockerfile and image_name):
        raise ValueError("The `dockerfile` and `image_name` needs to be provided.")

    if not os.path.isfile(dockerfile):
        raise FileNotFoundError(f"The file `{dockerfile}` does not exist")

    image_name = f"{image_name}:{tag or 'latest'}"

    command = [
        "docker",
        "build",
        "-t",
        image_name,
        "-f",
        dockerfile,
    ]

    if target:
        command += ["--target", target]
    if os.environ.get("no_proxy"):
        command += ["--build-arg", f"no_proxy={os.environ['no_proxy']}"]
    if os.environ.get("http_proxy"):
        command += ["--build-arg", f"http_proxy={os.environ['http_proxy']}"]
    if os.environ.get("https_proxy"):
        command += ["--build-arg", f"https_proxy={os.environ['https_proxy']}"]
    if os.environ.get(CONTAINER_NETWORK):
        command += ["--network", os.environ[CONTAINER_NETWORK]]
    command += [os.path.dirname(dockerfile)]

    logger.info(f"Build image: {command}")

    proc = run_command(command)
    if proc.returncode != 0:
        raise RuntimeError("Docker build failed.")

    return image_name


def _convert_schema_to_html(module_name: str, module_schema: str) -> str:
    t = Template(
        """
        <style type="text/css">
          table {
            background: #fff;
            font-family: monospace;
            font-size: 1.0rem;
          }

          table,
          thead,
          tbody,
          tfoot,
          tr,
          td,
          th {
            margin: auto;
            border: 1px solid #ececec;
            padding: 0.5rem;
          }

          table {
            display: table;
            width: 50%;
          }

          tr {
            display: table-row;
          }

          thead {
            display: table-header-group
          }

          tbody {
            display: table-row-group
          }

          tfoot {
            display: table-footer-group
          }

          col {
            display: table-column
          }

          colgroup {
            display: table-column-group
          }

          td,
          th {
            display: table-cell;
            width: 50%;
          }

          caption {
            display: table-caption
          }

          table,
          thead,
          tbody,
          tfoot,
          tr,
          td,
          th {
            margin: auto;
            padding: 0.5rem;
          }

          table {
            background: #fff;
            margin: auto;
            border: none;
            padding: 0;
            margin-bottom: 2rem;
          }

          th {
            text-align: right;
            font-weight: 700;
            border: 1px solid #ececec;

          }
        </style>
        <h1>Operator: $module_name</h1>

        $table

    """
    )

    return t.substitute(
        module_name=module_name,
        table=convert(
            OperatorValidator(module_schema, allow_unknown=True).schema.schema,
            build_direction="LEFT_TO_RIGHT",
        ),
    )
