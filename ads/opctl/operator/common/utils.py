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


class OperatorValidator(Validator):
    """The custom validator class."""

    pass


@dataclass
class OperatorInfo:
    """Class representing short information about the operator.

    Attributes
    ----------
    name: str
        The name of the operator.
    short_description: str
        The short description of the operator.
    description: str
        The detailed description of the operator.
    version: str
        The version of the operator.
    conda: str
        The conda environment that have to be used to run the operator.
    """

    name: str
    short_description: str
    description: str
    version: str
    conda: str

    @classmethod
    def from_init(*args: List, **kwargs: Dict) -> "OperatorInfo":
        """Instantiates the class from the initial operator details config."""
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
    Class for generating the YAML config based on the given YAML schema.

    Attributes
    ----------
    schema: Dict
        The schema of the template.
    """

    schema: Dict[str, Any] = None

    def generate_example(self, values: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate the YAML config based on the YAML schema.

        Properties
        ----------
        values: Optional dictionary containing specific values for the attributes.

        Returns
        -------
        str
            The generated YAML config.
        """
        example = self._generate_example(self.schema, values)
        return yaml.dump(example)

    def _check_condition(
        self, condition: Dict[str, Any], example: Dict[str, Any]
    ) -> bool:
        """
        Checks if the YAML schema condition fulfils.
        This method is used to include conditional fields into the final config.

        Properties
        ----------
        condition: Dict[str, Any]
            The schema condition.
            Example:
            In the example below the `owner_name` field has dependency on the `model` field.
            The `owner_name` will be included to the final config if only `model` is `prophet`.
                owner_name:
                    type: string
                    dependencies: {"model":"prophet"}
        example: Dict[str, Any]
            The config to check if the dependable value presented there.
        Returns
        -------
        bool
            True if the condition fulfils, false otherwise.
        """
        for key, value in condition.items():
            if key not in example or example[key] != value:
                return False
        return True

    def _generate_example(
        self, schema: Dict[str, Any], values: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates the final YAML config.
        This is a recursive method, which iterates through the entire schema.

        Properties
        ----------
        schema: Dict[str, Any]
            The schema to generate the config.
        values: Optional[Dict[str, Any]]
            The optional values that would be used instead of default values provided in the schama.

        Returns
        -------
        Dict
            The result config.
        """
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


def _build_image(
    dockerfile: str,
    image_name: str,
    tag: str = None,
    target: str = None,
    **kwargs: Dict[str, Any],
) -> str:
    """
    Builds the operator image.

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
    kwargs: (Dict, optional).
        Additional key value arguments.

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


def _extant_file(x: str):
    """Checks the extension of the file to yaml."""
    if not (x.lower().endswith(".yml") or x.lower().endswith(".yaml")):
        raise argparse.ArgumentTypeError(
            f"{x} exists, but must be a yaml file (.yaml/.yml)"
        )
    return x


def _parse_input_args(raw_args) -> Tuple:
    """Parses operator inout arguments."""
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


def _load_yaml_from_string(doc: str, **kwargs) -> Dict:
    """Loads YAML from string and merge it with env variables and kwargs."""
    template_dict = {**os.environ, **kwargs}
    return yaml.safe_load(
        Template(doc).safe_substitute(
            **template_dict,
        )
    )


def _load_multi_document_yaml_from_string(doc: str, **kwargs) -> Dict:
    """Loads multiline YAML from string and merge it with env variables and kwargs."""
    template_dict = {**os.environ, **kwargs}
    return yaml.load_all(
        Template(doc).substitute(
            **template_dict,
        ),
        Loader=SafeLoader,
    )


def _load_multi_document_yaml_from_uri(uri: str, **kwargs) -> Dict:
    """Loads multiline YAML from file and merge it with env variables and kwargs."""
    with fsspec.open(uri) as f:
        return _load_multi_document_yaml_from_string(str(f.read(), "UTF-8"), **kwargs)


def _load_yaml_from_uri(uri: str, **kwargs) -> str:
    """Loads YAML from the URI path. Can be OS path."""
    with fsspec.open(uri) as f:
        return _load_yaml_from_string(str(f.read(), "UTF-8"), **kwargs)


def _convert_schema_to_html(module_name: str, module_schema: str) -> str:
    """Converts operator YAML schema to HTML."""
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
