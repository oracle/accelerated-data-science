#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import argparse
import os
from string import Template
from typing import Any, Dict, List, Tuple

import fsspec
import yaml
from cerberus import Validator
from yaml import SafeLoader

from ads.opctl import logger
from ads.opctl.operator import __operators__
from ads.opctl.utils import run_command

CONTAINER_NETWORK = "CONTAINER_NETWORK"


class OperatorValidator(Validator):
    """The custom validator class."""

    pass


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


def _extant_file(x: str):
    """Checks the extension of the file to yaml."""
    if not (x.lower().endswith(".yml") or x.lower().endswith(".yaml")):
        raise argparse.ArgumentTypeError(
            f"The {x} exists, but must be a yaml file (.yaml/.yml)"
        )
    return x


def _parse_input_args(raw_args: List) -> Tuple:
    """Parses operator input arguments."""
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


def _load_yaml_from_uri(uri: str, **kwargs) -> str:
    """Loads YAML from the URI path. Can be Object Storage path."""
    with fsspec.open(uri) as f:
        return _load_yaml_from_string(str(f.read(), "UTF-8"), **kwargs)
