#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict

import click
import fsspec
import yaml

from ads.common.auth import AuthContext, AuthType
from ads.opctl.decorator.common import click_options
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.utils import suppress_traceback

from .__init__ import __operators__
from .cmd import build_conda as cmd_build_conda
from .cmd import build_image as cmd_build_image
from .cmd import create as cmd_create
from .cmd import info as cmd_info
from .cmd import init as cmd_init
from .cmd import list as cmd_list
from .cmd import publish_conda as cmd_publish_conda
from .cmd import publish_image as cmd_publish_image
from .cmd import verify as cmd_verify

DEBUG_OPTION = (
    click.option("--debug", "-d", help="Set debug mode.", is_flag=True, default=False),
)

ADS_CONFIG_OPTION = (
    click.option(
        "--ads-config",
        help=(
            "The folder where the ADS opctl config located. "
            "The default location is: `~/.ads_ops` folder."
        ),
        required=False,
        default=None,
    ),
)

OPERATOR_NAME_OPTION = (
    click.option(
        "--name",
        "-n",
        help=(
            "The name of the service operator. "
            f"Available operators: `{'`, `'.join(__operators__)}`."
        ),
        required=True,
    ),
)


@click.group("operator")
def commands():
    pass


@commands.command()
@click_options(DEBUG_OPTION)
def list(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Prints the list of the registered operators."""
    suppress_traceback(debug)(cmd_list)(**kwargs)


@commands.command()
@click_options(DEBUG_OPTION + OPERATOR_NAME_OPTION)
def info(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Prints the detailed information about the particular operator."""
    suppress_traceback(debug)(cmd_info)(**kwargs)


@commands.command()
@click_options(DEBUG_OPTION + OPERATOR_NAME_OPTION + ADS_CONFIG_OPTION)
@click.option(
    "--output",
    help=f"The folder name to save the resulting specification templates.",
    required=False,
    default=None,
)
@click.option(
    "--overwrite",
    "-o",
    help="Overwrite result file if it already exists.",
    is_flag=True,
    default=False,
)
def init(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Generates starter YAML configs for the operator."""
    suppress_traceback(debug)(cmd_init)(**kwargs)


@commands.command()
@click_options(DEBUG_OPTION + OPERATOR_NAME_OPTION)
@click.option(
    "--gpu",
    "-g",
    help="Build a GPU-enabled Docker image.",
    is_flag=True,
    default=False,
    required=False,
)
@click.option(
    "--rebuild-base-image",
    "-r",
    help="Rebuild operator's base image. This option is useful when developing a new operator.",
    is_flag=True,
    default=False,
)
def build_image(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Builds a new image for the particular operator."""
    suppress_traceback(debug)(cmd_build_image)(**kwargs)


@commands.command()
@click_options(DEBUG_OPTION + OPERATOR_NAME_OPTION + ADS_CONFIG_OPTION)
@click.option(
    "--registry",
    "-r",
    help="Registry to publish to. By default the value will be taken from the ADS opctl config.",
    required=False,
    default=None,
)
def publish_image(debug, **kwargs):
    """Publishes an operator's image to the container registry."""
    suppress_traceback(debug)(cmd_publish_image)(**kwargs)


@commands.command(hidden=True)
@click_options(DEBUG_OPTION + OPERATOR_NAME_OPTION + ADS_CONFIG_OPTION)
@click.option(
    "--overwrite",
    "-o",
    help="Overwrite result file if it already exists.",
    is_flag=True,
    default=False,
)
@click.option(
    "--output",
    help="The folder to save the resulting specification template YAML.",
    required=False,
    default=None,
)
def create(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Creates new operator."""
    suppress_traceback(debug)(cmd_create)(**kwargs)


@commands.command()
@click_options(DEBUG_OPTION + OPERATOR_NAME_OPTION + ADS_CONFIG_OPTION)
@click.option(
    "--file", "-f", help="The path to resource YAML file.", required=True, default=None
)
@click.option(
    "--auth",
    "-a",
    help=(
        "The authentication method to leverage OCI resources. "
        "The default value will be taken from the ADS `config.ini` file."
    ),
    type=click.Choice(AuthType.values()),
    default=None,
)
def verify(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Verifies the operator config."""

    p = ConfigProcessor().step(ConfigMerger, **kwargs)

    with AuthContext(
        **{
            key: value
            for key, value in {
                "auth": kwargs["auth"],
                "oci_config_location": p.config["execution"]["oci_config"],
                "profile": p.config["execution"]["oci_profile"],
            }.items()
            if value
        }
    ) as auth:
        with fsspec.open(kwargs["file"], "r", **auth) as f:
            operator_spec = suppress_traceback(debug)(yaml.safe_load)(f.read())

    suppress_traceback(debug)(cmd_verify)(operator_spec, **kwargs)


@commands.command()
@click_options(DEBUG_OPTION + OPERATOR_NAME_OPTION + ADS_CONFIG_OPTION)
@click.option(
    "--conda-pack-folder",
    help=(
        "The destination folder to save the conda environment. "
        "By default will be used the path specified in the ADS config file generated "
        "with `ads opctl configure` command."
    ),
    required=False,
    default=None,
)
@click.option(
    "--overwrite",
    "-o",
    help="Overwrite conda environment if it already exists.",
    is_flag=True,
    default=False,
)
def build_conda(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Builds a new conda environment for the particular operator."""
    suppress_traceback(debug)(cmd_build_conda)(**kwargs)


@commands.command()
@click_options(DEBUG_OPTION + OPERATOR_NAME_OPTION + ADS_CONFIG_OPTION)
@click.option(
    "--conda-pack-folder",
    help=(
        "The source folder to search the conda environment. "
        "By default will be used the path specified in the ADS config file generated "
        "with `ads opctl configure` command."
    ),
    required=False,
    default=None,
)
@click.option(
    "--overwrite",
    "-o",
    help="Overwrite conda environment if it already exists.",
    is_flag=True,
    default=False,
)
def publish_conda(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Publishes an operator's conda environment to the Object Storage bucket."""
    suppress_traceback(debug)(cmd_publish_conda)(**kwargs)
