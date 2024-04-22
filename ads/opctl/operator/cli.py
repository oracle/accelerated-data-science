#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict

import click
import fsspec
import yaml
import logging
from ads.opctl.operator.common.utils import default_signer
from ads.common.auth import AuthType
from ads.common.object_storage_details import ObjectStorageDetails
from ads.opctl.constants import BACKEND_NAME, RUNTIME_TYPE
from ads.opctl.decorator.common import click_options, with_auth, with_click_unknown_args
from ads.opctl.utils import suppress_traceback
from ads.opctl import logger

from .__init__ import __operators__
from .cmd import run as cmd_run
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
            "The folder where the ADS `config.ini` located. "
            "The default location is: `~/.ads_ops` folder. "
            "Check the `ads opctl configure --help` command to get details about the `config.ini`."
        ),
        required=False,
        default=None,
    ),
)

OPERATOR_TYPE_OPTION = (
    click.option(
        "--type",
        "-t",
        help=(
            "The type of the operator. "
            f"Available service operators: `{'`, `'.join(__operators__)}`."
        ),
        required=True,
    ),
)

AUTH_TYPE_OPTION = (
    click.option(
        "--auth",
        "-a",
        help=(
            "The authentication method to leverage OCI resources. "
            "The default value will be taken from the ADS `config.ini` file. "
            "Check the `ads opctl configure --help` command to get details about the `config.ini`."
        ),
        type=click.Choice(AuthType.values()),
        default=None,
    ),
    click.option(
        "--oci-profile",
        help=(
            "OCI profile name to use for authentication. "
            "By default will be used the value specified in the ADS config file. "
            "Check the `ads opctl configure --help` command to get details about the `config.ini`. "
        ),
        required=False,
        default=None,
    ),
)


@click.group("operator")
def commands():
    "The CLI to assist in the management of the ADS operators."
    pass


@commands.command()
@click_options(DEBUG_OPTION)
def list(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Prints the list of the registered operators."""
    suppress_traceback(debug)(cmd_list)(**kwargs)


@commands.command()
@click_options(
    DEBUG_OPTION + OPERATOR_TYPE_OPTION + ADS_CONFIG_OPTION + AUTH_TYPE_OPTION
)
@with_auth
def info(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Prints the detailed information about the particular operator."""
    suppress_traceback(debug)(cmd_info)(**kwargs)


@commands.command()
@click_options(
    DEBUG_OPTION + OPERATOR_TYPE_OPTION + ADS_CONFIG_OPTION + AUTH_TYPE_OPTION
)
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
@click.option(
    "--merge-config",
    "-m",
    help=(
        "Merge the operator's configuration with various backend configurations, "
        "resulting in multiple operator configurations, each paired with a distinct backend. "
        "By default, the operator's configuration will remain distinct from the backend configuration."
    ),
    is_flag=True,
    default=False,
)
@with_auth
def init(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Generates starter YAML configs for the operator."""
    suppress_traceback(debug)(cmd_init)(**kwargs)


@commands.command()
@click_options(DEBUG_OPTION + OPERATOR_TYPE_OPTION)
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
@with_auth
def build_image(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Creates a new image for the specified operator."""
    suppress_traceback(debug)(cmd_build_image)(**kwargs)


@commands.command()
@click_options(DEBUG_OPTION + OPERATOR_TYPE_OPTION + ADS_CONFIG_OPTION)
@click.option(
    "--registry",
    "-r",
    help="Registry to publish to. By default the value will be taken from the ADS opctl config.",
    required=False,
    default=None,
)
@with_auth
def publish_image(debug, **kwargs):
    """Publishes an operator's image to the container registry."""
    suppress_traceback(debug)(cmd_publish_image)(**kwargs)


@commands.command(hidden=True)
@click_options(DEBUG_OPTION + OPERATOR_TYPE_OPTION + ADS_CONFIG_OPTION)
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
@with_auth
def create(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Creates new operator."""
    suppress_traceback(debug)(cmd_create)(**kwargs)


@commands.command()
@click_options(DEBUG_OPTION + ADS_CONFIG_OPTION + AUTH_TYPE_OPTION)
@click.option(
    "--file", "-f", help="The path to resource YAML file.", required=True, default=None
)
@with_auth
def verify(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Verifies the operator config."""

    with fsspec.open(
        kwargs["file"],
        "r",
        **(
            default_signer() if ObjectStorageDetails.is_oci_path(kwargs["file"]) else {}
        ),
    ) as f:
        operator_spec = suppress_traceback(debug)(yaml.safe_load)(f.read())

    suppress_traceback(debug)(cmd_verify)(operator_spec, **kwargs)


@commands.command()
@click_options(DEBUG_OPTION + OPERATOR_TYPE_OPTION + ADS_CONFIG_OPTION)
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
@with_auth
def build_conda(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Creates a new conda environment for the specified operator."""
    suppress_traceback(debug)(cmd_build_conda)(**kwargs)


@commands.command()
@click_options(
    DEBUG_OPTION + OPERATOR_TYPE_OPTION + ADS_CONFIG_OPTION + AUTH_TYPE_OPTION
)
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
@with_auth
def publish_conda(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Publishes an operator's conda environment to the Object Storage bucket."""
    suppress_traceback(debug)(cmd_publish_conda)(**kwargs)


@commands.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click_options(DEBUG_OPTION + ADS_CONFIG_OPTION + AUTH_TYPE_OPTION)
@click.option(
    "--file",
    "-f",
    help="The path to the operator's specification YAML file.",
    required=True,
    default=None,
)
@click.option(
    "--backend",
    "-b",
    help=(
        "Backend name or the path to the operator's backend config YAML file. "
        f"\n\nExample 1:\n\n`ads operator run -f operator.yaml -b {BACKEND_NAME.LOCAL.value}`\n\n"
        "Supported backend names: "
        f"{(BACKEND_NAME.JOB.value,BACKEND_NAME.JOB.value + '.' + RUNTIME_TYPE.CONTAINER.value,BACKEND_NAME.DATAFLOW.value,BACKEND_NAME.LOCAL.value,BACKEND_NAME.LOCAL.value + '.'+ RUNTIME_TYPE.CONTAINER.value,)}. "
        "However some operators may support only a subset of these backends."
        "\n\nExample 2:\n\n`ads operator run -f operator.yaml -b backend.yaml`\n\n"
        "Use the `ads operator init --help` command to generate the operator's specification "
        "and all required backend configs. Generating configs is optional and fully automated. "
    ),
    required=False,
    default=None,
)
@click.option(
    "--dry-run",
    "-r",
    default=False,
    is_flag=True,
    help="During dry run, the actual operation is not performed, only the steps are enumerated.",
)
@click.pass_context
@with_click_unknown_args
@with_auth
def run(ctx: click.core.Context, debug: bool = False, **kwargs: Dict[str, Any]) -> None:
    """
    Runs the operator with the given specification on the targeted backend.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)
    operator_spec = {}
    backend = kwargs.pop("backend")

    with fsspec.open(
        kwargs["file"],
        "r",
        **(
            default_signer() if ObjectStorageDetails.is_oci_path(kwargs["file"]) else {}
        ),
    ) as f:
        operator_spec = suppress_traceback(debug)(yaml.safe_load)(f.read())

    if backend and backend.lower().endswith((".yaml", ".yml")):
        with fsspec.open(
            backend,
            "r",
            **(default_signer() if ObjectStorageDetails.is_oci_path(backend) else {}),
        ) as f:
            backend = suppress_traceback(debug)(yaml.safe_load)(f.read())

    suppress_traceback(debug)(cmd_run)(config=operator_spec, backend=backend, **kwargs)
