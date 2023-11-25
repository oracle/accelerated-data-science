#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import click

from ads.common.auth import AuthType
from ads.opctl.conda.cmds import create as create_cmd
from ads.opctl.conda.cmds import install as install_cmd
from ads.opctl.conda.cmds import publish as publish_cmd
from ads.opctl.constants import DEFAULT_ADS_CONFIG_FOLDER
from ads.opctl.utils import suppress_traceback


@click.group("conda")
@click.help_option("--help", "-h")
def commands():
    "The CLI to assist in the management of conda environments."
    pass


@commands.command()
@click.option("--name", "-n", help="name for the conda pack", required=False)
@click.option(
    "--version", "-v", help="version of the pack", required=False, default="1", type=str
)
@click.option(
    "--environment-file", "-f", help="path to environment file", required=True
)
@click.option(
    "--conda-pack-folder",
    help="folder to save conda pack",
    required=False,
    default=None,
)
@click.option(
    "--ads-config",
    help="folder that saves ads opctl config",
    required=False,
    default=DEFAULT_ADS_CONFIG_FOLDER,
)
@click.option(
    "--gpu",
    "-g",
    help="whether to create conda pack with gpu image",
    is_flag=True,
    default=False,
    required=False,
)
@click.option(
    "--overwrite",
    "-o",
    help="whether to overwrite downloaded pack",
    is_flag=True,
    default=False,
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Set logging to debug for more information",
)
@click.help_option("--help", "-h")
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def create(**kwargs):
    suppress_traceback(kwargs["debug"])(create_cmd)(**kwargs)


@commands.command()
@click.option(
    "--conda-uri", "-u", help="uri to conda pack", required=False, default=None
)
@click.option(
    "--slug", "-s", help="slug for the conda pack", required=False, default=None
)
@click.option(
    "--conda-pack-os-prefix",
    "-p",
    help="object storage prefix to download conda packs",
    required=False,
    default=None,
)
@click.option(
    "--conda-pack-folder",
    help="folder to save conda pack",
    required=False,
    default=None,
)
@click.option(
    "--oci-config",
    help="oci config file",
    required=False,
    default=None,
)
@click.option("--oci-profile", help="oci config profile", required=False, default=None)
@click.option(
    "--ads-config",
    help="folder that saves ads opctl config",
    required=False,
    default=DEFAULT_ADS_CONFIG_FOLDER,
)
@click.option(
    "--overwrite",
    "-o",
    help="whether to overwrite downloaded pack",
    is_flag=True,
    default=False,
)
@click.help_option("--help", "-h")
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default=None,
)
def install(**kwargs):
    suppress_traceback(kwargs["debug"])(install_cmd)(**kwargs)


@commands.command()
@click.option("--slug", "-s", help="slug for the conda pack", required=False)
@click.option("--name", "-n", help="name for the conda pack", required=False)
@click.option(
    "--version",
    "-v",
    help="version of the pack",
    required=False,
    default="1",
    type=str,
)
@click.option(
    "--environment-file",
    "-f",
    help="conda environment yaml path",
    required=False,
    default=None,
)
@click.option(
    "--conda-pack-os-prefix",
    "-p",
    help="object storage prefix to save published conda packs, in the form oci://<bucket>@<namespace>/path/",
    required=False,
    default=None,
)
@click.option(
    "--conda-pack-folder",
    help="folder to save conda pack",
    required=False,
    default=None,
)
@click.option(
    "--gpu",
    "-g",
    help="whether to create conda pack with gpu image",
    is_flag=True,
    default=False,
    required=False,
)
@click.option(
    "--oci-config",
    help="oci config file",
    required=False,
    default=None,
)
@click.option("--oci-profile", help="oci config profile", required=False, default=None)
@click.option(
    "--ads-config",
    help="folder that saves ads opctl config",
    required=False,
    default=DEFAULT_ADS_CONFIG_FOLDER,
)
@click.option(
    "--overwrite",
    "-o",
    help="whether to overwrite downloaded pack",
    is_flag=True,
    default=False,
)
@click.help_option("--help", "-h")
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default=None,
)
def publish(**kwargs):
    suppress_traceback(kwargs["debug"])(publish_cmd)(**kwargs)
