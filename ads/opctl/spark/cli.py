#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import click

from ads.common.auth import AuthType
from ads.opctl.utils import suppress_traceback
from ads.opctl.spark.cmds import core_site as core_site_cmd


@click.group("spark")
@click.help_option("--help", "-h")
def commands():
    "The CLI to assist in the management of the Spark workloads."
    pass


@commands.command()
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default=None,
)
@click.option("--oci-config", help="oci config file path", default=None, required=False)
@click.option(
    "--overwrite", "-o", help="overwrite core-site.xml", default=False, is_flag=True
)
@click.option("--oci-profile", help="oci profile to use", default=None, required=False)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
@click.help_option("--help", "-h")
def core_site(auth, oci_config, overwrite, oci_profile, debug):
    suppress_traceback(debug)(core_site_cmd)(
        auth,
        oci_config,
        overwrite,
        oci_profile,
    )
