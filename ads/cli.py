#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import traceback
import sys

import fire
from ads.common import logger
from ads.aqua.cli import AquaCommand

try:
    import click
    import ads.opctl.cli
    import ads.jobs.cli
    import ads.pipeline.cli
    import ads.opctl.operator.cli
except Exception as ex:
    print(
        "Please run `pip install oracle-ads[opctl]` to install "
        "the required dependencies for ADS CLI. \n"
        f"{str(ex)}"
    )
    logger.debug(ex)
    logger.debug(traceback.format_exc())
    exit()

# https://packaging.python.org/en/latest/guides/single-sourcing-package-version/#single-sourcing-the-package-version
if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

ADS_VERSION = metadata.version("oracle_ads")


@click.group()
@click.version_option(version=ADS_VERSION, prog_name="ads")
@click.help_option("--help", "-h")
def click_cli():
    pass


@click.command
def aqua_cli():
    """CLI for AQUA."""
    # This is a dummy entry for click.
    # The `ads aqua` commands are handled by AquaCommand


click_cli.add_command(ads.opctl.cli.commands)
click_cli.add_command(ads.jobs.cli.commands)
click_cli.add_command(ads.pipeline.cli.commands)
click_cli.add_command(ads.opctl.operator.cli.commands)
click_cli.add_command(aqua_cli, name="aqua")


# fix for fire issue with --help
# https://github.com/google/python-fire/issues/258
def _SeparateFlagArgs(args):
    try:
        index = args.index("--help")
        args = args[:index]
        return args, ["--help"]
    except ValueError:
        return args, []


fire.core.parser.SeparateFlagArgs = _SeparateFlagArgs


def cli():
    if len(sys.argv) > 1 and sys.argv[1] == "aqua":
        fire.Fire(AquaCommand, command=sys.argv[2:], name="ads aqua")
    else:
        click_cli()


if __name__ == "__main__":
    cli()
