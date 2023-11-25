#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import traceback
import sys

from ads.common import logger

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
def cli():
    pass


cli.add_command(ads.opctl.cli.commands)
cli.add_command(ads.jobs.cli.commands)
cli.add_command(ads.pipeline.cli.commands)
cli.add_command(ads.opctl.operator.cli.commands)


if __name__ == "__main__":
    cli()
