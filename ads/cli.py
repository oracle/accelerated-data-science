#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from ads.common import logger
import traceback

try:
    import click
    import ads.opctl.cli
    import ads.jobs.cli
    import ads.pipeline.cli
    import os
    import json
except Exception as ex:
    print(
        "Please run `pip install oracle-ads[opctl]` to install "
        "the required dependencies for ADS CLI."
    )
    logger.debug(ex)
    logger.debug(traceback.format_exc())
    exit()


with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ads_version.json")
) as version_file:
    ADS_VERSION = json.load(version_file)["version"]


@click.group()
@click.version_option(version=ADS_VERSION, prog_name="ads")
@click.help_option("--help", "-h")
def cli():
    pass


cli.add_command(ads.opctl.cli.commands)
cli.add_command(ads.jobs.cli.commands)
cli.add_command(ads.pipeline.cli.commands)


if __name__ == "__main__":
    cli()
