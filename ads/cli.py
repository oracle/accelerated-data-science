#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import click
import ads.opctl.cli
import os
import json

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


if __name__ == "__main__":
    cli()
