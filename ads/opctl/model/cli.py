#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import click
from ads.common.auth import AuthType
from ads.opctl.utils import suppress_traceback
from ads.opctl.model.cmds import download_model as download_model_cmd
from ads.opctl.constants import DEFAULT_MODEL_FOLDER


@click.group("model")
@click.help_option("--help", "-h")
def commands():
    pass


@commands.command()
@click.argument("ocid", required=True)
@click.option(
    "--model-save-folder",
    "-mf",
    nargs=1,
    required=False,
    default=DEFAULT_MODEL_FOLDER,
    help="Which location to store model artifact folders. Defaults to ~/.ads_ops/models. This is only used when model id is passed to `ocid` and a local predict is conducted.",
)
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default=None,
)
@click.option(
    "--bucket-uri",
    nargs=1,
    required=False,
    help="The OCI Object Storage URI where model artifacts will be copied to. The `bucket_uri` is only necessary for uploading large artifacts which size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`. This is only used when the model id is passed.",
)
@click.option(
    "--region",
    nargs=1,
    required=False,
    help="The destination Object Storage bucket region. By default the value will be extracted from the `OCI_REGION_METADATA` environment variables. This is only used when the model id is passed.",
)
@click.option(
    "--timeout",
    nargs=1,
    required=False,
    help="The connection timeout in seconds for the client. This is only used when the model id is passed.",
)
@click.option(
    "--force-overwrite",
    "-f",
    help="Overwrite existing model.",
    is_flag=True,
    default=False,
)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def download(**kwargs):
    suppress_traceback(kwargs["debug"])(download_model_cmd)(**kwargs)
