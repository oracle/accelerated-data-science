#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os

import click
import yaml

from ads.opctl.cmds import cancel as cancel_cmd
from ads.opctl.cmds import configure as configure_cmd
from ads.opctl.cmds import delete as delete_cmd
from ads.opctl.cmds import init_vscode as init_vscode_cmd
from ads.opctl.cmds import run as run_cmd
from ads.opctl.cmds import watch as watch_cmd
from ads.opctl.utils import _list_ads_operators
from ads.opctl.utils import build_image as build_image_cmd
from ads.opctl.utils import publish_image as publish_image_cmd
from ads.opctl.utils import suppress_traceback
from ads.opctl.config.merger import ConfigMerger

import ads.opctl.conda.cli


@click.group("opctl")
@click.help_option("--help", "-h")
def commands():
    pass


@commands.command()
@click.help_option("--help", "-h")
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def configure(debug):
    suppress_traceback(debug)(configure_cmd)()


@commands.command()
@click.argument(
    "image-type", type=click.Choice(["job-local", "ads-ops-base", "ads-ops-custom"])
)
@click.help_option("--help", "-h")
@click.option(
    "--gpu",
    "-g",
    help="whether to build a gpu version of image",
    is_flag=True,
    default=False,
    required=False,
)
@click.option(
    "--source-folder",
    "-s",
    help="when building custom operator image, source folder of the custom operator",
    default=None,
    required=False,
)
@click.option(
    "--image",
    "-i",
    help="image name, used when building custom image",
    default=None,
    required=False,
)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def build_image(image_type, gpu, source_folder, image, debug):
    suppress_traceback(debug)(build_image_cmd)(image_type, gpu, source_folder, image)


@commands.command()
@click.argument("image")
@click.option(
    "--registry", "-r", help="registry to push to", required=False, default=None
)
@click.option(
    "--ads-config",
    help="folder that saves ads opctl config",
    required=False,
    default=None,
)
@click.help_option("--help", "-h")
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def publish_image(**kwargs):
    debug = kwargs.pop("debug")
    if kwargs.get("registry", None):
        registry = kwargs["registry"]
    else:
        p = ConfigMerger({}).process(**kwargs)
        registry = p.config.get("infrastructure", {}).get("docker_registry", None)
    suppress_traceback(debug)(publish_image_cmd)(kwargs["image"], registry)


@commands.command()
@click.option(
    "--source-folder",
    "-s",
    help="source folder for scripts",
    required=False,
    default=None,
)
@click.option(
    "--image",
    "-i",
    help="docker image name",
    required=False,
    default=None,
)
@click.option(
    "--oci-config",
    help="oci config file",
    required=False,
    default=None,
)
@click.option(
    "--conda-pack-folder",
    help="folder where conda packs are saved",
    required=False,
    default=None,
)
@click.option(
    "--ads-config",
    help="folder that saves ads opctl config",
    required=False,
    default=None,
)
@click.option(
    "--gpu",
    "-g",
    help="whether to run against GPU image",
    is_flag=True,
    default=False,
    required=False,
)
@click.option(
    "--env-var",
    help="environment variables to pass in",
    multiple=True,
    required=False,
    default=None,
)
@click.option(
    "--volume",
    "-v",
    help="volumes to mount to image",
    multiple=True,
    required=False,
    default=None,
)
@click.help_option("--help", "-h")
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def init_vscode(**kwargs):
    # kwargs["use_conda"] = True
    suppress_traceback(kwargs["debug"])(init_vscode_cmd)(**kwargs)


_options = [
    click.option(
        "--file", "-f", help="path to operator YAML file", required=False, default=None
    ),
    click.option(
        "--operator-name", "-n", help="operator name", required=False, default=None
    ),
    click.option(
        "--backend",
        "-b",
        type=click.Choice(["local", "job"]),
        help="backend to run the operator",
        required=False,
        default=None,
    ),
    click.option(
        "--oci-config",
        help="oci config file",
        required=False,
        default=None,
    ),
    click.option(
        "--oci-profile", "-op", help="oci config profile", required=False, default=None
    ),
    click.option(
        "--conf-file",
        help="path to conf file",
        required=False,
        default=None,
    ),
    click.option(
        "--conf-profile",
        help="conf profile",
        required=False,
        default=None,
    ),
    click.option(
        "--conda-pack-folder",
        help="folder where conda packs are saved",
        required=False,
        default=None,
    ),
    click.option(
        "--ads-config",
        help="folder that saves ads opctl config",
        required=False,
        default=None,
    ),
    click.option("--job-id", "-j", help="ML Job ocid", required=False, default=None),
    click.option(
        "--run-id", "-r", help="ML Job run ocid", required=False, default=None
    ),
    click.help_option("--help", "-h"),
    click.option(
        "--job-name", help="display name of job", required=False, default=None
    ),
    click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False),
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


@commands.command()
@add_options(_options)
@click.option("--image", "-i", help="image name", required=False, default=None)
@click.option("--conda-slug", help="slug name", required=False, default=None)
@click.option(
    "--conda-uri",
    help="conda pack uri in object storage",
    required=False,
    default=None,
)
@click.option(
    "--use-conda",
    help="whether to use conda pack to run a job",
    is_flag=True,
    required=False,
    default=None,
)
@click.option(
    "--source-folder",
    "-s",
    help="source scripts folder that will be mounted during a local run",
    required=False,
    default=None,
)
@click.option(
    "--env-var",
    help="environment variables to pass in",
    multiple=True,
    required=False,
    default=None,
)
@click.option(
    "--entrypoint",
    "-e",
    help="entrypoint, a script in case of running with conda pack or entrypoint in case of running with an image",
    required=False,
    default=None,
)
@click.option(
    "--command",
    "-c",
    help="docker command, used when running with an image",
    required=False,
    default=None,
)
@click.option(
    "--cmd-args",
    help="command line args, used when running with conda pack",
    required=False,
    default=None,
)
@click.option(
    "--gpu",
    "-g",
    help="whether to run against GPU image",
    is_flag=True,
    default=None,
    required=False,
)
@click.option(
    "--volume",
    "-v",
    help="volumes to mount to image, used only for local testing at the moment",
    multiple=True,
    required=False,
    default=None,
)
@click.option("--namespace", help="OCI namespace", default=None, required=False)
@click.option(
    "--publish-image",
    "-p",
    help="whether to rebuild and publish image before submitting a ML Job run",
    is_flag=True,
    default=None,
)
def run(file, **kwargs):
    debug = kwargs["debug"]
    if file:
        if os.path.exists(file):
            with open(file, "r") as f:
                config = suppress_traceback(debug)(yaml.safe_load)(f.read())
        else:
            raise FileNotFoundError(f"{file} is not found")
    else:
        config = {}

    suppress_traceback(debug)(run_cmd)(config, **kwargs)


@commands.command()
@click.argument("ocid", nargs=1)
@add_options(_options)
def delete(**kwargs):
    suppress_traceback(kwargs["debug"])(delete_cmd)(**kwargs)


@commands.command()
@click.argument("ocid", nargs=1)
@add_options(_options)
def cancel(**kwargs):
    suppress_traceback(kwargs["debug"])(cancel_cmd)(**kwargs)


@commands.command()
@click.argument("ocid", nargs=1)
@add_options(_options)
def watch(**kwargs):
    suppress_traceback(kwargs["debug"])(watch_cmd)(**kwargs)


commands.add_command(ads.opctl.conda.cli.commands)


# @commands.command()
# @click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
# def list(debug):
#     print(json.dumps(suppress_traceback(debug)(_list_ads_operators)(), indent=2))
