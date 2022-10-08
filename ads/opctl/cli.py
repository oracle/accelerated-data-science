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
from ads.opctl.cmds import run_diagnostics as run_diagnostics_cmd
from ads.opctl.cmds import watch as watch_cmd
from ads.opctl.cmds import init_operator as init_operator_cmd
from ads.opctl.utils import build_image as build_image_cmd
from ads.opctl.utils import publish_image as publish_image_cmd
from ads.opctl.utils import suppress_traceback
from ads.opctl.config.merger import ConfigMerger

import ads.opctl.conda.cli
import ads.opctl.spark.cli
import ads.opctl.distributed.cli


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
@click.argument("image-type", type=click.Choice(["job-local", "ads-ops-base"]))
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
    """
    Generates devcontainer.json with docker details to setup a development environment for OCI Data Science Jobs
    """
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
        type=click.Choice(["local", "job", "dataflow"]),
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
        "--oci-profile", help="oci config profile", required=False, default=None
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
    click.help_option("--help", "-h"),
    click.option(
        "--job-name", help="display name of job", required=False, default=None
    ),
    click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False),
    click.option(
        "--auth",
        "-a",
        help="authentication method",
        type=click.Choice(["api_key", "resource_principal"]),
        default=None,
    ),
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
    "--use-conda",
    help="whether to use conda pack to run an operator",
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
@click.option(
    "--overwrite",
    "-o",
    help="overwrite object storage files when uploading local files during dataflow run",
    is_flag=True,
    default=False,
)
@click.option("--archive", help="path to archive zip for dataflow run", default=None)
@click.option("--exclude-tag", multiple=True, default=None)
@click.option(
    "--dry-run",
    "-r",
    default=False,
    is_flag=True,
    help="During dry run, the actual operation is not performed, only the steps are enumerated.",
)
@click.option(
    "--nobuild",
    "-nobuild",
    default=False,
    is_flag=True,
    help="skip re-building the docker image",
)
@click.option(
    "--auto_increment",
    "-i",
    default=False,
    is_flag=True,
    help="Increments tag of the image while rebuilding",
)
@click.option(
    "--nopush",
    "-nopush",
    default=False,
    is_flag=True,
    help="Image is not pushed to OCIR",
)
@click.option(
    "--tag",
      "-t",
      help="tag of image",
      required=False,
      default=None
)
@click.option(
    "--registry",
      "-reg",
      help="registry to push to",
      required=False,
      default=None
)
@click.option(
    "--dockerfile",
      "-df",
      help="relative path to Dockerfile",
      required=False,
      default=None
)
def run(file, **kwargs):
    """
    Runs the workload on the targeted backend. When run `distributed` yaml spec, the backend is always OCI Data Science Jobs
    """
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
@add_options(_options)
@click.option(
    "--dry-run",
    "-r",
    default=False,
    is_flag=True,
    help="During dry run, the actual operation is not performed, only the steps are enumerated.",
)
@click.option(
    "--output", help="filename to save the report", default="diagnostic_report.html"
)
def check(file, **kwargs):
    """
    Run diagnostics to check if your setup meets all the requirements to run the job
    """
    debug = kwargs["debug"]
    if file:
        if os.path.exists(file):
            with open(file, "r") as f:
                config = suppress_traceback(debug)(yaml.safe_load)(f.read())
        else:
            raise FileNotFoundError(f"{file} is not found")
    else:
        config = {}
    suppress_traceback(debug)(run_diagnostics_cmd)(config, **kwargs)


@commands.command()
@click.argument("operator_slug", nargs=1)
@click.option(
    "--folder_path",
    "-fp",
    help="the name of the folder wherein to put the operator code",
    multiple=True,
    required=False,
    default=None,
)
@add_options(_options)
def init_operator(**kwargs):
    suppress_traceback(kwargs["debug"])(init_operator_cmd)(**kwargs)


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
    """
    ``tail`` logs form a job run or dataflow run. Connects to the logging service that was configured with the JobRun or the Application Run and streams the logs.
    """
    suppress_traceback(kwargs["debug"])(watch_cmd)(**kwargs)


commands.add_command(ads.opctl.conda.cli.commands)
commands.add_command(ads.opctl.spark.cli.commands)
commands.add_command(ads.opctl.distributed.cli.commands)


# @commands.command()
# @click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
# def list(debug):
#     print(json.dumps(suppress_traceback(debug)(_list_ads_operators)(), indent=2))
