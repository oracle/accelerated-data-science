#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from typing import Any, Dict

import click
import fsspec
import yaml

import ads.opctl.conda.cli
import ads.opctl.distributed.cli
import ads.opctl.model.cli
import ads.opctl.operator.cli
import ads.opctl.spark.cli
from ads.common import auth as authutil
from ads.common.auth import AuthType
from ads.opctl.cmds import activate as activate_cmd
from ads.opctl.cmds import cancel as cancel_cmd
from ads.opctl.cmds import configure as configure_cmd
from ads.opctl.cmds import deactivate as deactivate_cmd
from ads.opctl.cmds import delete as delete_cmd
from ads.opctl.cmds import init as init_cmd
from ads.opctl.cmds import init_vscode as init_vscode_cmd
from ads.opctl.cmds import predict as predict_cmd
from ads.opctl.cmds import run as run_cmd
from ads.opctl.cmds import run_diagnostics as run_diagnostics_cmd
from ads.opctl.cmds import watch as watch_cmd
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.constants import (
    BACKEND_NAME,
    DEFAULT_MODEL_FOLDER,
    RESOURCE_TYPE,
    RUNTIME_TYPE,
)
from ads.opctl.decorator.common import with_auth
from ads.opctl.utils import build_image as build_image_cmd
from ads.opctl.utils import publish_image as publish_image_cmd
from ads.opctl.utils import suppress_traceback


@click.group("opctl")
@click.help_option("--help", "-h")
def commands():
    pass


@commands.command()
@click.help_option("--help", "-h")
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def configure(debug):
    """Sets up the initial configurations for the ADS OPCTL."""
    suppress_traceback(debug)(configure_cmd)()


@commands.command()
@click.argument("image-type", type=click.Choice(["job-local"]))
@click.help_option("--help", "-h")
@click.option(
    "--gpu",
    "-g",
    help="whether to build a gpu version of image",
    is_flag=True,
    default=False,
    required=False,
)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def build_image(image_type, gpu, debug):
    """Builds the local Data Science Jobs image."""
    suppress_traceback(debug)(build_image_cmd)(image_type, gpu)


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
    """Publishes image to the OCI Container Registry."""
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
        "--file", "-f", help="path to resource YAML file", required=False, default=None
    ),
    click.option(
        "--operator-name", "-n", help="operator name", required=False, default=None
    ),
    click.option(
        "--backend",
        "-b",
        type=click.Choice([backend.value for backend in BACKEND_NAME]),
        help="backend to run the operator",
        required=False,
        default=None,
    ),
    click.option(
        "--oci-config",
        help="oci config file",
        required=False,
        default=authutil.DEFAULT_LOCATION,
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
        type=click.Choice(["api_key", "resource_principal", "security_token"]),
        default=None,
    ),
]

_model_deployment_options = [
    click.option(
        "--wait-for-completion",
        help="either to wait for process to complete or not for model deployment",
        is_flag=True,
        required=False,
    ),
    click.option(
        "--max-wait-time",
        help="maximum wait time in seconds for progress to complete for model deployment",
        type=int,
        required=False,
        default=1200,
    ),
    click.option(
        "--poll-interval",
        help="poll interval in seconds for model deployment",
        type=int,
        required=False,
        default=10,
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
    "--ocid",
    required=False,
    default=None,
    help="create an OCI Data Science service run from existing ocid.",
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
@click.option("--tag", "-t", help="tag of image", required=False, default=None)
@click.option(
    "--registry", "-reg", help="registry to push to", required=False, default=None
)
@click.option(
    "--dockerfile",
    "-df",
    help="relative path to Dockerfile",
    required=False,
    default=None,
)
@with_auth
def run(file, debug, **kwargs):
    """
    Runs the operator with the given specification on the targeted backend.
    For the distributed backend, the operator is always run as a OCI Data Science job.
    """
    config = {}
    if file:
        with fsspec.open(file, "r", **authutil.default_signer()) as f:
            config = suppress_traceback(debug)(yaml.safe_load)(f.read())

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
@click.argument("ocid", nargs=1)
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default=None,
)
@click.option(
    "--oci-profile",
    help="oci profile",
    default=None,
)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def delete(**kwargs):
    """Deletes a data science service resource."""
    suppress_traceback(kwargs["debug"])(delete_cmd)(**kwargs)


@commands.command()
@click.argument("ocid", nargs=1)
@add_options(_model_deployment_options)
@click.option(
    "--conda-pack-folder",
    required=False,
    default=None,
    help="folder where conda packs are saved",
)
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default=None,
)
@click.option(
    "--oci-profile",
    help="oci profile",
    default=None,
)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def cancel(**kwargs):
    """Aborts the execution of the OCI resource run."""
    suppress_traceback(kwargs["debug"])(cancel_cmd)(**kwargs)


@commands.command()
@click.argument("ocid", nargs=1)
@click.option("--debug", "-d", help="Set debug mode", is_flag=True, default=False)
@click.option(
    "--log-type",
    help="the type of logging. Allowed value: `custom_log` and `service_log` for pipeline, `access` and `predict` for model deployment.",
    required=False,
    default=None,
)
@click.option(
    "--log-filter",
    help="expression for filtering the logs for model deployment.",
    required=False,
    default=None,
)
@click.option(
    "--interval",
    help="log interval in seconds",
    type=int,
    required=False,
    default=3,
)
@click.option(
    "--wait",
    help="time in seconds to keep updating the logs after the job run finished for job.",
    type=int,
    required=False,
    default=90,
)
@click.option(
    "--conda-pack-folder",
    required=False,
    default=None,
    help="folder where conda packs are saved",
)
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default=None,
)
@click.option(
    "--oci-profile",
    help="oci profile",
    default=None,
)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def watch(**kwargs):
    """
    Tails the logs form a job run, data flow run or pipeline run.
    Connects to the logging service that was configured with the JobRun, Application Run or Pipeline Run and streams the logs.
    """
    suppress_traceback(kwargs["debug"])(watch_cmd)(**kwargs)


@commands.command()
@click.argument("ocid", nargs=1)
@click.option("--debug", "-d", help="Set debug mode", is_flag=True, default=False)
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default=None,
)
@click.option(
    "--oci-profile",
    help="oci profile",
    default=None,
)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def activate(**kwargs):
    """
    Activates a data science service resource.
    """
    suppress_traceback(kwargs["debug"])(activate_cmd)(**kwargs)


@commands.command()
@click.argument("ocid", nargs=1)
@click.option("--debug", "-d", help="Set debug mode", is_flag=True, default=False)
@add_options(_model_deployment_options)
@click.option(
    "--conda-pack-folder",
    required=False,
    default=None,
    help="folder where conda packs are saved",
)
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default=None,
)
@click.option(
    "--oci-profile",
    help="oci profile",
    default=None,
)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def deactivate(**kwargs):
    """
    Deactivates a data science service resource.
    """
    suppress_traceback(kwargs["debug"])(deactivate_cmd)(**kwargs)


@commands.command()
@click.argument(
    "resource-type",
    type=click.Choice(RESOURCE_TYPE.values()),
    required=True,
)
@click.help_option("--help", "-h")
@click.option("--debug", "-d", help="Set debug mode", is_flag=True, default=False)
@click.option(
    "--runtime-type",
    type=click.Choice(RUNTIME_TYPE.values()),
    help="The runtime type",
    required=False,
)
@click.option(
    "--output",
    help=f"The filename to save the resulting specification template YAML",
    required=False,
    default=None,
)
@click.option(
    "--overwrite",
    "-o",
    help="Overwrite result file if it already exists",
    is_flag=True,
    default=False,
)
@click.option(
    "--ads-config",
    help="The folder where the ADS opctl config located",
    required=False,
    default=None,
)
def init(debug: bool, **kwargs: Dict[str, Any]) -> None:
    """Generates a starter specification template YAML for the Data Science resources."""
    suppress_traceback(debug)(init_cmd)(**kwargs)


@commands.command()
@click.option(
    "--ocid",
    nargs=1,
    required=False,
    help="This can be either a model id or model deployment id. When model id is passed, it conducts a local predict/test. This is designed for local dev purpose in order to test whether deployment will be successful locally. When you pass in `model_save_folder`, the model artifact will be downloaded and saved to a subdirectory of `model_save_folder` where model id is the name of subdirectory. Or you can pass in a model deployment id and this will invoke the remote endpoint and conduct a prediction on the server.",
)
@click.option(
    "--model-save-folder",
    nargs=1,
    required=False,
    default=DEFAULT_MODEL_FOLDER,
    help="Which location to store model artifact folders. Defaults to ~/.ads_ops/models. This is only used when model id is passed to `ocid` and a local predict is conducted.",
)
@click.option(
    "--conda-pack-folder",
    nargs=1,
    required=False,
    help="Which location to store the conda pack locally. Defaults to ~/.ads_ops/conda. This is only used when model id is passed to `ocid` and a local predict is conducted.",
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
    "--artifact-directory",
    nargs=1,
    required=False,
    default=None,
    help="The artifact directory where stores your models, score.py and etc. This is used when you have a model artifact locally and have not saved it to the model catalog yet. In this case, you dont need to pass in model ",
)
@click.option(
    "--payload",
    nargs=1,
    help="The payload sent to the model for prediction. For example, --payload '[[-1.68671955,-0.27358368,0.82731396,-0.14530245,0.80733585]]'.",
)
@click.option(
    "--conda-slug",
    nargs=1,
    required=False,
    help="The conda slug used to load the model and conduct the prediction. This is only used when model id is passed to `ocid` and a local predict is conducted. It should match the inference conda env specified in the runtime.yaml file which is the conda pack being used when conducting real model deployment.",
)
@click.option(
    "--conda-path",
    nargs=1,
    required=False,
    help="The conda path used to load the model and conduct the prediction. This is only used when model id is passed to `ocid` and a local predict is conducted. It should match the inference conda env specified in the runtime.yaml file which is the conda pack being used when conducting real model deployment.",
)
@click.option(
    "--model-version",
    nargs=1,
    required=False,
    help="When the `inference_server='triton'`, the version of the model to invoke. This can only be used when model deployment id is passed in. For the other cases, it will be ignored.",
)
@click.option(
    "--model-name",
    nargs=1,
    required=False,
    help="When the `inference_server='triton'`, the name of the model to invoke. This can only be used when model deployment id is passed in. For the other cases, it will be ignored.",
)
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default=None,
)
@click.option(
    "--oci-profile",
    help="oci profile",
    default=None,
)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def predict(**kwargs):
    """
    Make prediction using the model with the payload.
    """
    suppress_traceback(kwargs["debug"])(predict_cmd)(**kwargs)


commands.add_command(ads.opctl.conda.cli.commands)
commands.add_command(ads.opctl.model.cli.commands)
commands.add_command(ads.opctl.spark.cli.commands)
commands.add_command(ads.opctl.distributed.cli.commands)
