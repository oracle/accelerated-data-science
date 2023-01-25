#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import click
from ads.common.auth import AuthType, OCIAuthContext
from ads.opctl.utils import suppress_traceback
from ads.pipeline.ads_pipeline import Pipeline
from ads.pipeline.ads_pipeline_run import LogType, PipelineRun


@click.group("pipeline")
@click.help_option("--help", "-h")
def commands():
    pass


@commands.command()
@click.help_option("--help", "-h")
@click.option("--file", "-f", help="Path to YAML file", required=True)
@click.option(
    "--auth",
    "-a",
    help="Authentication method",
    type=click.Choice(AuthType.values()),
    default=AuthType.API_KEY,
)
@click.option(
    "--oci-profile", "-p", help="OCI config profile", required=False, default="DEFAULT"
)
@click.option("--debug", "-d", help="Set debug mode", is_flag=True, default=False)
def run(file, auth, oci_profile, debug):
    suppress_traceback(debug)(_run)(file, auth, oci_profile)


def _run(file, auth, oci_profile):
    if auth == AuthType.RESOURCE_PRINCIPAL:
        oci_profile = None
    with OCIAuthContext(profile=oci_profile):
        pipeline = Pipeline.from_yaml(uri=file)
        pipeline.create()
        print(pipeline.id)
        pipeline_run = pipeline.run()
        print(pipeline_run.id)


@commands.command()
@click.help_option("--help", "-h")
@click.argument("ocid")
@click.option(
    "--auth",
    "-a",
    help="Authentication method",
    type=click.Choice(AuthType.values()),
    default=AuthType.API_KEY,
)
@click.option(
    "--oci-profile", "-p", help="OCI config profile", required=False, default="DEFAULT"
)
@click.option("--debug", "-d", help="Set debug mode", is_flag=True, default=False)
@click.option(
    "--delete-related-job-runs",
    "-j",
    help="Whether to delete related Job runs or not.",
    is_flag=True,
    default=True,
)
def delete(ocid, auth, oci_profile, debug, delete_related_job_runs):
    suppress_traceback(debug)(_delete)(ocid, auth, oci_profile, delete_related_job_runs)


def _delete(ocid, auth, oci_profile, delete_related_job_runs):
    if auth == AuthType.RESOURCE_PRINCIPAL:
        oci_profile = None
    with OCIAuthContext(profile=oci_profile):
        run = PipelineRun.from_ocid(ocid)
        run.delete(delete_related_job_runs=delete_related_job_runs)


@commands.command()
@click.help_option("--help", "-h")
@click.argument("ocid")
@click.option(
    "--log-type",
    "-l",
    help="Log type",
    type=click.Choice(LogType.values()),
    default=LogType.CUSTOM_LOG,
)
@click.option(
    "--steps",
    "-s",
    help="Pipleine steps to monitor",
    default=None,
)
@click.option(
    "--auth",
    "-a",
    help="Authentication method",
    type=click.Choice(AuthType.values()),
    default=AuthType.API_KEY,
)
@click.option(
    "--oci-profile", "-p", help="OCI config profile", required=False, default="DEFAULT"
)
@click.option("--debug", "-d", help="Set debug mode", is_flag=True, default=False)
def watch(ocid, log_type, steps, auth, oci_profile, debug):
    suppress_traceback(debug)(_watch)(ocid, log_type, steps, auth, oci_profile)


def _watch(ocid, log_type, steps, auth, oci_profile):
    if auth == AuthType.RESOURCE_PRINCIPAL:
        oci_profile = None
    with OCIAuthContext(profile=oci_profile):
        run = PipelineRun.from_ocid(ocid)
        run.watch(steps=steps, log_type=log_type)
