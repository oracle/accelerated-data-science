#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import click

import ads
from ads.common.auth import AuthType, AuthContext
from ads.jobs import Job, DataFlowRun, DataScienceJobRun
from ads.opctl.utils import suppress_traceback


@click.group("jobs")
@click.help_option("--help", "-h")
def commands():
    pass


@commands.command()
@click.help_option("--help", "-h")
@click.option("--file", "-f", help="path to YAML file", required=True)
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default="api_key",
)
@click.option(
    "--oci-profile", help="oci config profile", required=False, default="DEFAULT"
)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def run(file, auth, oci_profile, debug):
    def _run(file, auth, oci_profile):
        with AuthContext():
            ads.set_auth(auth=auth, profile=oci_profile)
            job = Job.from_yaml(uri=file)
            job.create()
            print(job.id)
            job_run = job.run()
            print(job_run.id)

    suppress_traceback(debug)(_run)(file, auth, oci_profile)


@commands.command()
@click.help_option("--help", "-h")
@click.argument("ocid")
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default="api_key",
)
@click.option(
    "--oci-profile", help="oci config profile", required=False, default="DEFAULT"
)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def watch(ocid, auth, oci_profile, debug):
    def _watch(ocid, auth, oci_profile):
        with AuthContext():
            ads.set_auth(auth=auth, profile=oci_profile)
            if "datasciencejobrun" in ocid:
                run = DataScienceJobRun.from_ocid(ocid)
            elif "dataflowrun" in ocid:
                run = DataFlowRun.from_ocid(ocid)
            else:
                raise ValueError("Run OCID not recognized.")
            run.watch()

    suppress_traceback(debug)(_watch)(ocid, auth, oci_profile)


@commands.command()
@click.help_option("--help", "-h")
@click.argument("ocid")
@click.option(
    "--auth",
    "-a",
    help="authentication method",
    type=click.Choice(AuthType.values()),
    default="api_key",
)
@click.option(
    "--oci-profile", help="oci config profile", required=False, default="DEFAULT"
)
@click.option("--debug", "-d", help="set debug mode", is_flag=True, default=False)
def delete(ocid, auth, oci_profile, debug):
    def _delete(ocid, auth, oci_profile):
        with AuthContext():
            ads.set_auth(auth=auth, profile=oci_profile)
            if "datasciencejobrun" in ocid:
                run = DataScienceJobRun.from_ocid(ocid)
            elif "dataflowrun" in ocid:
                run = DataFlowRun.from_ocid(ocid)
            else:
                raise ValueError("Run OCID not recognized.")
            run.delete()

    suppress_traceback(debug)(_delete)(ocid, auth, oci_profile)
