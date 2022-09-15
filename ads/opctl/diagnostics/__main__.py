#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.diagnostics.check_requirements import RequirementChecker
from ads.opctl.diagnostics.check_distributed_job_requirements import (
    PortsChecker,
    Default,
)
import argparse

RequirementChecker.register_provider("cluster-ports-check", PortsChecker)
RequirementChecker.register_provider("cluster-generic", Default)

parser = argparse.ArgumentParser(description="Run diagnostics to check your setup")
parser.add_argument(
    "-t",
    "--type",
    help="What type of diagnostics to run. Eg. `distributed` to run diagnostics for distributed training setup",
)

args = parser.parse_args()
RequirementChecker().verify_requirements(args.type)
