#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import yaml
import os
from ads.opctl.distributed.common.abstract_cluster_provider import ClusterProvider
from ads.opctl.diagnostics.check_distributed_job_requirements import (
    PortsChecker,
    Default,
)
from ads.opctl.diagnostics.requirement_exception import RequirementException
from jinja2 import Environment, PackageLoader
import fsspec
import importlib
import sys

cp = ClusterProvider(mode="MAIN")
auth = cp.get_oci_auth()


class RequirementChecker:

    providers = {}

    CHECK = "\u2705"
    CROSS = "\u274E"
    TADA = "\u2728"
    DIAGNOSTIC_REPORT_FILE = "diagnostic_report.html"

    @classmethod
    def register_provider(cls, key: str, value):
        RequirementChecker.providers[key] = value

    def load_requirement(self, framework):
        req = {}
        with open(f"{framework}_req.yaml") as rf:
            req = yaml.load(rf, Loader=yaml.SafeLoader)
        return req

    def setup_syspath(self):
        code_dir = os.environ.get("OCI__CODE_DIR", "/code/")
        sys.path.append(code_dir)
        for dirname in os.listdir(code_dir):
            file = os.path.join(code_dir, dirname)
            if os.path.isdir(file):
                sys.path.append(file)

    def load_external_modules(self, modules):
        for module in modules:
            importlib.import_module(module)
        print(f"Loaded modules: {modules}")

    def verify_requirements(self, kind):

        self.kind = kind
        self.setup_syspath()
        configuration = {}
        requirements = {}
        cluster_type = os.environ["OCI__CLUSTER_TYPE"].lower()
        DIAGNOSTIC_CONFIG = os.environ.get(
            "OCI__DIAGNOSTIC_DEFAULT_CONFIG"
        ) or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "diagnostics",
            self.kind,
            "default_requirements_config.yaml",
        )
        print(f"Loading requirements for diagnostics from {DIAGNOSTIC_CONFIG}")
        with open(DIAGNOSTIC_CONFIG) as cf:
            configuration = yaml.load(cf, Loader=yaml.SafeLoader)[cluster_type]
            requirements = configuration.get("checkers")
        if configuration.get("checker_provider"):
            self.load_external_modules(configuration["checker_provider"])

        failures = []
        passed = []
        for requirement in requirements:
            print(f"Checking requirement: {requirement}")
            provider = ".".join(requirement["checker"].split(".")[:-1])
            method = requirement["checker"].split(".")[-1]
            kwargs = requirement.get("args", {})
            try:
                getattr(RequirementChecker.providers[provider](auth=auth), method)(
                    **kwargs
                )
                passed.append(requirement["description"])
            except RequirementException as re:
                failures.append((requirement["description"], str(re)))

        for item in passed:
            print(f"{item}: {RequirementChecker.CHECK}", flush=True)
        for item in failures:
            print(
                f"{item[0]}: {RequirementChecker.CROSS} . Resolution: {item[1]}",
                flush=True,
            )
        if len(failures) == 0:
            print(
                f"{RequirementChecker.TADA} You have satisfied all the requirements for cluster: {cluster_type} \u2728",
                flush=True,
            )
        self.generate_report(cluster_type, passed, failures)

    def generate_report(self, cluster_type, passed, failures):
        _env = Environment(loader=PackageLoader("ads", "opctl/templates"))
        # import jinja2

        # loader = jinja2.FileSystemLoader(".")
        # _env = jinja2.Environment(loader=loader)
        report_template = _env.get_template(f"diagnostic_report_template.jinja2")
        report = [(req, RequirementChecker.CHECK) for req in passed] + [
            (req[0], RequirementChecker.CROSS) for req in failures
        ]
        success_message = f"{RequirementChecker.TADA} You have satisfied all the requirements for cluster: {cluster_type} {RequirementChecker.TADA}"
        report_path = os.path.join(
            os.path.join(os.environ.get("OCI__WORK_DIR", ""), cp.my_work_dir),
            RequirementChecker.DIAGNOSTIC_REPORT_FILE,
        )
        with fsspec.open(
            # "diagnostic_report.html",
            report_path,
            "w",
            **auth,
        ) as of:
            of.write(
                report_template.render(
                    cluster_type=cluster_type,
                    report=report,
                    action_item=failures,
                    success_message=success_message,
                )
            )
            print(f"Finished writing to {report_path}")


if __name__ == "__main__":
    # fetch_security_list(subnet_id)
    RequirementChecker.register_provider("cluster-ports-check", PortsChecker)
    RequirementChecker.register_provider("cluster-generic", Default)
    RequirementChecker().verify_requirements()
