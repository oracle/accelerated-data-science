#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import time
from collections import namedtuple
from logging import getLogger
import json
import os

from ads.opctl.config.yaml_parsers import YamlSpecParser

from ads.jobs import Job, DataScienceJob, ScriptRuntime


logger = getLogger("ads.yaml")


class OperatorSpecParser(YamlSpecParser):
    def __init__(self, operator):
        # TODO: validate yamlInput
        self.operator = operator

    def parse(self):

        self.operator_spec = self.operator["spec"]
        execution = self.operator["execution"]

        operator_args = json.dumps(self.operator_spec)

        runtime = ScriptRuntime().with_custom_conda(execution["conda_pack"])
        runtime = runtime.with_environment_variable(OPERATOR_ARGS=operator_args)

        infra_override = execution.get("infrastructure", {"spec": {}})["spec"]

        infra = (
            DataScienceJob(**infra_override)
            .with_log_group_id(execution.get("logGroupId"))
            .with_log_id(execution.get("logId"))
        )

        runtime.with_source(self.generate_script())

        return Job(
            name=execution.get("job_name"), infrastructure=infra, runtime=runtime
        )

    def generate_script(self):
        from string import Template
        import tempfile

        string_template = """
python -m ads.operators.low_code.$operator_name
"""
        print(self.operator)
        code = (
            Template(string_template)
            .substitute(operator_name=self.operator.get("name").replace("-", "_"))
            .encode()
            .decode("utf-8")
        )

        dir = tempfile.mkdtemp()
        script_file = os.path.join(
            dir, f'{self.operator["name"]}_{int(time.time())}_run.sh'
        )
        with open(script_file, "w") as fp:
            fp.write(code)
        with open(script_file) as fp:
            print(fp.read())

        return script_file