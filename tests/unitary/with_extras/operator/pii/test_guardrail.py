#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
from io import StringIO

import yaml

from ads.opctl.operator.lowcode.pii.constant import DEFAULT_REPORT_FILENAME
from ads.opctl.operator.lowcode.pii.model.guardrails import PIIGuardrail
from ads.opctl.operator.lowcode.pii.operator_config import PiiOperatorConfig


class TestPiiGuardrail:
    test_files_uri = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_files"
    )

    def yaml_content_simple(self):
        content = StringIO(
            f"""
kind: operator
spec:
  detectors:
  - action: anonymize
    name: default.phone
  input_data:
    url: {self.test_files_uri}/test_data.csv
  output_directory:
    url: {self.test_files_uri}
  report:
    report_filename: {DEFAULT_REPORT_FILENAME}
  target_column: text
type: pii
version: v1

"""
        )
        return content

    def yaml_content_complex(self):
        content = StringIO(
            """
kind: operator
spec:
  detectors:
  - action: anonymize
    name: default.phone
  - action: mask
    name: default.social_security_number
  input_data:
    url: oci://my-bucket@my-tenancy/input_data/mydata.csv
  output_directory:
    name: myProcesseData.csv
    url: oci://my-bucket@my-tenancy/result/
  report:
    report_filename: myreport.html
    show_sensitive_content: true
    show_rows: 10
  target_column: text
type: pii
version: v1

"""
        )
        return content

    def test_init(self):
        conf = yaml.load(self.yaml_content_complex(), yaml.SafeLoader)
        operator_config = PiiOperatorConfig.from_yaml(
            yaml_string=self.yaml_content_complex()
        )
        guardrail = PIIGuardrail(config=operator_config)

        assert guardrail.dst_uri == os.path.join(
            conf["spec"]["output_directory"]["url"],
            conf["spec"]["output_directory"]["name"],
        )
        assert guardrail.report_uri == os.path.join(
            conf["spec"]["output_directory"]["url"],
            conf["spec"]["report"]["report_filename"],
        )
        assert len(guardrail.scrubber._detectors) == 2
        assert not guardrail.storage_options == {}

    def test_load_data(self):
        conf = yaml.load(self.yaml_content_simple(), yaml.SafeLoader)

        operator_config = PiiOperatorConfig.from_yaml(
            yaml_string=self.yaml_content_simple()
        )
        guardrail = PIIGuardrail(config=operator_config)
        guardrail._load_data()

        assert guardrail.datasets is not None
        assert guardrail.storage_options == {}
        assert guardrail.dst_uri == os.path.join(
            conf["spec"]["output_directory"]["url"],
            "test_data_out.csv",
        )
        assert guardrail.report_uri == os.path.join(
            conf["spec"]["output_directory"]["url"],
            DEFAULT_REPORT_FILENAME,
        )

    def test_process(self):
        operator_config = PiiOperatorConfig.from_yaml(
            yaml_string=self.yaml_content_simple()
        )
        guardrail = PIIGuardrail(config=operator_config)
        with tempfile.TemporaryDirectory() as temp_dir:
            dst_uri = os.path.join(temp_dir, "test_out.csv")
            report_uri = os.path.join(temp_dir, DEFAULT_REPORT_FILENAME)
            guardrail.process(
                dst_uri=dst_uri,
                report_uri=report_uri,
            )
            assert os.path.exists(dst_uri)
            assert os.path.exists(report_uri)
