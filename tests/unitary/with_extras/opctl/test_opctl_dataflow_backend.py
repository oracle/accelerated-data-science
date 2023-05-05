#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ads.opctl.backend.ads_dataflow import DataFlowBackend


class TestDataFlowBackend:
    @property
    def curr_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def config(self):
        return {
            "execution": {
                "backend": "dataflow",
                "auth": "api_key",
                "oci_profile": "DEFAULT",
                "oci_config": "~/.oci/config",
            },
            "infrastructure": {
                "compartment_id": "ocid1.compartment.oc1..<unique_id>",
                "driver_shape": "VM.Standard.E2.4",
                "executor_shape": "VM.Standard.E2.4",
                "logs_bucket_uri": "oci://bucket@namespace",
                "script_bucket": "oci://bucket@namespace/prefix",
                "num_executors": "1",
                "spark_version": "3.2.1",
            },
        }

    def test_dataflow_apply(self):
        with pytest.raises(NotImplementedError):
            DataFlowBackend(self.config).apply()

    @patch("ads.jobs.builders.infrastructure.dataflow.DataFlowApp.create")
    @patch("ads.opctl.backend.ads_dataflow.Job.run")
    @patch("ads.jobs.DataFlow._upload_file")
    def test_dataflow_run(self, file_upload, job_run, job_create):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "src"))
            Path(os.path.join(td, "src", "main.py")).touch()
            config = {
                "execution": {
                    "backend": "dataflow",
                    "source_folder": os.path.join(td, "src"),
                    "entrypoint": "main.py",
                    "command": "--opt v arg",
                    "auth": "api_key",
                    "oci_profile": "DEFAULT",
                    "oci_config": "~/.oci/config",
                    "archive": "archive.zip",
                },
                "infrastructure": {
                    "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
                    "driver_shape": "VM.Standard2.1",
                    "executor_shape": "VM.Standard2.1",
                    "logs_bucket_uri": "oci://<bucket_name>@<namespace>/<prefix>",
                    "configurations": json.dumps({"spark.driver.memory": "512m"}),
                    "script_bucket": "oci://<bucket_name>@<namespace>/<prefix>",
                    "archive_bucket": "oci://<bucket_name>@<namespace>/<prefix>",
                },
            }

            DataFlowBackend(config).run()
            job_create.assert_called()
            job_run.assert_called()
            file_upload.assert_any_call(
                os.path.join(td, "src", "main.py"),
                "oci://<bucket_name>@<namespace>/<prefix>",
                False,
            )
            file_upload.assert_any_call(
                "archive.zip",
                "oci://<bucket_name>@<namespace>/<prefix>",
                False,
            )

    @pytest.mark.parametrize(
        "runtime_type",
        ["dataFlow", "dataFlowNotebook"],
    )
    def test_init(self, runtime_type, monkeypatch):
        """Ensures that starter YAML can be generated for every supported runtime of the Data Flow."""
        monkeypatch.delenv("NB_SESSION_OCID", raising=False)

        with tempfile.TemporaryDirectory() as td:
            test_yaml_uri = os.path.join(td, f"dataflow_{runtime_type}.yaml")
            expected_yaml_uri = os.path.join(
                self.curr_dir, "test_files", f"dataflow_{runtime_type}.yaml"
            )

            DataFlowBackend(self.config).init(
                uri=test_yaml_uri,
                overwrite=False,
                runtime_type=runtime_type,
            )

            with open(test_yaml_uri, "r") as stream:
                test_yaml_dict = yaml.safe_load(stream)
            with open(expected_yaml_uri, "r") as stream:
                expected_yaml_dict = yaml.safe_load(stream)

            assert test_yaml_dict == expected_yaml_dict