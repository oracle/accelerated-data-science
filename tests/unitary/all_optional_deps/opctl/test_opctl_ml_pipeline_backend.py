#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from ads.opctl.backend.ads_ml_pipeline import PipelineBackend
from ads.pipeline import Pipeline, PipelineRun


class TestMLPipelineBackend:
    @property
    def curr_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def config(self):
        return {
            "execution": {
                "backend": "pipeline",
                "use_conda": True,
                "debug": False,
                "oci_config": "~/.oci/config",
                "oci_profile": "DEFAULT",
                "ocid": "test",
                "auth": "api_key",
            },
            "infrastructure": {
                "compartment_id": "ocid1.compartment.oc1..<unique_id>",
                "project_id": "ocid1.datascienceproject.oc1.<unique_id>",
                "log_group_id": "ocid1.loggroup.oc1.iad.<unique_id>",
                "log_id": "ocid1.log.oc1.iad.<unique_id>",
            },
        }

    @patch(
        "ads.opctl.backend.ads_ml_pipeline.Pipeline.run",
        return_value=PipelineRun(id="TestRunId"),
    )
    @patch(
        "ads.opctl.backend.ads_ml_pipeline.Pipeline.create",
        return_value=Pipeline(id="TestId"),
    )
    @patch(
        "ads.opctl.backend.ads_ml_pipeline.Pipeline.from_dict", return_value=Pipeline()
    )
    def test_apply(self, mock_from_dict, mock_create, mock_run):
        config = self.config
        backend = PipelineBackend(config)
        backend.apply()
        mock_from_dict.assert_called_with(config)
        mock_create.assert_called_with()
        mock_run.assert_called_with()

    @patch(
        "ads.opctl.backend.ads_ml_pipeline.Pipeline.run",
        return_value=PipelineRun(),
    )
    @patch(
        "ads.opctl.backend.ads_ml_pipeline.Pipeline.from_ocid", return_value=Pipeline()
    )
    def test_run(self, mock_from_ocid, mock_run):
        config = self.config
        backend = PipelineBackend(config)
        backend.run()
        mock_from_ocid.assert_called_with(ocid=config["execution"]["ocid"])
        mock_run.assert_called_with()

    @patch("ads.opctl.backend.ads_ml_pipeline.Pipeline.delete", return_value=Pipeline())
    @patch(
        "ads.opctl.backend.ads_ml_pipeline.Pipeline.from_ocid", return_value=Pipeline()
    )
    def test_delete(self, mock_from_ocid, mock_delete):
        config = self.config
        config["execution"]["id"] = "test_pipeline_id"
        backend = PipelineBackend(config)
        backend.delete()
        mock_from_ocid.assert_called_with("test_pipeline_id")
        mock_delete.assert_called_with()

    @patch(
        "ads.opctl.backend.ads_ml_pipeline.PipelineRun.delete",
        return_value=PipelineRun(),
    )
    @patch(
        "ads.opctl.backend.ads_ml_pipeline.PipelineRun.from_ocid",
        return_value=PipelineRun(),
    )
    def test_delete_run(self, mock_from_ocid, mock_delete):
        config = self.config
        config["execution"]["run_id"] = "test_pipeline_run_id"
        backend = PipelineBackend(config)
        backend.delete()
        mock_from_ocid.assert_called_with("test_pipeline_run_id")
        mock_delete.assert_called_with()

    @patch(
        "ads.opctl.backend.ads_ml_pipeline.PipelineRun.cancel",
        return_value=PipelineRun(),
    )
    @patch(
        "ads.opctl.backend.ads_ml_pipeline.PipelineRun.from_ocid",
        return_value=PipelineRun(),
    )
    def test_cancel(self, mock_from_ocid, mock_cancel):
        config = self.config
        config["execution"]["run_id"] = "test_pipeline_run_id"
        backend = PipelineBackend(config)
        backend.cancel()
        mock_from_ocid.assert_called_with("test_pipeline_run_id")
        mock_cancel.assert_called_with()

    @patch(
        "ads.opctl.backend.ads_ml_pipeline.PipelineRun.watch",
        return_value=PipelineRun(),
    )
    @patch(
        "ads.opctl.backend.ads_ml_pipeline.PipelineRun.from_ocid",
        return_value=PipelineRun(),
    )
    def test_watch(self, mock_from_ocid, mock_watch):
        config = self.config
        config["execution"]["run_id"] = "test_pipeline_run_id"
        config["execution"]["log_type"] = "custom_log"
        config["execution"]["interval"] = 10
        backend = PipelineBackend(config)
        backend.watch()
        mock_from_ocid.assert_called_with("test_pipeline_run_id")
        mock_watch.assert_called_with(interval=10, log_type="custom_log")

    @pytest.mark.parametrize(
        "runtime_type",
        ["container", "script", "python", "notebook", "gitPython"],
    )
    def test_init(self, runtime_type, monkeypatch):
        """Ensures that starter YAML can be generated for every supported runtime of the Data Flow."""

        monkeypatch.delenv("NB_SESSION_OCID", raising=False)
        monkeypatch.setenv(
            "NB_SESSION_COMPARTMENT_OCID",
            self.config["infrastructure"]["compartment_id"],
        )
        monkeypatch.setenv("PROJECT_OCID", self.config["infrastructure"]["project_id"])

        with tempfile.TemporaryDirectory() as td:
            test_yaml_uri = os.path.join(td, f"pipeline_{runtime_type}.yaml")
            expected_yaml_uri = os.path.join(
                self.curr_dir, "test_files", f"pipeline_{runtime_type}.yaml"
            )

            PipelineBackend(self.config).init(
                uri=test_yaml_uri,
                overwrite=False,
                runtime_type=runtime_type,
            )

            with open(test_yaml_uri, "r") as stream:
                test_yaml_dict = yaml.safe_load(stream)
            with open(expected_yaml_uri, "r") as stream:
                expected_yaml_dict = yaml.safe_load(stream)

            assert test_yaml_dict == expected_yaml_dict
