#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
from unittest.mock import patch

try:
    from ads.opctl.backend.ads_ml_pipeline import PipelineBackend
    from ads.pipeline import Pipeline, PipelineRun
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest(
        "OCI MLPipeline is not available. Skipping the MLPipeline tests."
    )


class TestMLPipelineBackend:
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
            }
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
        backend = PipelineBackend(config)
        backend.watch()
        mock_from_ocid.assert_called_with("test_pipeline_run_id")
        mock_watch.assert_called_with(log_type="custom_log")
