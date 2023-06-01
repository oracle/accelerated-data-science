#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest

try:
    from oci.data_science.models import Pipeline, PipelineRun
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest("Pipeline or PipelineRun not available.")

from tests.integration.pipeline.test_base import PipelineTestCase


class TestPipelineCSStep(PipelineTestCase):
    def test_pipeline_with_cs_steps(self):
        pipeline = self.define_pipeline_with_custom_script_steps()

        try:
            pipeline.create()

            self.assert_pipeline_infrastructure_config_details(pipeline)

            pipeline_run = pipeline.run()

            self.assert_pipeline_run(pipeline, pipeline_run, "CUSTOM_SCRIPT")

            self.wait_for_pipeline_run_to_succeed(pipeline_run)
        except Exception as ex:
            print("Process failed with error: %s", ex)
            exit(1)

        assert pipeline_run.lifecycle_state == PipelineRun.LIFECYCLE_STATE_SUCCEEDED

        self.clean_up_pipeline_run(pipeline_run)

        assert pipeline_run.lifecycle_state == PipelineRun.LIFECYCLE_STATE_DELETED

        self.clean_up_pipeline(pipeline)

        assert (
            pipeline.data_science_pipeline.lifecycle_state
            == Pipeline.LIFECYCLE_STATE_DELETED
        )

    def test_cancel_pipeline_with_cs_steps(self):
        pipeline = self.define_pipeline_with_custom_script_steps()

        try:
            pipeline.create()
            pipeline_run = pipeline.run()

            self.assert_pipeline_run(pipeline, pipeline_run, "CUSTOM_SCRIPT")

            pipeline_run.cancel()
        except Exception as ex:
            print("Process failed with error: %s", ex)
            exit(1)

        assert pipeline_run.lifecycle_state == PipelineRun.LIFECYCLE_STATE_CANCELED

        self.clean_up_pipeline_run(pipeline_run)

        assert pipeline_run.lifecycle_state == PipelineRun.LIFECYCLE_STATE_DELETED

        self.clean_up_pipeline(pipeline)

        assert (
            pipeline.data_science_pipeline.lifecycle_state
            == Pipeline.LIFECYCLE_STATE_DELETED
        )
