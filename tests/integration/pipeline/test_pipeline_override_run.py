#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest

from tests.integration.config import secrets
from tests.integration.pipeline.test_base import PipelineTestCase

try:
    from oci.data_science.models import Pipeline, PipelineRun
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest("Pipeline or PipelineRun not available.")

display_override_name = "TestOverrideName"

configuration_override_details = {
    "maximum_runtime_in_minutes": 30,
    "environment_variables": {"a": "b"},
    "command_line_arguments": "ARGUMENT --KEY VALUE",
}

log_configuration_override_details = {
    "log_group_id": secrets.pipeline.LOG_GROUP_ID_OVERRIDE
}

step_override_details = [
    {
        "step_name": "PipelineStepOne",
        "step_configuration_details": {
            "maximum_runtime_in_minutes": 200,
            "environment_variables": {"1": "2"},
            "command_line_arguments": "argument --key value",
        },
    }
]


class TestPipelineOverrideRun(PipelineTestCase):
    def test_pipeline_run_override(self):
        pipeline = self.define_pipeline_with_ml_steps()

        try:
            pipeline.create()

            pipeline_run = pipeline.run(
                display_name=display_override_name,
                configuration_override_details=configuration_override_details,
                log_configuration_override_details=log_configuration_override_details,
                step_override_details=step_override_details,
            )

            self.assert_pipeline_run_override(pipeline_run)

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

    def assert_pipeline_run_override(self, pipeline_run):
        assert pipeline_run.display_name == display_override_name
        assert pipeline_run.id != None
        assert pipeline_run.configuration_override_details.type == "DEFAULT"
        assert pipeline_run.configuration_override_details.environment_variables == {
            "a": "b"
        }
        assert (
            pipeline_run.configuration_override_details.maximum_runtime_in_minutes == 30
        )
        assert (
            pipeline_run.configuration_override_details.command_line_arguments
            == "ARGUMENT --KEY VALUE"
        )
        assert (
            pipeline_run.log_configuration_override_details.log_group_id
            == secrets.pipeline.LOG_GROUP_ID_OVERRIDE
        )
        assert pipeline_run.step_override_details[0].step_configuration_details != None
        assert (
            pipeline_run.step_override_details[
                0
            ].step_configuration_details.maximum_runtime_in_minutes
            == 200
        )
        assert pipeline_run.step_override_details[
            0
        ].step_configuration_details.environment_variables == {"1": "2"}
        assert (
            pipeline_run.step_override_details[
                0
            ].step_configuration_details.command_line_arguments
            == "argument --key value"
        )
