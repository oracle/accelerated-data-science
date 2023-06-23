#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
import time

from ads.jobs.builders.runtimes.python_runtime import (
    NotebookRuntime,
    ScriptRuntime,
    GitPythonRuntime,
)
from tests.integration.config import secrets

try:
    from oci.data_science.models import PipelineRun
    from ads.pipeline.ads_pipeline import Pipeline
    from ads.pipeline.ads_pipeline_step import PipelineStep
    from ads.pipeline.builders.infrastructure.custom_script import (
        CustomScriptStep,
    )
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest(
        "Pipeline, PipelineStep, PipelineRun, CustomScriptStep not available."
    )

MAXIMUM_TIMEOUT = 43
SLEEP_INTERVAL = 60


class PipelineTestCase(unittest.TestCase):
    TENANCY_ID = secrets.common.TENANCY_ID
    COMPARTMENT_ID = secrets.common.COMPARTMENT_ID
    PROJECT_ID = secrets.pipeline.PROJECT_ID
    LOG_GROUP_ID = secrets.common.LOG_GROUP_ID
    LOG_ID = secrets.pipeline.LOG_ID
    SCRIPT_ARTIFACT = secrets.pipeline.SCRIPT_ARTIFACT
    NOTEBOOK_ARTIFACT = secrets.pipeline.NOTEBOOK_ARTIFACT
    ML_JOB_ID = secrets.pipeline.ML_JOB_ID

    def define_pipeline_with_ml_steps(self) -> "Pipeline":
        step_one = (
            PipelineStep("PipelineStepOne")
            .with_job_id(self.ML_JOB_ID)
            .with_description("This is a test pipeline step one")
            ._with_depends_on([])
        )

        step_two = (
            PipelineStep("PipelineStepTwo")
            .with_job_id(self.ML_JOB_ID)
            .with_description("This is a test pipeline step two")
            ._with_depends_on([step_one])
        )

        step_three = (
            PipelineStep("PipelineStepThree")
            .with_job_id(self.ML_JOB_ID)
            .with_description("This is a test pipeline step three")
            ._with_depends_on([step_one])
        )

        step_four = (
            PipelineStep("PipelineStepFour")
            .with_job_id(self.ML_JOB_ID)
            .with_description("This is a test pipeline step four")
            ._with_depends_on([step_two, step_three])
        )

        step_five = (
            PipelineStep("PipelineStepFive")
            .with_job_id(self.ML_JOB_ID)
            .with_description("This is a test pipeline step five")
            ._with_depends_on([step_three])
        )

        pipeline = (
            Pipeline("MLJobIntegrationTest")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_project_id(self.PROJECT_ID)
            .with_log_group_id(self.LOG_GROUP_ID)
            .with_log_id(self.LOG_ID)
            .with_argument("argument", key="value")
            .with_environment_variable(env="value")
            .with_description(
                "This is a pipeline with ml job steps for integration test."
            )
            .with_maximum_runtime_in_minutes(20)
            .with_freeform_tags({"TestFreeformTags": "value"})
            .with_step_details([step_one, step_two, step_three, step_four, step_five])
        )

        return pipeline

    def define_pipeline_with_custom_script_steps(self) -> "Pipeline":
        infrastructure = (
            CustomScriptStep()
            .with_block_storage_size(200)
            .with_shape_name("VM.Standard2.1")
        )

        script_runtime = ScriptRuntime().with_source(self.SCRIPT_ARTIFACT)

        notebook_runtime = (
            NotebookRuntime()
            .with_notebook(path=self.NOTEBOOK_ARTIFACT, encoding="utf-8")
            .with_service_conda("tensorflow26_p37_cpu_v2")
        )

        git_repo_runtime = (
            GitPythonRuntime()
            .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
            .with_service_conda("pytorch19_p37_gpu_v1")
            .with_source("https://github.com/pytorch/tutorials.git")
            .with_entrypoint("beginner_source/examples_nn/polynomial_nn.py")
        )

        pipeline_step_one = (
            PipelineStep("ScriptRuntimeStep")
            .with_infrastructure(infrastructure)
            .with_runtime(script_runtime)
        )

        pipeline_step_two = (
            PipelineStep("NotebookRuntimeStep")
            .with_infrastructure(infrastructure)
            .with_runtime(notebook_runtime)
        )

        pipeline_step_three = (
            PipelineStep("GitRepoRuntimeStep")
            .with_infrastructure(infrastructure)
            .with_runtime(git_repo_runtime)
        )

        pipeline = (
            Pipeline("CustomScriptIntegrationTest")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_project_id(self.PROJECT_ID)
            .with_log_group_id(self.LOG_GROUP_ID)
            .with_log_id(self.LOG_ID)
            .with_argument("argument", key="value")
            .with_block_storage_size_in_gbs(50)
            .with_shape_name("VM.Standard3.Flex")
            .with_shape_config_details(ocpus=1, memory_in_gbs=16)
            .with_environment_variable(env="value")
            .with_description(
                "This is a pipeline with custom script steps for integration test."
            )
            .with_maximum_runtime_in_minutes(20)
            .with_freeform_tags({"TestFreeformTags": "value"})
            .with_step_details(
                # TODO: add back pipeline_step_three, when it fixed (https://jira.oci.oraclecorp.com/browse/ODSC-43071)
                # [pipeline_step_one, pipeline_step_two, pipeline_step_three]
                [pipeline_step_one, pipeline_step_two]
            )
        )

        return pipeline

    def clean_up_pipeline(self, pipeline):
        try:
            pipeline.delete()

            if (
                pipeline.data_science_pipeline.sync().lifecycle_state
                != Pipeline.LIFECYCLE_STATE_DELETED
            ):
                print(
                    "Pipeline stopping after 1800 seconds of not reaching DELETED state."
                )
                exit(1)
        except Exception as ex:
            print("Failed to delete pipeline: ", str(ex))
            exit(1)

    def clean_up_pipeline_run(self, pipeline_run):
        try:
            pipeline_run.delete()

            if (
                pipeline_run.sync().lifecycle_state
                != PipelineRun.LIFECYCLE_STATE_DELETED
            ):
                print(
                    "Pipeline run stopping after 1800 seconds of not reaching DELETED state."
                )
                exit(1)
        except Exception as ex:
            print("Failed to delete pipeline run: ", str(ex))
            exit(1)

    def wait_for_pipeline_run_to_succeed(self, pipeline_run):
        time_counter = 0
        while (
            pipeline_run.sync().lifecycle_state != PipelineRun.LIFECYCLE_STATE_SUCCEEDED
        ):
            time.sleep(SLEEP_INTERVAL)
            if (
                pipeline_run.sync().lifecycle_state
                == PipelineRun.LIFECYCLE_STATE_FAILED
            ):
                print("Pipeline run failed, exiting process.")
                exit(1)
            if time_counter > MAXIMUM_TIMEOUT:
                print(
                    "Pipeline run stopping after 43 minutes of not reaching SUCCESS or FAILED state."
                )
                exit(1)
            time_counter += 1

    def assert_pipeline_infrastructure_config_details(self, pipeline):
        temp_pipeline = Pipeline.from_id(pipeline.id)

        assert (
            temp_pipeline.block_storage_size_in_gbs
            == pipeline.block_storage_size_in_gbs
        )
        assert temp_pipeline.shape_name == pipeline.shape_name
        assert (
            temp_pipeline.shape_config_details["memoryInGBs"]
            == pipeline.shape_config_details["memoryInGBs"]
        )
        assert (
            temp_pipeline.shape_config_details["ocpus"]
            == pipeline.shape_config_details["ocpus"]
        )

    def assert_pipeline_run(self, pipeline, pipeline_run, step_type):
        assert pipeline_run.display_name == pipeline.name
        assert pipeline_run.id != None
        assert pipeline_run.compartment_id == pipeline.compartment_id
        assert pipeline_run.project_id == pipeline.project_id
        assert pipeline_run.pipeline_id == pipeline.id
        assert pipeline_run.configuration_details.type == "DEFAULT"
        assert (
            pipeline_run.configuration_details.environment_variables
            == pipeline.environment_variable
        )
        assert (
            pipeline_run.configuration_details.maximum_runtime_in_minutes
            == pipeline.maximum_runtime_in_minutes
        )
        assert (
            pipeline_run.configuration_details.command_line_arguments
            == pipeline.argument
        )
        assert pipeline_run.log_details.log_group_id == pipeline.log_group_id
        assert pipeline_run.log_details.log_id == pipeline.log_id
        assert pipeline_run.freeform_tags == {"TestFreeformTags": "value"}
        assert pipeline_run.lifecycle_state != None

        for i in range(len(pipeline_run.step_runs)):
            assert pipeline_run.step_runs[i].step_name == pipeline.step_details[i].name
            assert pipeline_run.step_runs[i].step_type == step_type
            assert pipeline_run.step_runs[i].lifecycle_state != None
