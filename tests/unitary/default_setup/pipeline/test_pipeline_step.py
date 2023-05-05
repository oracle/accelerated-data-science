#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
import pytest

try:
    from ads.jobs.builders.runtimes.python_runtime import ScriptRuntime
    from ads.pipeline.ads_pipeline_step import PipelineStep
    from ads.pipeline.builders.infrastructure.custom_script import (
        CustomScriptStep,
    )
except ImportError:
    raise unittest.SkipTest(
        "OCI MLPipeline is not available. Skipping the MLPipeline tests."
    )


class DataSciencePipelineStepBaseTest(unittest.TestCase):
    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard2.1")
    )

    runtime = (
        ScriptRuntime()
        .with_source("oci://bucket_name@namespace/path/to/train.py")
        .with_service_conda("tensorflow26_p37_cpu_v2")
        .with_environment_variable(ENV="value")
        .with_argument("argument", key="value")
        .with_freeform_tag(tag_name="tag_value")
    )

    upstream_pipeline_step_one = (
        PipelineStep("TestUpstreamPipelineStepOne")
        .with_description("Test upstream pipeline step description one")
        .with_job_id("TestJobIdOne")
    )

    upstream_pipeline_step_two = (
        PipelineStep("TestUpstreamPipelineStepTwo")
        .with_description("Test upstream pipeline step description two")
        .with_infrastructure(infrastructure)
        .with_runtime(runtime)
    )

    def test_pipeline_step_define(self):
        assert self.upstream_pipeline_step_one.kind == "ML_JOB"
        assert self.upstream_pipeline_step_two.kind == "CUSTOM_SCRIPT"

    def test_pipeline_step_define_from_arguments(self):
        pipeline_step_one = PipelineStep(
            name="TestNameStepOne",
            job_id="TestJobId",
            description="TestDescription",
            maximum_runtime_in_minutes=200,
            environment_variable={"TestKey": "TestValue"},
            command_line_argument="TestCommandLine",
        )

        pipeline_step_two = PipelineStep(
            infrastructure=self.infrastructure,
            runtime=self.runtime,
            name="TestNameStepTwo",
            description="TestDescription",
        )

        assert pipeline_step_one.job_id == "TestJobId"
        assert pipeline_step_one.description == "TestDescription"
        assert pipeline_step_one.maximum_runtime_in_minutes == 200
        assert pipeline_step_one.environment_variable == {"TestKey": "TestValue"}
        assert pipeline_step_one.argument == "TestCommandLine"

        assert pipeline_step_two.runtime == self.runtime
        assert pipeline_step_two.infrastructure == self.infrastructure
        assert pipeline_step_two.description == "TestDescription"

    def test_pipeline_step_to_dict(self):
        assert self.upstream_pipeline_step_one.to_dict() == {
            "kind": "dataScienceJob",
            "spec": {
                "name": "TestUpstreamPipelineStepOne",
                "jobId": "TestJobIdOne",
                "description": "Test upstream pipeline step description one",
            },
        }

        assert self.upstream_pipeline_step_two.to_dict() == {
            "kind": "customScript",
            "spec": {
                "name": "TestUpstreamPipelineStepTwo",
                "runtime": {
                    "kind": "runtime",
                    "type": "script",
                    "spec": {
                        "scriptPathURI": "oci://bucket_name@namespace/path/to/train.py",
                        "conda": {"type": "service", "slug": "tensorflow26_p37_cpu_v2"},
                        "env": [{"name": "ENV", "value": "value"}],
                        "args": ["argument", "--key", "value"],
                        "freeformTags": {"tag_name": "tag_value"},
                    },
                },
                "infrastructure": {
                    "kind": "infrastructure",
                    "spec": {"blockStorageSize": 200, "shapeName": "VM.Standard2.1"},
                },
                "description": "Test upstream pipeline step description two",
            },
        }

    def test_pipeline_step_from_yaml_job(self):
        yaml_string = """
        kind: dataScienceJob
        spec:
          description: Test description one
          jobId: TestJobIdOne
          name: TestPipelineStepOne
        """
        step = PipelineStep.from_yaml(yaml_string)
        assert step.kind == "ML_JOB"

    def test_pipeline_step_from_yaml_custom(self):
        yaml_string = """
        kind: customScript
        spec:
          description: Test description two
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              jobInfrastructureType: ME_STANDALONE
              jobType: DEFAULT
              shapeName: VM.Standard2.1
          name: TestPipelineStepTwo
          runtime:
            kind: runtime
            spec:
              conda:
                slug: tensorflow26_p37_cpu_v2
                type: service
              notebookEncoding: utf-8
              notebookPathURI: https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb
              outputUri: oci://bucket_name@namespace/path/to/dir
            type: notebook
        """
        step = PipelineStep.from_yaml(yaml_string)
        assert step.kind == "CUSTOM_SCRIPT"
        assert step.name == "TestPipelineStepTwo"

    def test_pipeline_step_name_value_error(self):
        """Validate error when step name not provided."""
        with pytest.raises(ValueError):
            PipelineStep(name="*&%$#!@)(*&*%$#@")
