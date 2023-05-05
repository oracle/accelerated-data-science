#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import datetime
import os
import random
import tempfile
import unittest
from unittest.mock import Mock, PropertyMock, patch

import oci
import pytest
import yaml

from ads.common.oci_datascience import DSCNotebookSession
from ads.common.oci_logging import OCILog
from ads.common.oci_mixin import OCIModelMixin
from ads.common.utils import batch_convert_case
from ads.jobs.builders.runtimes.python_runtime import NotebookRuntime
from ads.pipeline.ads_pipeline import DataSciencePipeline, Pipeline
from ads.pipeline.ads_pipeline_run import PipelineRun
from ads.pipeline.ads_pipeline_step import PipelineStep
from ads.pipeline.builders.infrastructure.custom_script import CustomScriptStep
from ads.pipeline.visualizer.base import PipelineVisualizer

PIPELINE_PAYLOAD = dict(
    compartment_id="ocid1.compartment.oc1..<unique_ocid>",
    project_id="ocid1.datascienceproject.oc1.iad.<unique_ocid>",
    nb_session_ocid="ocid1.datasciencenotebooksession.oc1.iad..<unique_ocid>",
    shape_name="VM.Standard.E3.Flex",
    block_storage_size_in_gbs=100,
    shape_config_details={"ocpus": 1, "memory_in_gbs": 16},
)
PIPELINE_OCID = "ocid.xxx.datasciencepipeline.<unique_ocid>"

DEFAULT_WAITER_KWARGS = {"max_wait_seconds": 1800}
DEFAULT_OPERATION_KWARGS = {
    "delete_related_pipeline_runs": True,
    "delete_related_job_runs": True,
}

pipeline_step_one = (
    PipelineStep("TestPipelineStepOne")
    .with_description("Test description one")
    .with_job_id("TestJobIdOne")
    .with_maximum_runtime_in_minutes(20)
    .with_environment_variable(ENV="VALUE")
    .with_argument("ARGUMENT", KEY="VALUE")
)

infrastructure = (
    CustomScriptStep()
    .with_block_storage_size(200)
    .with_shape_name("VM.Standard3.Flex")
    .with_shape_config_details(ocpus=1, memory_in_gbs=16)
)

runtime = (
    NotebookRuntime()
    .with_notebook(
        path="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb"
    )
    .with_service_conda("tensorflow26_p37_cpu_v2")
    .with_output("oci://bucket_name@namespace/path/to/dir")
)

pipeline_step_two = (
    PipelineStep("TestPipelineStepTwo")
    .with_description("Test description two")
    .with_infrastructure(infrastructure)
    .with_runtime(runtime)
)

pipeline_one = (
    Pipeline("TestPipeline")
    .with_id("TestId")
    .with_compartment_id("TestCompartmentId")
    .with_project_id("TestProjectId")
    .with_log_group_id("TestLogGroupId")
    .with_log_id("TestLogId")
    .with_description("TestDescription")
    .with_maximum_runtime_in_minutes(200)
    .with_argument("argument", key="value")
    .with_environment_variable(env="value")
    .with_freeform_tags({"key": "value"})
    .with_defined_tags({"key": "value"})
    .with_block_storage_size_in_gbs(200)
    .with_shape_name("VM.Standard2.1")
    .with_shape_config_details(ocpus=1, memory_in_gbs=2)
    .with_step_details([pipeline_step_one, pipeline_step_two])
    .with_dag(["TestPipelineStepOne >> TestPipelineStepTwo"])
)

pipeline_two = Pipeline(
    id="TestId",
    compartment_id="TestCompartmentId",
    project_id="TestProjectId",
    display_name="TestPipeline",
    environment_variables={"env": "value"},
    command_line_arguments="argument --key value",
    log_id="TestLogId",
    log_group_id="TestLogGroupId",
    description="TestDescription",
    maximum_runtime_in_minutes=200,
    freeform_tags={"key": "value"},
    defined_tags={"key": "value"},
    block_storage_size_in_gbs=200,
    shape_name="VM.Standard2.1",
    shape_config_details={"ocpus": 1, "memoryInGBs": 2},
    step_details=[pipeline_step_one, pipeline_step_two],
    dag=["TestPipelineStepOne >> TestPipelineStepTwo"],
)

pipeline_three = Pipeline(
    name="TestPipeline",
    spec={
        "id": "TestId",
        "compartment_id": "TestCompartmentId",
        "project_id": "TestProjectId",
        "environment_variables": {"env": "value"},
        "command_line_arguments": "argument --key value",
        "log_id": "TestLogId",
        "log_group_id": "TestLogGroupId",
        "description": "TestDescription",
        "maximum_runtime_in_minutes": 200,
        "freeform_tags": {"key": "value"},
        "defined_tags": {"key": "value"},
        "block_storage_size_in_gbs": 200,
        "shape_name": "VM.Standard2.1",
        "shape_config_details": {"ocpus": 1, "memoryInGBs": 2},
        "step_details": [pipeline_step_one, pipeline_step_two],
        "dag": ["TestPipelineStepOne >> TestPipelineStepTwo"],
    },
)

pipeline_four = Pipeline(
    name="TestPipeline",
    spec={
        "id": "TestId",
        "compartmentId": "TestCompartmentId",
        "projectId": "TestProjectId",
        "environmentVariables": {"env": "value"},
        "commandLineArguments": "argument --key value",
        "logId": "TestLogId",
        "logGroupId": "TestLogGroupId",
        "description": "TestDescription",
        "maximumRuntimeInMinutes": 200,
        "freeformTags": {"key": "value"},
        "definedTags": {"key": "value"},
        "blockStorageSizeInGBs": 200,
        "shapeName": "VM.Standard2.1",
        "shapeConfigDetails": {"ocpus": 1, "memoryInGBs": 2},
        "stepDetails": [pipeline_step_one, pipeline_step_two],
        "dag": ["TestPipelineStepOne >> TestPipelineStepTwo"],
    },
)

pipeline_details = {
    "commandLineArguments": "argument --key value",
    "compartmentId": "TestCompartmentId",
    "displayName": "TestPipeline",
    "environmentVariables": {
        "env": "value",
    },
    "id": "TestId",
    "logGroupId": "TestLogGroupId",
    "logId": "TestLogId",
    "description": "TestDescription",
    "maximumRuntimeInMinutes": 200,
    "projectId": "TestProjectId",
    "freeformTags": {
        "key": "value",
    },
    "definedTags": {
        "key": "value",
    },
    "blockStorageSizeInGBs": 200,
    "shapeName": "VM.Standard2.1",
    "shapeConfigDetails": {"ocpus": 1, "memoryInGBs": 2},
    "dag": ["TestPipelineStepOne >> TestPipelineStepTwo"],
    "stepDetails": [
        {
            "stepName": "TestPipelineStepOne",
            "stepType": "ML_JOB",
            "jobId": "TestJobIdOne",
            "description": "Test description one",
            "stepConfigurationDetails": {
                "commandLineArguments": "ARGUMENT --KEY VALUE",
                "environmentVariables": {"ENV": "VALUE"},
                "maximumRuntimeInMinutes": 20,
                "type": "DEFAULT",
            },
            "dependsOn": [],
        },
        {
            "stepName": "TestPipelineStepTwo",
            "stepType": "CUSTOM_SCRIPT",
            "description": "Test description two",
            "dependsOn": ["TestPipelineStepOne"],
            "stepInfrastructureConfigurationDetails": {
                "shapeName": "VM.Standard3.Flex",
                "blockStorageSizeInGBs": 200,
                "shapeConfigDetails": {"memoryInGBs": 16, "ocpus": 1},
            },
            "stepConfigurationDetails": {
                "type": "DEFAULT",
                "environmentVariables": {
                    "CONDA_ENV_TYPE": "service",
                    "CONDA_ENV_SLUG": "tensorflow26_p37_cpu_v2",
                    "JOB_RUN_NOTEBOOK": "basics.ipynb",
                    "JOB_RUN_ENTRYPOINT": "driver_notebook.py",
                    "NOTEBOOK_ENCODING": "utf-8",
                    "OUTPUT_URI": "oci://bucket_name@namespace/path/to/dir",
                },
            },
        },
    ],
}

yaml_string = """
kind: pipeline
spec:
  compartmentId: TestCompartmentId
  dag:
  - TestPipelineStepOne >> TestPipelineStepTwo
  description: This is a test pipeline using ads sdk
  displayName: TestPipeline
  freeformTags:
    key: value
  id: TestId
  logGroupId: TestLogGroupId
  logId: TestLogId
  projectId: TestProjectId
  stepConfigDetails:
    ocpus: 1
    memoryInGBs: 2
  stepDetails:
  - kind: dataScienceJob
    spec:
      description: Test description one
      jobId: TestJobIdOne
      name: TestPipelineStepOne
  - kind: customScript
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


class TestDataSciencePipelineBase:
    def setup_method(self):
        self.mock_default_properties = {
            Pipeline.CONST_COMPARTMENT_ID: PIPELINE_PAYLOAD["compartment_id"],
            Pipeline.CONST_PROJECT_ID: PIPELINE_PAYLOAD["project_id"],
            Pipeline.CONST_SHAPE_NAME: PIPELINE_PAYLOAD["shape_name"],
            Pipeline.CONST_BLOCK_STORAGE_SIZE: PIPELINE_PAYLOAD[
                "block_storage_size_in_gbs"
            ],
            Pipeline.CONST_SHAPE_CONFIG_DETAILS: PIPELINE_PAYLOAD[
                "shape_config_details"
            ],
        }
        self.nb_session = DSCNotebookSession(
            **{
                "notebook_session_configuration_details": {
                    "shape": "VM.Standard.E3.Flex",
                    "block_storage_size_in_gbs": 100,
                    "subnet_id": "test_subnet_id",
                    "notebook_session_shape_config_details": {
                        "ocpus": 1.0,
                        "memory_in_gbs": 16.0,
                    },
                }
            }
        )

    def test_pipeline_define(self):
        output_yaml_one = pipeline_one.to_yaml()
        output_yaml_two = pipeline_two.to_yaml()
        output_yaml_three = pipeline_three.to_yaml()
        output_yaml_four = pipeline_four.to_yaml()
        actual_yaml_output = yaml.safe_load(output_yaml_one)

        assert actual_yaml_output == {
            "kind": "pipeline",
            "spec": {
                "commandLineArguments": "argument --key value",
                "compartmentId": "TestCompartmentId",
                "displayName": "TestPipeline",
                "environmentVariables": {
                    "env": "value",
                },
                "id": "TestId",
                "logGroupId": "TestLogGroupId",
                "logId": "TestLogId",
                "description": "TestDescription",
                "maximumRuntimeInMinutes": 200,
                "projectId": "TestProjectId",
                "freeformTags": {
                    "key": "value",
                },
                "definedTags": {
                    "key": "value",
                },
                "blockStorageSizeInGBs": 200,
                "shapeName": "VM.Standard2.1",
                "shapeConfigDetails": {"ocpus": 1, "memoryInGBs": 2},
                "dag": ["TestPipelineStepOne >> TestPipelineStepTwo"],
                "stepDetails": [
                    {
                        "kind": "dataScienceJob",
                        "spec": {
                            "name": "TestPipelineStepOne",
                            "jobId": "TestJobIdOne",
                            "description": "Test description one",
                        },
                    },
                    {
                        "kind": "customScript",
                        "spec": {
                            "name": "TestPipelineStepTwo",
                            "description": "Test description two",
                            "infrastructure": {
                                "kind": "infrastructure",
                                "spec": {
                                    "blockStorageSize": 200,
                                    "shapeName": "VM.Standard3.Flex",
                                    "shapeConfigDetails": {
                                        "memoryInGBs": 16,
                                        "ocpus": 1,
                                    },
                                },
                            },
                            "runtime": {
                                "kind": "runtime",
                                "spec": {
                                    "conda": {
                                        "slug": "tensorflow26_p37_cpu_v2",
                                        "type": "service",
                                    },
                                    "notebookEncoding": "utf-8",
                                    "notebookPathURI": "https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
                                    "outputUri": "oci://bucket_name@namespace/path/to/dir",
                                },
                                "type": "notebook",
                            },
                        },
                    },
                ],
            },
            "type": "pipeline",
        }

        assert actual_yaml_output == yaml.safe_load(output_yaml_two)
        assert actual_yaml_output == yaml.safe_load(output_yaml_three)
        assert actual_yaml_output == yaml.safe_load(output_yaml_four)

    def test_get_pipeline_details(self):
        actual_pipeline_details = pipeline_one._Pipeline__pipeline_details()

        assert actual_pipeline_details == {
            "compartmentId": "TestCompartmentId",
            "displayName": "TestPipeline",
            "pipelineId": "TestId",
            "configurationDetails": {
                "type": "DEFAULT",
                "commandLineArguments": "argument --key value",
                "environmentVariables": {
                    "env": "value",
                },
                "maximumRuntimeInMinutes": 200,
            },
            "logConfigurationDetails": {
                "logId": "TestLogId",
                "logGroupId": "TestLogGroupId",
                "enableLogging": True,
                "enableAutoLogCreation": False,
            },
            "description": "TestDescription",
            "projectId": "TestProjectId",
            "freeformTags": {
                "key": "value",
            },
            "definedTags": {
                "key": "value",
            },
            "infrastructureConfigurationDetails": {
                "blockStorageSizeInGBs": 200,
                "shapeName": "VM.Standard2.1",
                "shapeConfigDetails": {"ocpus": 1, "memoryInGBs": 2},
            },
            "dag": ["TestPipelineStepOne >> TestPipelineStepTwo"],
            "stepDetails": [
                {
                    "stepName": "TestPipelineStepOne",
                    "stepType": "ML_JOB",
                    "jobId": "TestJobIdOne",
                    "description": "Test description one",
                    "stepConfigurationDetails": {
                        "commandLineArguments": "ARGUMENT --KEY VALUE",
                        "environmentVariables": {"ENV": "VALUE"},
                        "maximumRuntimeInMinutes": 20,
                        "type": "DEFAULT",
                    },
                    "dependsOn": [],
                },
                {
                    "stepName": "TestPipelineStepTwo",
                    "stepType": "CUSTOM_SCRIPT",
                    "description": "Test description two",
                    "dependsOn": ["TestPipelineStepOne"],
                    "stepInfrastructureConfigurationDetails": {
                        "blockStorageSizeInGBs": 200,
                        "shapeName": "VM.Standard3.Flex",
                        "shapeConfigDetails": {"memoryInGBs": 16, "ocpus": 1},
                    },
                    "stepConfigurationDetails": {
                        "type": "DEFAULT",
                        "environmentVariables": {
                            "CONDA_ENV_TYPE": "service",
                            "CONDA_ENV_SLUG": "tensorflow26_p37_cpu_v2",
                            "JOB_RUN_NOTEBOOK": "basics.ipynb",
                            "JOB_RUN_ENTRYPOINT": "driver_notebook.py",
                            "NOTEBOOK_ENCODING": "utf-8",
                            "OUTPUT_URI": "oci://bucket_name@namespace/path/to/dir",
                        },
                    },
                },
            ],
        }

        assert actual_pipeline_details == pipeline_two._Pipeline__pipeline_details()
        assert actual_pipeline_details == pipeline_three._Pipeline__pipeline_details()

    def test_get_pipeline_configuration_details(self):
        temp_pipeline_details = copy.deepcopy(pipeline_details)
        pipeline_configuration_details = (
            pipeline_one._Pipeline__pipeline_configuration_details(
                temp_pipeline_details
            )
        )

        assert pipeline_configuration_details == {
            "type": "DEFAULT",
            "commandLineArguments": "argument --key value",
            "environmentVariables": {
                "env": "value",
            },
            "maximumRuntimeInMinutes": 200,
        }
        assert temp_pipeline_details == {
            "compartmentId": "TestCompartmentId",
            "displayName": "TestPipeline",
            "id": "TestId",
            "logGroupId": "TestLogGroupId",
            "logId": "TestLogId",
            "projectId": "TestProjectId",
            "description": "TestDescription",
            "freeformTags": {
                "key": "value",
            },
            "definedTags": {
                "key": "value",
            },
            "blockStorageSizeInGBs": 200,
            "shapeName": "VM.Standard2.1",
            "shapeConfigDetails": {"ocpus": 1, "memoryInGBs": 2},
            "dag": ["TestPipelineStepOne >> TestPipelineStepTwo"],
            "stepDetails": [
                {
                    "stepName": "TestPipelineStepOne",
                    "stepType": "ML_JOB",
                    "jobId": "TestJobIdOne",
                    "description": "Test description one",
                    "stepConfigurationDetails": {
                        "commandLineArguments": "ARGUMENT --KEY VALUE",
                        "environmentVariables": {"ENV": "VALUE"},
                        "maximumRuntimeInMinutes": 20,
                        "type": "DEFAULT",
                    },
                    "dependsOn": [],
                },
                {
                    "stepName": "TestPipelineStepTwo",
                    "stepType": "CUSTOM_SCRIPT",
                    "description": "Test description two",
                    "dependsOn": ["TestPipelineStepOne"],
                    "stepInfrastructureConfigurationDetails": {
                        "shapeName": "VM.Standard3.Flex",
                        "blockStorageSizeInGBs": 200,
                        "shapeConfigDetails": {
                            "memoryInGBs": 16,
                            "ocpus": 1,
                        },
                    },
                    "stepConfigurationDetails": {
                        "type": "DEFAULT",
                        "environmentVariables": {
                            "CONDA_ENV_TYPE": "service",
                            "CONDA_ENV_SLUG": "tensorflow26_p37_cpu_v2",
                            "JOB_RUN_NOTEBOOK": "basics.ipynb",
                            "JOB_RUN_ENTRYPOINT": "driver_notebook.py",
                            "NOTEBOOK_ENCODING": "utf-8",
                            "OUTPUT_URI": "oci://bucket_name@namespace/path/to/dir",
                        },
                    },
                },
            ],
        }

    def test_get_pipeline_log_configuration_details(self):
        temp_pipeline_details = copy.deepcopy(pipeline_details)
        pipeline_log_configuration_details = (
            pipeline_one._Pipeline__pipeline_log_configuration_details(
                temp_pipeline_details
            )
        )

        assert pipeline_log_configuration_details == {
            "logId": "TestLogId",
            "logGroupId": "TestLogGroupId",
            "enableLogging": True,
            "enableAutoLogCreation": False,
        }

        assert temp_pipeline_details == {
            "commandLineArguments": "argument --key value",
            "compartmentId": "TestCompartmentId",
            "displayName": "TestPipeline",
            "environmentVariables": {
                "env": "value",
            },
            "id": "TestId",
            "projectId": "TestProjectId",
            "freeformTags": {
                "key": "value",
            },
            "definedTags": {
                "key": "value",
            },
            "description": "TestDescription",
            "maximumRuntimeInMinutes": 200,
            "blockStorageSizeInGBs": 200,
            "shapeName": "VM.Standard2.1",
            "shapeConfigDetails": {"ocpus": 1, "memoryInGBs": 2},
            "dag": ["TestPipelineStepOne >> TestPipelineStepTwo"],
            "stepDetails": [
                {
                    "stepName": "TestPipelineStepOne",
                    "stepType": "ML_JOB",
                    "jobId": "TestJobIdOne",
                    "description": "Test description one",
                    "stepConfigurationDetails": {
                        "commandLineArguments": "ARGUMENT --KEY VALUE",
                        "environmentVariables": {"ENV": "VALUE"},
                        "maximumRuntimeInMinutes": 20,
                        "type": "DEFAULT",
                    },
                    "dependsOn": [],
                },
                {
                    "stepName": "TestPipelineStepTwo",
                    "stepType": "CUSTOM_SCRIPT",
                    "description": "Test description two",
                    "dependsOn": ["TestPipelineStepOne"],
                    "stepInfrastructureConfigurationDetails": {
                        "shapeName": "VM.Standard3.Flex",
                        "blockStorageSizeInGBs": 200,
                        "shapeConfigDetails": {
                            "memoryInGBs": 16,
                            "ocpus": 1,
                        },
                    },
                    "stepConfigurationDetails": {
                        "type": "DEFAULT",
                        "environmentVariables": {
                            "CONDA_ENV_TYPE": "service",
                            "CONDA_ENV_SLUG": "tensorflow26_p37_cpu_v2",
                            "JOB_RUN_NOTEBOOK": "basics.ipynb",
                            "JOB_RUN_ENTRYPOINT": "driver_notebook.py",
                            "NOTEBOOK_ENCODING": "utf-8",
                            "OUTPUT_URI": "oci://bucket_name@namespace/path/to/dir",
                        },
                    },
                },
            ],
        }

    def test_get_pipeline_infrastructure_configuration_details(self):
        temp_pipeline_details = copy.deepcopy(pipeline_details)
        pipeline_infrastructure_configuration_details = (
            pipeline_one._Pipeline__pipeline_infrastructure_configuration_details(
                temp_pipeline_details
            )
        )

        assert pipeline_infrastructure_configuration_details == {
            "blockStorageSizeInGBs": 200,
            "shapeName": "VM.Standard2.1",
            "shapeConfigDetails": {"ocpus": 1, "memoryInGBs": 2},
        }

        assert temp_pipeline_details == {
            "commandLineArguments": "argument --key value",
            "compartmentId": "TestCompartmentId",
            "displayName": "TestPipeline",
            "environmentVariables": {
                "env": "value",
            },
            "id": "TestId",
            "logGroupId": "TestLogGroupId",
            "logId": "TestLogId",
            "projectId": "TestProjectId",
            "freeformTags": {
                "key": "value",
            },
            "definedTags": {
                "key": "value",
            },
            "description": "TestDescription",
            "maximumRuntimeInMinutes": 200,
            "dag": ["TestPipelineStepOne >> TestPipelineStepTwo"],
            "stepDetails": [
                {
                    "stepName": "TestPipelineStepOne",
                    "stepType": "ML_JOB",
                    "jobId": "TestJobIdOne",
                    "description": "Test description one",
                    "stepConfigurationDetails": {
                        "commandLineArguments": "ARGUMENT --KEY VALUE",
                        "environmentVariables": {"ENV": "VALUE"},
                        "maximumRuntimeInMinutes": 20,
                        "type": "DEFAULT",
                    },
                    "dependsOn": [],
                },
                {
                    "stepName": "TestPipelineStepTwo",
                    "stepType": "CUSTOM_SCRIPT",
                    "description": "Test description two",
                    "dependsOn": ["TestPipelineStepOne"],
                    "stepInfrastructureConfigurationDetails": {
                        "blockStorageSizeInGBs": 200,
                        "shapeName": "VM.Standard3.Flex",
                        "shapeConfigDetails": {
                            "memoryInGBs": 16,
                            "ocpus": 1,
                        },
                    },
                    "stepConfigurationDetails": {
                        "type": "DEFAULT",
                        "environmentVariables": {
                            "CONDA_ENV_TYPE": "service",
                            "CONDA_ENV_SLUG": "tensorflow26_p37_cpu_v2",
                            "JOB_RUN_NOTEBOOK": "basics.ipynb",
                            "JOB_RUN_ENTRYPOINT": "driver_notebook.py",
                            "NOTEBOOK_ENCODING": "utf-8",
                            "OUTPUT_URI": "oci://bucket_name@namespace/path/to/dir",
                        },
                    },
                },
            ],
        }

    def test_get_step_details(self):
        step_details = pipeline_one._Pipeline__step_details(pipeline_details)

        assert step_details == [
            {
                "stepName": "TestPipelineStepOne",
                "stepType": "ML_JOB",
                "jobId": "TestJobIdOne",
                "description": "Test description one",
                "stepConfigurationDetails": {
                    "commandLineArguments": "ARGUMENT --KEY VALUE",
                    "environmentVariables": {"ENV": "VALUE"},
                    "maximumRuntimeInMinutes": 20,
                    "type": "DEFAULT",
                },
                "dependsOn": [],
            },
            {
                "stepName": "TestPipelineStepTwo",
                "stepType": "CUSTOM_SCRIPT",
                "description": "Test description two",
                "dependsOn": ["TestPipelineStepOne"],
                "stepInfrastructureConfigurationDetails": {
                    "shapeName": "VM.Standard3.Flex",
                    "blockStorageSizeInGBs": 200,
                    "shapeConfigDetails": {
                        "memoryInGBs": 16,
                        "ocpus": 1,
                    },
                },
                "stepConfigurationDetails": {
                    "type": "DEFAULT",
                    "environmentVariables": {
                        "CONDA_ENV_TYPE": "service",
                        "CONDA_ENV_SLUG": "tensorflow26_p37_cpu_v2",
                        "JOB_RUN_NOTEBOOK": "basics.ipynb",
                        "JOB_RUN_ENTRYPOINT": "driver_notebook.py",
                        "NOTEBOOK_ENCODING": "utf-8",
                        "OUTPUT_URI": "oci://bucket_name@namespace/path/to/dir",
                    },
                },
            },
        ]

    def test_override_configurations(self):
        project_id = "TestOverrideProjectId"
        compartment_id = "TestOverrideCompartmentId"
        display_name = "TestOverrideName"
        configuration_override_details = {
            "maximum_runtime_in_minutes": 30,
            "environment_variables": {"a": "b"},
            "command_line_arguments": "ARGUMENT --KEY TESTOVERRIDE",
        }

        log_configuration_override_details = {
            "log_group_id": "TestOverrideLogGroupId",
        }

        step_override_details = [
            {
                "step_name": "TestPipelineStepOne",
                "step_configuration_details": {
                    "maximum_runtime_in_minutes": 200,
                    "environment_variables": {"1": "2"},
                    "command_line_arguments": "argument --key testoverride",
                },
            }
        ]

        free_form_tags = {
            "OverrideKey": "OverrideValue",
        }

        defined_tags = {
            "OverrideKey": "OverrideValue",
        }

        system_tags = {
            "OverrideKey": "OverrideValue",
        }

        original_pipeline_details = pipeline_one._Pipeline__pipeline_details()
        pipeline_one._Pipeline__override_configurations(
            pipeline_details=original_pipeline_details,
            display_name=display_name,
            project_id=project_id,
            compartment_id=compartment_id,
            configuration_override_details=configuration_override_details,
            log_configuration_override_details=log_configuration_override_details,
            step_override_details=step_override_details,
            free_form_tags=free_form_tags,
            defined_tags=defined_tags,
            system_tags=system_tags,
        )

        assert original_pipeline_details["displayName"] == "TestOverrideName"
        assert original_pipeline_details["compartmentId"] == "TestOverrideCompartmentId"
        assert original_pipeline_details["projectId"] == "TestOverrideProjectId"
        assert original_pipeline_details["configurationOverrideDetails"] == {
            "maximumRuntimeInMinutes": 30,
            "type": "DEFAULT",
            "environmentVariables": {"a": "b"},
            "commandLineArguments": "ARGUMENT --KEY TESTOVERRIDE",
        }
        assert original_pipeline_details["logConfigurationOverrideDetails"] == {
            "logGroupId": "TestOverrideLogGroupId",
            "enableLogging": True,
            "enableAutoLogCreation": True,
        }
        assert original_pipeline_details["stepOverrideDetails"] == [
            {
                "stepName": "TestPipelineStepOne",
                "stepConfigurationDetails": {
                    "maximumRuntimeInMinutes": 200,
                    "environmentVariables": {"1": "2"},
                    "commandLineArguments": "argument --key testoverride",
                },
            }
        ]
        assert original_pipeline_details["freeformTags"] == {
            "OverrideKey": "OverrideValue",
        }
        assert original_pipeline_details["definedTags"] == {
            "OverrideKey": "OverrideValue",
        }
        assert original_pipeline_details["systemTags"] == {
            "OverrideKey": "OverrideValue",
        }

    def test_with_dag_details(self):
        pipeline = Pipeline(
            step_details=[pipeline_step_one, pipeline_step_two],
            dag=["TestPipelineStepOne >> TestPipelineStepTwo"],
        )
        assert pipeline.dag == ["TestPipelineStepOne >> TestPipelineStepTwo"]
        assert pipeline_step_two.depends_on == ["TestPipelineStepOne"]

    @patch.object(DataSciencePipeline, "to_dict")
    def test_build_ads_pipeline(self, mock_to_dict):
        pipeline_response_details = {
            "id": "Test id",
            "timeCreated": "2022-05-02T18:56:47.792000Z",
            "timeUpdated": "2022-05-02T18:56:47.792000Z",
            "createdBy": "Test user id",
            "projectId": "Test project id",
            "compartmentId": "Test compartment id",
            "displayName": "ADSPipeline",
            "description": "This is a test pipeline using ads sdk",
            "configurationDetails": {"type": "DEFAULT"},
            "logConfigurationDetails": {
                "enableLogging": True,
                "enableAutoLogCreation": False,
                "logGroupId": "Test log group id",
                "logId": "Test log id",
            },
            "infrastructureConfigurationDetails": {
                "shapeName": "VM.Standard2.1",
                "blockStorageSizeInGBs": 200,
                "shapeConfigDetails": {"ocpus": 1, "memoryInGBs": 2},
            },
            "stepDetails": [
                {
                    "stepType": "ML_JOB",
                    "stepName": "PipelineStepOne",
                    "description": "This is a test pipeline step one",
                    "stepConfigurationDetails": {
                        "commandLineArguments": "ARGUMENT --KEY VALUE",
                        "environmentVariables": {"ENV": "VALUE"},
                        "maximumRuntimeInMinutes": 20,
                    },
                    "dependsOn": [],
                    "jobId": "Test job id one",
                },
                {
                    "stepType": "CUSTOM_SCRIPT",
                    "stepName": "PipelineStepTwo",
                    "description": "This is a test pipeline step two",
                    "dependsOn": ["PipelineStepOne"],
                    "stepInfrastructureConfigurationDetails": {
                        "shapeName": "VM.Standard2.1",
                        "blockStorageSizeInGBs": 200,
                    },
                    "stepConfigurationDetails": {
                        "environmentVariables": {
                            "CONDA_ENV_TYPE": "service",
                            "CONDA_ENV_SLUG": "tensorflow26_p37_cpu_v2",
                            "JOB_RUN_NOTEBOOK": "basics.ipynb",
                            "JOB_RUN_ENTRYPOINT": "driver_notebook.py",
                            "NOTEBOOK_ENCODING": "utf-8",
                            "OUTPUT_URI": "oci://bucket_name@namespace/path/to/dir",
                        },
                    },
                },
            ],
            "lifecycleState": "ACTIVE",
            "freeformTags": {"TestFreeformTags": "value"},
        }
        mock_to_dict.return_value = pipeline_response_details
        ads_pipeline = DataSciencePipeline().build_ads_pipeline()

        ads_pipeline_step_one = ads_pipeline.step_details[0]
        ads_pipeline_step_two = ads_pipeline.step_details[1]

        assert ads_pipeline.id == "Test id"
        assert ads_pipeline.compartment_id == "Test compartment id"
        assert ads_pipeline.project_id == "Test project id"
        assert ads_pipeline.created_by == "Test user id"
        assert ads_pipeline.log_group_id == "Test log group id"
        assert ads_pipeline.log_id == "Test log id"
        assert ads_pipeline.name == "ADSPipeline"
        assert ads_pipeline.description == "This is a test pipeline using ads sdk"
        assert ads_pipeline.shape_name == "VM.Standard2.1"
        assert ads_pipeline.shape_config_details == {"ocpus": 1, "memoryInGBs": 2}

        assert ads_pipeline_step_one.job_id == "Test job id one"
        assert ads_pipeline_step_one.name == "PipelineStepOne"
        assert ads_pipeline_step_one.kind == "ML_JOB"
        assert ads_pipeline_step_one.description == "This is a test pipeline step one"
        assert ads_pipeline_step_one.maximum_runtime_in_minutes == 20
        assert ads_pipeline_step_one.environment_variable == {"ENV": "VALUE"}
        assert ads_pipeline_step_one.argument == "ARGUMENT --KEY VALUE"
        assert len(ads_pipeline_step_one.depends_on) == 0

        assert ads_pipeline_step_two.name == "PipelineStepTwo"
        assert ads_pipeline_step_two.kind == "CUSTOM_SCRIPT"
        assert ads_pipeline_step_two.description == "This is a test pipeline step two"
        assert ads_pipeline_step_two.depends_on == ["PipelineStepOne"]
        assert (
            ads_pipeline_step_two.infrastructure._spec["blockStorageSizeInGBs"] == 200
        )
        assert ads_pipeline_step_two.infrastructure.shape_name == "VM.Standard2.1"
        assert ads_pipeline_step_two.runtime.kind == "runtime"
        assert ads_pipeline_step_two.runtime.type == "notebook"
        assert ads_pipeline_step_two.runtime.conda == {
            "type": "service",
            "slug": "tensorflow26_p37_cpu_v2",
        }
        assert ads_pipeline_step_two.runtime.notebook_uri == "basics.ipynb"
        assert ads_pipeline_step_two.runtime.notebook_encoding == "utf-8"
        assert (
            ads_pipeline_step_two.runtime.output_uri
            == "oci://bucket_name@namespace/path/to/dir"
        )

    @patch.object(DataSciencePipeline, "create")
    @patch.object(Pipeline, "_Pipeline__create_service_log")
    def test_pipeline_create(self, mock_create_service_log, mock_create):
        pipeline = copy.deepcopy(pipeline_one)
        pipeline.with_enable_service_log(True)
        pipeline.create()
        mock_create_service_log.assert_called()
        mock_create.assert_called()

    @patch.object(DataSciencePipeline, "run")
    @patch.object(Pipeline, "create")
    def test_pipeline_run(self, mock_create, mock_run):
        pipeline = copy.deepcopy(pipeline_one)
        pipeline.create()
        pipeline.run()
        mock_create.assert_called()
        mock_run.assert_called()

    @patch.object(DataSciencePipeline, "delete")
    @patch.object(Pipeline, "create")
    def test_pipeline_delete(self, mock_create, mock_delete):
        pipeline = copy.deepcopy(pipeline_one)
        pipeline.create()
        pipeline.delete(allow_control_chars=True, succeed_on_not_found=False)
        mock_create.assert_called()
        mock_delete.assert_called_with(
            id=pipeline.id,
            operation_kwargs={
                "delete_related_pipeline_runs": True,
                "delete_related_job_runs": True,
                "allow_control_chars": True,
            },
            waiter_kwargs={"max_wait_seconds": 1800, "succeed_on_not_found": False},
        )

    @patch.object(oci.data_science.DataScienceClient, "create_pipeline")
    @patch.object(DataSciencePipeline, "upload_artifact")
    @patch.object(OCIModelMixin, "sync")
    def test_datascience_create(
        self,
        mock_sync,
        mock_create_pipeline,
        mock_upload_artifact,
    ):
        pipeline = copy.deepcopy(pipeline_one)
        data_science_pipeline = DataSciencePipeline()
        data_science_pipeline.create(pipeline.step_details, delete_if_fail=True)
        mock_create_pipeline.assert_called()
        mock_upload_artifact.assert_called()

    @patch.object(
        oci.data_science.DataScienceClientCompositeOperations,
        "delete_pipeline_and_wait_for_state",
    )
    @patch.object(OCIModelMixin, "sync")
    def test_datascience_delete(
        self,
        mock_sync,
        mock_delete_pipeline_and_wait_for_state,
    ):
        data_science_pipeline = DataSciencePipeline()
        data_science_pipeline.id = PIPELINE_OCID
        data_science_pipeline.delete(PIPELINE_OCID)
        mock_delete_pipeline_and_wait_for_state.assert_called_with(
            pipeline_id=PIPELINE_OCID,
            wait_for_states=[
                oci.data_science.models.WorkRequest.STATUS_SUCCEEDED,
                oci.data_science.models.WorkRequest.STATUS_FAILED,
            ],
            operation_kwargs=DEFAULT_OPERATION_KWARGS,
            waiter_kwargs=DEFAULT_WAITER_KWARGS,
        )
        mock_sync.assert_called()

    @patch.object(PipelineRun, "_set_service_logging_resource")
    @patch.object(PipelineRun, "create")
    def test_datascience_run(
        self,
        mock_create,
        mock_set_service_logging_resource,
    ):
        pipeline = copy.deepcopy(pipeline_one)
        data_science_pipeline = DataSciencePipeline()
        service_logging = OCILog(log_type="SERVICE")
        with patch.object(
            PipelineRun,
            "pipeline",
            new_callable=PropertyMock,
            return_value=pipeline,
        ):
            data_science_pipeline.run(
                pipeline._Pipeline__pipeline_details(), service_logging
            )
        mock_set_service_logging_resource.assert_called_with(service_logging)
        mock_create.assert_called()

    def assert_attributes(self, pipeline):
        assert pipeline.data_science_pipeline.compartment_id == "TestCompartmentId"
        assert pipeline.data_science_pipeline.project_id == "TestProjectId"
        assert pipeline.data_science_pipeline.display_name == "TestPipeline"
        assert pipeline.data_science_pipeline.defined_tags == {"key": "value"}
        assert pipeline.data_science_pipeline.freeform_tags == {"key": "value"}

    def test_pipeline_from_yaml(self):
        testpipeline = Pipeline.from_yaml(yaml_string)
        assert isinstance(testpipeline, Pipeline)

    def test_pipeline_to_dict(self):
        assert pipeline_one.to_dict() == {
            "kind": "pipeline",
            "type": "pipeline",
            "spec": {
                "displayName": "TestPipeline",
                "id": "TestId",
                "compartmentId": "TestCompartmentId",
                "projectId": "TestProjectId",
                "logGroupId": "TestLogGroupId",
                "logId": "TestLogId",
                "description": "TestDescription",
                "maximumRuntimeInMinutes": 200,
                "commandLineArguments": "argument --key value",
                "environmentVariables": {"env": "value"},
                "freeformTags": {"key": "value"},
                "definedTags": {"key": "value"},
                "blockStorageSizeInGBs": 200,
                "shapeName": "VM.Standard2.1",
                "shapeConfigDetails": {"ocpus": 1, "memoryInGBs": 2},
                "stepDetails": [
                    {
                        "kind": "dataScienceJob",
                        "spec": {
                            "name": "TestPipelineStepOne",
                            "jobId": "TestJobIdOne",
                            "description": "Test description one",
                        },
                    },
                    {
                        "kind": "customScript",
                        "spec": {
                            "name": "TestPipelineStepTwo",
                            "runtime": {
                                "kind": "runtime",
                                "type": "notebook",
                                "spec": {
                                    "notebookEncoding": "utf-8",
                                    "notebookPathURI": "https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
                                    "conda": {
                                        "type": "service",
                                        "slug": "tensorflow26_p37_cpu_v2",
                                    },
                                    "outputUri": "oci://bucket_name@namespace/path/to/dir",
                                },
                            },
                            "infrastructure": {
                                "kind": "infrastructure",
                                "spec": {
                                    "blockStorageSize": 200,
                                    "shapeName": "VM.Standard3.Flex",
                                    "shapeConfigDetails": {
                                        "memoryInGBs": 16,
                                        "ocpus": 1,
                                    },
                                },
                            },
                            "description": "Test description two",
                        },
                    },
                ],
                "dag": ["TestPipelineStepOne >> TestPipelineStepTwo"],
            },
        }

    def test_pipeline_from_yaml_to_dict(self):
        testpipeline = Pipeline.from_yaml(yaml_string)

        assert testpipeline.to_dict() == {
            "kind": "pipeline",
            "type": "pipeline",
            "spec": {
                "compartmentId": "TestCompartmentId",
                "dag": ["TestPipelineStepOne >> TestPipelineStepTwo"],
                "description": "This is a test pipeline using ads sdk",
                "displayName": "TestPipeline",
                "freeformTags": {"key": "value"},
                "id": "TestId",
                "logGroupId": "TestLogGroupId",
                "logId": "TestLogId",
                "projectId": "TestProjectId",
                "stepConfigDetails": {
                    "ocpus": 1,
                    "memoryInGBs": 2,
                },
                "stepDetails": [
                    {
                        "kind": "dataScienceJob",
                        "spec": {
                            "name": "TestPipelineStepOne",
                            "jobId": "TestJobIdOne",
                            "description": "Test description one",
                        },
                    },
                    {
                        "kind": "customScript",
                        "spec": {
                            "name": "TestPipelineStepTwo",
                            "runtime": {
                                "kind": "runtime",
                                "type": "notebook",
                                "spec": {
                                    "conda": {
                                        "slug": "tensorflow26_p37_cpu_v2",
                                        "type": "service",
                                    },
                                    "notebookEncoding": "utf-8",
                                    "notebookPathURI": "https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
                                    "outputUri": "oci://bucket_name@namespace/path/to/dir",
                                },
                            },
                            "infrastructure": {
                                "kind": "infrastructure",
                                "spec": {
                                    "blockStorageSize": 200,
                                    "shapeName": "VM.Standard2.1",
                                },
                            },
                            "description": "Test description two",
                        },
                    },
                ],
            },
        }

    @patch.object(PipelineVisualizer, "render")
    def test_pipeline_show(self, mock_render):
        pipeline_one.show()
        mock_render.assert_called_once()

    def test_pipeline__add_dag_to_node(self):
        dag = ["TestPipelineStepOne >> TestPipelineStepTwo"]
        stepname_to_step_map = {
            "TestPipelineStepOne": pipeline_step_one,
            "TestPipelineStepTwo": pipeline_step_two,
        }
        step_details = Pipeline._add_dag_to_node(dag, stepname_to_step_map)

        assert isinstance(step_details, list)
        assert isinstance(step_details[0], PipelineStep)
        assert isinstance(step_details[1], PipelineStep)
        assert stepname_to_step_map["TestPipelineStepTwo"].depends_on == [
            "TestPipelineStepOne"
        ]

    def test_pipeline__add_dag_to_node_fail(self):
        dag = ["'xxx >> yyy >> 7'"]
        stepname_to_step_map = {
            "TestPipelineStepOne": pipeline_step_one,
            "TestPipelineStepTwo": pipeline_step_two,
        }
        with pytest.raises(ValueError):
            Pipeline._add_dag_to_node(dag, stepname_to_step_map)

    def test_set_service_logging_resource(self):
        pipeline = copy.deepcopy(pipeline_one)
        service_logging = OCILog(log_type="SERVICE")
        pipeline._Pipeline__set_service_logging_resource(service_logging)
        assert pipeline.service_logging == service_logging

    def test_status(self):
        pipeline = copy.deepcopy(pipeline_one)
        pipeline.data_science_pipeline = DataSciencePipeline()
        pipeline.data_science_pipeline.lifecycle_state = (
            Pipeline.LIFECYCLE_STATE_DELETED
        )
        assert pipeline.status == Pipeline.LIFECYCLE_STATE_DELETED

    @patch(
        "ads.pipeline.ads_pipeline.COMPARTMENT_OCID", PIPELINE_PAYLOAD["compartment_id"]
    )
    @patch("ads.pipeline.ads_pipeline.PROJECT_OCID", PIPELINE_PAYLOAD["project_id"])
    @patch(
        "ads.pipeline.ads_pipeline.NB_SESSION_OCID", PIPELINE_PAYLOAD["nb_session_ocid"]
    )
    @patch.object(DSCNotebookSession, "from_ocid")
    def test__load_default_properties(self, mock_from_ocid):
        pipeline = copy.deepcopy(pipeline_one)
        mock_from_ocid.return_value = self.nb_session
        assert pipeline._load_default_properties() == self.mock_default_properties
        mock_from_ocid.assert_called_with(PIPELINE_PAYLOAD["nb_session_ocid"])

    @patch(
        "ads.pipeline.ads_pipeline.COMPARTMENT_OCID", PIPELINE_PAYLOAD["compartment_id"]
    )
    @patch("ads.pipeline.ads_pipeline.PROJECT_OCID", PIPELINE_PAYLOAD["project_id"])
    @patch(
        "ads.pipeline.ads_pipeline.NB_SESSION_OCID", PIPELINE_PAYLOAD["nb_session_ocid"]
    )
    @patch.object(DSCNotebookSession, "from_ocid")
    def test__load_default_properties_fail(self, mock_from_ocid):
        pipeline = copy.deepcopy(pipeline_one)
        mock_from_ocid.side_effect = ValueError("Something went wrong")
        assert pipeline._load_default_properties() == {
            Pipeline.CONST_COMPARTMENT_ID: PIPELINE_PAYLOAD["compartment_id"],
            Pipeline.CONST_PROJECT_ID: PIPELINE_PAYLOAD["project_id"],
        }
        mock_from_ocid.assert_called_with(PIPELINE_PAYLOAD["nb_session_ocid"])


class TestDataSciencePipeline:
    @patch.object(OCIModelMixin, "from_ocid")
    def test_from_ocid(self, mock_from_ocid):
        """Tests getting a Model Version Set by OCID."""
        DataSciencePipeline.from_ocid(PIPELINE_OCID)
        mock_from_ocid.assert_called_with(PIPELINE_OCID)


class TestPipeline:
    mock_active_ds_pipeline = DataSciencePipeline(lifecycle_state="ACTIVE")

    def setup_class(cls):
        cls.mock_date = datetime.datetime(2022, 7, 1)

    def setup_method(self):
        self.mock_default_properties = {
            "compartment_id": PIPELINE_PAYLOAD["compartment_id"],
            "project_id": PIPELINE_PAYLOAD["project_id"],
            "display_name": f"pipeline-{self.mock_date.strftime('%Y%m%d-%H%M')}",
        }

        self.payload = {
            **self.mock_default_properties,
            "step_details": pipeline_details["stepDetails"],
        }

    @patch.object(Pipeline, "_populate_step_artifact_content")
    def test_list(
        self,
        mock_populate_step_artifact_content,
    ):
        """Tests listing pipelines in a given compartment."""
        datascience_pipeline_list = [
            DataSciencePipeline(**self.payload),
            DataSciencePipeline(**self.payload),
            DataSciencePipeline(**self.payload),
        ]
        expected_result = [
            item.build_ads_pipeline() for item in datascience_pipeline_list
        ]
        with patch.object(DataSciencePipeline, "list_resource") as mock_list_resource:
            mock_list_resource.return_value = datascience_pipeline_list
            test_result = Pipeline.list(compartment_id="test_compartment_id")
            for i, item in enumerate(test_result):
                assert item.to_dict() == expected_result[i].to_dict()
            mock_list_resource.assert_called_with("test_compartment_id")

    @patch.object(Pipeline, "_populate_step_artifact_content")
    def test_from_ocid(
        self,
        mock_populate_step_artifact_content,
    ):
        """Ensures that Pipeline can be built form OCID."""
        datascience_pipeline = DataSciencePipeline(**self.payload)
        with patch.object(
            DataSciencePipeline, "from_ocid", return_value=datascience_pipeline
        ) as mock_from_ocid:
            test_pipeline = Pipeline.from_ocid(PIPELINE_OCID)
            mock_from_ocid.assert_called_with(PIPELINE_OCID)
            mock_populate_step_artifact_content.assert_called()

            expected_result = batch_convert_case(
                self.mock_default_properties, to_fmt="camel"
            )
            expected_result["stepDetails"] = [
                pipeline_step_one.to_dict(),
                pipeline_step_two.to_dict(),
            ]

            assert (
                test_pipeline.to_dict()["spec"]
                == datascience_pipeline.build_ads_pipeline().to_dict()["spec"]
            )

    @patch.object(Pipeline, "from_ocid")
    def test_from_id(self, mock_from_ocid):
        """Ensures that Pipeline can be built form OCID."""
        Pipeline.from_id(PIPELINE_OCID)
        mock_from_ocid.assert_called_with(PIPELINE_OCID)

    @patch.object(PipelineRun, "list")
    def test_run_list(self, mock_list):
        """Tests getting a list of runs of the pipeline."""
        pipeline_one.run_list()
        mock_list.assert_called_with(
            compartment_id=pipeline_one.compartment_id, pipeline_id=pipeline_one.id
        )

    @patch.object(
        oci.data_science.DataScienceClient,
        "get_step_artifact_content",
        return_value=oci.response.Response(
            status=200, headers={}, data=None, request=None
        ),
    )
    @patch.object(os, "mkdir")
    @patch.object(os.path, "exists", return_value=True)
    @patch.object(
        OCIModelMixin,
        "sync",
        return_value=mock_active_ds_pipeline,
    )
    def test_download_artifacts_override(
        self,
        mock_sync,
        mock_mkdir,
        mock_exists,
        mock_get_step_artifact_content,
    ):
        pipeline = copy.deepcopy(pipeline_one)
        pipeline.data_science_pipeline = DataSciencePipeline(**self.payload)
        temp_dir = tempfile.TemporaryDirectory()
        pipeline.download(to_dir=temp_dir.name, override_if_exists=True)

        mock_sync.assert_called()
        mock_mkdir.assert_called()
        mock_exists.assert_called()
        mock_get_step_artifact_content.assert_called_with(
            pipeline.id, "TestPipelineStepTwo"
        )

    @patch.object(
        oci.data_science.DataScienceClient,
        "get_step_artifact_content",
        return_value=oci.response.Response(
            status=200, headers={}, data=None, request=None
        ),
    )
    @patch.object(os.path, "exists", return_value=True)
    @patch.object(
        OCIModelMixin,
        "sync",
        return_value=mock_active_ds_pipeline,
    )
    def test_download_artifacts_not_override(
        self, mock_sync, mock_exists, mock_get_step_artifact_content
    ):
        pipeline = copy.deepcopy(pipeline_one)
        pipeline.data_science_pipeline = DataSciencePipeline(**self.payload)
        temp_dir = tempfile.TemporaryDirectory()
        pipeline.download(to_dir=temp_dir.name, override_if_exists=False)

        mock_sync.assert_called()
        mock_exists.assert_called()
        mock_get_step_artifact_content.assert_not_called()


class TestPipelineName:
    random_seed = 42

    @pytest.mark.parametrize(
        "name, name_in_spec, name_in_kwargs, expected",
        [
            (None, {}, {}, "delightful-donkey"),  # auto-generated name
            (None, {}, {"display_name": "kwarg_display_name"}, "kwarg_display_name"),
            (None, {"display_name": "spec_display_name"}, {}, "spec_display_name"),
            ("name", {}, {}, "name"),
            (
                None,
                {"display_name": "spec_display_name"},
                {"display_name": "kwarg_display_name"},
                "kwarg_display_name",
            ),
            ("name", {}, {"display_name": "kwarg_display_name"}, "name"),
            ("name", {"display_name": "spec_display_name"}, {}, "name"),
            (
                "name",
                {"display_name": "spec_display_name"},
                {"display_name": "kwarg_display_name"},
                "name",
            ),
        ],
    )
    def test_pipeline_name(self, name, name_in_spec, name_in_kwargs, expected):
        """Tests variations of names that provided or not provided into parameters of Pipeline() and with
        combination of .with_name() method. When no name provided in any of parameters validate, that
        a randomly generated easy to remember name will be generated, like 'strange-spider-2022-08-17-23:55.02'.
        With random seed = 42 name has to be delightful-donkey-timestamp.
        """
        random.seed(self.random_seed)
        p = Pipeline(name=name, spec=name_in_spec, **name_in_kwargs)
        # autogenerated name with current timestamp at the end - check expected value "in" autogenerated name
        if not name and not name_in_spec and not name_in_kwargs:
            assert expected in p.name
        else:
            assert p.name == expected

        # when .with_name() method, expected name specified in .with_name(), no matter what in Pipeline() params
        p_with_name = Pipeline(
            name=name, spec=name_in_spec, **name_in_kwargs
        ).with_name("another_name")
        assert p_with_name.name == "another_name"
