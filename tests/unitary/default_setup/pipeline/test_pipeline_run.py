#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from datetime import datetime
from unittest import SkipTest
from unittest.mock import MagicMock, patch

import oci
from ads.common.oci_mixin import OCIModelMixin
import pytest
from ads.common.oci_logging import ConsolidatedLog, OCILog
from ads.common.utils import batch_convert_case
from ads.jobs.builders.runtimes.python_runtime import NotebookRuntime

try:
    from ads.pipeline.ads_pipeline import Pipeline
    from ads.pipeline.ads_pipeline_run import PipelineRun, LogNotConfiguredError
    from ads.pipeline.ads_pipeline_step import PipelineStep
    from ads.pipeline.builders.infrastructure.custom_script import CustomScriptStep
    from ads.pipeline.visualizer.base import StepStatus
except (ImportError, AttributeError) as e:
    raise SkipTest("OCI MLPipeline is not available. Skipping the MLPipeline tests.")


PIPELINE_PAYLOAD = {
    "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
    "project_id": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
}
PIPELINE_OCID = "ocid.xxx.datasciencepipeline.<unique_ocid>"
PIPELINE_RUN_OCID = "ocid.xxx.datasciencepipelinerun.<unique_ocid>"
PIPELINE_RUN_LOG_DETAILS = {
    "log_id": "ocid1.log.oc1.xxx.<unique_ocid>",
    "log_group_id": "ocid1.loggroup.oc1.xxx.<unique_ocid>",
}

OCI_LOG_DETAILS = {
    "id": "ocid1.log.oc1.xxx.<unique_ocid>",
    "log_group_id": "ocid1.loggroup.oc1.xxx.<unique_ocid>",
    "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
}

CUSTOM_SCRIPT_STEP_RUN_PAYLOAD = {
    "step_type": "CUSTOM_SCRIPT",
    "time_started": "2023-01-19T22:15:26.154000Z",
    "time_finished": "2023-01-19T22:19:22.389000Z",
    "step_name": "TestPipelineStepScriptRuntimeOne",
    "lifecycle_state": "SUCCEEDED",
    "lifecycle_details": "",
}

ML_JOB_STEP_RUN_PAYLOAD = {
    "step_type": "ML_JOB",
    "time_started": "2023-01-19T22:23:39.501000Z",
    "time_finished": "2023-01-19T22:27:25.214000Z",
    "step_name": "TestPipelineMLStepOne",
    "lifecycle_state": "SUCCEEDED",
    "lifecycle_details": "",
    "job_run_id": "ocid1.datasciencejobrun.oc1.iad.<unique_ocid>",
}

ML_JOB_STEP_RUN_NO_ID_PAYLOAD = {
    "step_type": "ML_JOB",
    "time_started": "2023-01-19T22:22:37.273000Z",
    "time_finished": "",
    "step_name": "TestPipelineMLStepTwo",
    "lifecycle_state": "",
    "lifecycle_details": "",
    "job_run_id": "",
}


class TestPipelineRun:
    def setup_method(self):
        self.pipeline_run_details = {
            "project_id": "test project id",
            "compartment_id": "test compartment id",
            "pipeline_id": "test pipeline id",
            "display_name": "test display name",
            "freeform_tags": {
                "key": "value",
            },
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
            .with_shape_name("VM.Standard2.1")
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

        self.pipeline_one = (
            Pipeline("TestPipeline")
            .with_id("TestId")
            .with_compartment_id("TestCompartmentId")
            .with_project_id("TestProjectId")
            .with_log_group_id("TestLogGroupId")
            .with_log_id("TestLogId")
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

        self.pipeline_one.with_dag(["TestPipelineStepOne >> TestPipelineStepTwo"])

    def test_list(self):
        """Tests listing pipeline runs for a given pipeline."""
        with patch.object(PipelineRun, "list_resource") as mock_list_resource:
            PipelineRun.list(
                compartment_id=PIPELINE_PAYLOAD["compartment_id"],
                pipeline_id=PIPELINE_OCID,
            )
            mock_list_resource.assert_called_with(
                PIPELINE_PAYLOAD["compartment_id"], pipeline_id=PIPELINE_OCID
            )

    @patch.object(oci.data_science.DataScienceClient, "create_pipeline_run")
    def test_create(self, mock_create_pipeline_run):
        pipeline_run = PipelineRun()
        expected_result = pipeline_run.to_oci_model(
            oci.data_science.models.CreatePipelineRunDetails
        )
        mock_create_pipeline_run.return_value = MagicMock(
            data=oci.data_science.models.PipelineRun(**self.pipeline_run_details)
        )
        with patch.object(
            PipelineRun, "load_properties_from_env"
        ) as mock_load_properties_from_env:
            actual_pipeline_run = pipeline_run.create()
            mock_create_pipeline_run.assert_called_with(expected_result)
            mock_load_properties_from_env.assert_called()

            assert actual_pipeline_run.to_dict() == batch_convert_case(
                spec=self.pipeline_run_details, to_fmt="camel"
            )

    @patch.object(oci.data_science.DataScienceClient, "cancel_pipeline_run")
    def test_cancel(self, mock_cancel_pipeline_run):
        pipeline_run = PipelineRun()
        pipeline_run.id = PIPELINE_RUN_OCID
        pipeline_run.lifecycle_state = PipelineRun.LIFECYCLE_STATE_CANCELED
        with patch.object(PipelineRun, "sync", return_value=pipeline_run) as mock_sync:
            pipeline_run.cancel()
            mock_sync.assert_called()
            mock_cancel_pipeline_run.assert_called_with(PIPELINE_RUN_OCID)

    @patch.object(PipelineRun, "sync")
    @patch.object(
        oci.data_science.DataScienceClientCompositeOperations,
        "delete_pipeline_run_and_wait_for_state",
    )
    def test_delete(
        self,
        mock_delete_pipeline_run_and_wait_for_state,
        mock_sync,
    ):
        pipeline_run = PipelineRun()
        pipeline_run.id = PIPELINE_RUN_OCID
        pipeline_run.lifecycle_state = PipelineRun.LIFECYCLE_STATE_DELETED
        pipeline_run.delete(allow_control_chars=True)
        mock_delete_pipeline_run_and_wait_for_state.assert_called_with(
            pipeline_run_id=PIPELINE_RUN_OCID,
            wait_for_states=[PipelineRun.LIFECYCLE_STATE_DELETED],
            operation_kwargs={
                "delete_related_job_runs": True,
                "allow_control_chars": True,
            },
            waiter_kwargs={"max_wait_seconds": 1800},
        )
        mock_sync.assert_called()

    @patch.object(Pipeline, "from_ocid")
    def test_pipeline(self, mock_from_ocid):
        pipeline_run = PipelineRun()
        pipeline_run.pipeline_id = PIPELINE_OCID
        pipeline_run.pipeline
        mock_from_ocid.assert_called_with(PIPELINE_OCID)

    def test_sync_step_details(self):
        pipeline_run = PipelineRun()
        pipeline_run._pipeline = self.pipeline_one

        step_config_override_details = (
            oci.data_science.models.PipelineStepConfigurationDetails(
                maximum_runtime_in_minutes=10,
                environment_variables={"TestOverrideKey": "TestOverrideValue"},
                command_line_arguments="Test Override Argument",
            )
        )
        pipeline_run.step_override_details = [
            oci.data_science.models.PipelineStepOverrideDetails(
                step_name="TestPipelineStepOne",
                step_configuration_details=step_config_override_details,
            )
        ]

        pipeline_run._sync_step_details()
        actual_step_details = pipeline_run._pipeline.step_details
        expected_step_1 = PipelineStep.from_dict(
            {
                "kind": "dataScienceJob",
                "spec": {
                    "name": "TestPipelineStepOne",
                    "jobId": "TestJobIdOne",
                    "description": "Test description one",
                    "dependsOn": [],
                    "stepConfigurationDetails": {
                        "commandLineArguments": "Test Override Argument",
                        "environmentVariables": {
                            "TestOverrideKey": "TestOverrideValue",
                        },
                        "maximumRuntimeInMinutes": 10,
                    },
                },
            }
        )

        expected_step_2 = PipelineStep.from_dict(
            {
                "kind": "customScript",
                "spec": {
                    "name": "TestPipelineStepTwo",
                    "description": "Test description two",
                    "infrastructure": {
                        "kind": "infrastructure",
                        "spec": {
                            "blockStorageSize": 200,
                            "shapeName": "VM.Standard2.1",
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
                    "dependsOn": ["TestPipelineStepOne"],
                },
            }
        )

        assert actual_step_details[0].to_dict() == expected_step_1.to_dict()
        assert actual_step_details[0].argument == expected_step_1.argument
        assert (
            actual_step_details[0].environment_variable
            == expected_step_1.environment_variable
        )
        assert (
            actual_step_details[0].maximum_runtime_in_minutes
            == expected_step_1.maximum_runtime_in_minutes
        )
        assert actual_step_details[1].to_dict() == expected_step_2.to_dict()

    @patch.object(Pipeline, "from_ocid")
    def test_step_details(self, mock_pipeline_from_ocid):
        mock_pipeline_from_ocid.return_value = self.pipeline_one

        pipeline_run = PipelineRun()
        step_config_override_details = (
            oci.data_science.models.PipelineStepConfigurationDetails(
                maximum_runtime_in_minutes=10,
                environment_variables={"TestOverrideKey": "TestOverrideValue"},
                command_line_arguments="Test Override Argument",
            )
        )
        pipeline_run.step_override_details = [
            oci.data_science.models.PipelineStepOverrideDetails(
                step_name="TestPipelineStepOne",
                step_configuration_details=step_config_override_details,
            )
        ]
        actual_step_details = pipeline_run.pipeline.step_details
        mock_pipeline_from_ocid.assert_called()

        assert len(actual_step_details) == 2
        assert actual_step_details[0].job_id == "TestJobIdOne"
        assert actual_step_details[0].name == "TestPipelineStepOne"
        assert actual_step_details[0].description == "Test description one"
        assert actual_step_details[0].argument == "Test Override Argument"
        assert actual_step_details[0].environment_variable == {
            "TestOverrideKey": "TestOverrideValue"
        }
        assert actual_step_details[0].maximum_runtime_in_minutes == 10

        assert actual_step_details[1].name == "TestPipelineStepTwo"
        assert actual_step_details[1].description == "Test description two"
        assert actual_step_details[1].infrastructure.block_storage_size == 200
        assert actual_step_details[1].infrastructure.shape_name == "VM.Standard2.1"
        assert actual_step_details[1].runtime.conda == {
            "type": "service",
            "slug": "tensorflow26_p37_cpu_v2",
        }
        assert actual_step_details[1].runtime.notebook_encoding == "utf-8"
        assert (
            actual_step_details[1].runtime.notebook_uri
            == "https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb"
        )
        assert (
            actual_step_details[1].runtime.output_uri
            == "oci://bucket_name@namespace/path/to/dir"
        )
        assert actual_step_details[1].runtime.type == "notebook"

    def test_set_service_logging_resource(self):
        pipeline_run = PipelineRun()
        service_logging = OCILog()
        pipeline_run._set_service_logging_resource(service_logging)
        assert pipeline_run.service_logging != None

    @pytest.mark.parametrize(
        "step_status, expected_result",
        [(StepStatus.FAILED, True), (StepStatus.WAITING, False)],
    )
    @patch.object(PipelineRun, "sync")
    def test_stop_condition(self, mock_sync, step_status, expected_result):
        pipeline_run = PipelineRun()
        pipeline_run.lifecycle_state = step_status
        test_result = pipeline_run._stop_condition()
        mock_sync.assert_called()
        assert test_result == expected_result

    @patch.object(PipelineRun, "sync", return_value=None)
    @patch.object(ConsolidatedLog, "stream", return_value=1)
    def test_stream_log(self, mock_stream, mock_sync):
        pipeline_run = PipelineRun()
        pipeline_run.id = PIPELINE_RUN_OCID
        pipeline_run.log_details = oci.data_science.models.PipelineRunLogDetails()
        custom_script_step = oci.data_science.models.pipeline_custom_script_step_run.PipelineCustomScriptStepRun(
            **CUSTOM_SCRIPT_STEP_RUN_PAYLOAD
        )
        ml_job_step = (
            oci.data_science.models.pipeline_ml_job_step_run.PipelineMLJobStepRun(
                **ML_JOB_STEP_RUN_PAYLOAD
            )
        )
        ml_job_step_without_job_run_id = (
            oci.data_science.models.pipeline_ml_job_step_run.PipelineMLJobStepRun(
                **ML_JOB_STEP_RUN_NO_ID_PAYLOAD
            )
        )
        pipeline_run.step_runs = [
            custom_script_step,
            ml_job_step,
            ml_job_step_without_job_run_id,
        ]
        pipeline_run._PipelineRun__stream_log(ConsolidatedLog(OCILog()))
        mock_stream.assert_not_called()

        pipeline_run._PipelineRun__stream_log(ConsolidatedLog(OCILog()))
        mock_stream.assert_not_called()

        pipeline_run.log_details = oci.data_science.models.PipelineRunLogDetails(
            **PIPELINE_RUN_LOG_DETAILS
        )
        pipeline_run.time_accepted = datetime.now()
        pipeline_run._PipelineRun__stream_log(
            ConsolidatedLog(OCILog()),
            [custom_script_step.step_name, ml_job_step.step_name],
        )
        mock_stream.assert_called_with(
            interval=3,
            stop_condition=pipeline_run._stop_condition,
            time_start=pipeline_run.time_accepted,
            log_filter=f"(source = '*{PIPELINE_RUN_OCID}' AND (subject = '{custom_script_step.step_name}' OR subject = '{ml_job_step.step_name}')) OR source = '*{ml_job_step.job_run_id}'",
        )

    def test_to_yaml(self):
        create_pipeline_run_details = {
            "compartmentId": "TestCompartmentId",
            "displayName": "TestPipeline",
            "pipelineId": "TestPipelineId",
            "configurationOverrideDetails": {
                "maximumRuntimeInMinutes": 30,
                "type": "DEFAULT",
                "environmentVariables": {"a": "b"},
                "commandLineArguments": "ARGUMENT --KEY TESTOVERRIDE",
            },
            "logConfigurationOverrideDetails": {
                "logGroupId": "TestOverrideLogGroupId",
            },
            "stepOverrideDetails": [
                {
                    "stepName": "TestPipelineStepOne",
                    "stepConfigurationDetails": {
                        "maximumRuntimeInMinutes": 200,
                        "environmentVariables": {"1": "2"},
                        "commandLineArguments": "argument --key testoverride",
                    },
                }
            ],
            "freeformTags": {
                "Key": "Value",
            },
        }
        pipeline_run = PipelineRun(**create_pipeline_run_details)
        actual_yaml = pipeline_run.to_yaml()

        expected_yaml_string = """compartmentId: TestCompartmentId
configurationOverrideDetails:
  commandLineArguments: ARGUMENT --KEY TESTOVERRIDE
  environmentVariables:
    a: b
  maximumRuntimeInMinutes: 30
  type: DEFAULT
displayName: TestPipeline
freeformTags:
  Key: Value
logConfigurationOverrideDetails:
  logGroupId: TestOverrideLogGroupId
pipelineId: TestPipelineId
stepOverrideDetails:
- stepConfigurationDetails:
    commandLineArguments: argument --key testoverride
    environmentVariables:
      '1': '2'
    maximumRuntimeInMinutes: 200
  stepName: TestPipelineStepOne
"""

        assert actual_yaml == expected_yaml_string

    def test_status(self):
        pipeline_run = PipelineRun()
        pipeline_run.id = PIPELINE_RUN_OCID
        pipeline_run.lifecycle_state = PipelineRun.LIFECYCLE_STATE_CANCELED
        with patch.object(PipelineRun, "sync", return_value=pipeline_run) as mock_sync:
            test_status = pipeline_run.status
            mock_sync.assert_called()
            assert test_status == PipelineRun.LIFECYCLE_STATE_CANCELED

    @patch.object(PipelineRun, "_stop_condition", return_value=True)
    def test_logs_custom(self, mock_stop_condition):
        pipeline_run = PipelineRun()
        pipeline_run.id = PIPELINE_RUN_OCID
        pipeline_run.compartment_id = PIPELINE_PAYLOAD["compartment_id"]

        with pytest.raises(
            ValueError,
            match="Parameter log_type should be either custom_log, service_log or None.",
        ):
            pipeline_run.logs(log_type="unrecognized_log_type")

        pipeline_run.log_details = oci.data_science.models.PipelineRunLogDetails(
            log_group_id=""
        )
        with pytest.raises(
            LogNotConfiguredError,
            match="Log group OCID is not specified for this pipeline. Call with_log_group_id to add it.",
        ):
            pipeline_run.logs(log_type="custom_log")

        pipeline_run.log_details = oci.data_science.models.PipelineRunLogDetails(
            **PIPELINE_RUN_LOG_DETAILS
        )

        with patch.object(OCIModelMixin, "deserialize") as mock_deserialize:

            mock_deserialize.return_value = oci.logging.models.Log(**OCI_LOG_DETAILS)

            consolidated_log = pipeline_run.logs(log_type="custom_log")
            mock_stop_condition.assert_called()
            custom_log = consolidated_log.logging_instance[0]
            assert len(consolidated_log.logging_instance) == 1
            assert isinstance(consolidated_log, ConsolidatedLog)
            assert isinstance(custom_log, OCILog)

            assert custom_log.id == PIPELINE_RUN_LOG_DETAILS["log_id"]
            assert custom_log.log_group_id == PIPELINE_RUN_LOG_DETAILS["log_group_id"]
            assert custom_log.compartment_id == PIPELINE_PAYLOAD["compartment_id"]
            assert custom_log.annotation == "custom"

    @patch.object(PipelineRun, "_get_service_logging")
    def test_logs_service(self, mock_get_service_logging):
        pipeline_run = PipelineRun()
        with pytest.raises(
            LogNotConfiguredError,
            match="Pipeline log is not configured. Make sure log group id is added.",
        ):
            pipeline_run.logs(log_type="service_log")

        pipeline_run.log_details = oci.data_science.models.PipelineRunLogDetails(
            log_group_id=""
        )
        with pytest.raises(
            LogNotConfiguredError,
            match="Log group OCID is not specified for this pipeline. Call with_log_group_id to add it.",
        ):
            pipeline_run.logs(log_type="service_log")

        with patch.object(OCIModelMixin, "deserialize") as mock_deserialize:

            mock_deserialize.return_value = oci.logging.models.Log(**OCI_LOG_DETAILS)
            pipeline_run.log_details = oci.data_science.models.PipelineRunLogDetails(
                **PIPELINE_RUN_LOG_DETAILS
            )
            mock_get_service_logging.return_value = OCILog(
                id=PIPELINE_RUN_LOG_DETAILS["log_id"],
                log_group_id=PIPELINE_RUN_LOG_DETAILS["log_group_id"],
                compartment_id=PIPELINE_PAYLOAD["compartment_id"],
                annotation="service",
            )

            consolidated_log = pipeline_run.logs(log_type="service_log")
            service_log = consolidated_log.logging_instance[0]
            mock_get_service_logging.assert_called()

            assert len(consolidated_log.logging_instance) == 1
            assert isinstance(consolidated_log, ConsolidatedLog)
            assert isinstance(service_log, OCILog)

            assert service_log.id == PIPELINE_RUN_LOG_DETAILS["log_id"]
            assert service_log.log_group_id == PIPELINE_RUN_LOG_DETAILS["log_group_id"]
            assert service_log.compartment_id == PIPELINE_PAYLOAD["compartment_id"]
            assert service_log.annotation == "service"

    @patch.object(PipelineRun, "_get_service_logging")
    @patch.object(PipelineRun, "_stop_condition", return_value=True)
    @patch.object(OCIModelMixin, "sync")
    def test_logs_both(self, mock_sync, mock_stop_condition, mock_get_service_logging):
        with patch.object(OCIModelMixin, "deserialize") as mock_deserialize:

            mock_deserialize.return_value = oci.logging.models.Log(**OCI_LOG_DETAILS)

            pipeline_run = PipelineRun()
            pipeline_run.id = PIPELINE_RUN_OCID
            pipeline_run.compartment_id = PIPELINE_PAYLOAD["compartment_id"]

            pipeline_run.log_details = oci.data_science.models.PipelineRunLogDetails(
                **PIPELINE_RUN_LOG_DETAILS
            )

            mock_get_service_logging.return_value = OCILog(
                id=PIPELINE_RUN_LOG_DETAILS["log_id"],
                log_group_id=PIPELINE_RUN_LOG_DETAILS["log_group_id"],
                compartment_id=PIPELINE_PAYLOAD["compartment_id"],
                annotation="service",
            )

            consolidated_log = pipeline_run.logs()
            mock_stop_condition.assert_called()
            mock_get_service_logging.assert_called()
            custom_log = consolidated_log.logging_instance[0]
            service_log = consolidated_log.logging_instance[1]
            assert len(consolidated_log.logging_instance) == 2
            assert custom_log.id == PIPELINE_RUN_LOG_DETAILS["log_id"]
            assert custom_log.log_group_id == PIPELINE_RUN_LOG_DETAILS["log_group_id"]
            assert custom_log.compartment_id == PIPELINE_PAYLOAD["compartment_id"]
            assert custom_log.annotation == "custom"

            assert service_log.id == PIPELINE_RUN_LOG_DETAILS["log_id"]
            assert service_log.log_group_id == PIPELINE_RUN_LOG_DETAILS["log_group_id"]
            assert service_log.compartment_id == PIPELINE_PAYLOAD["compartment_id"]
            assert service_log.annotation == "service"

    @patch.object(PipelineRun, "_stop_condition", return_value=True)
    def test_custom_logging(self, mock_stop_condition):
        with patch.object(OCIModelMixin, "deserialize") as mock_deserialize:

            mock_deserialize.return_value = oci.logging.models.Log(**OCI_LOG_DETAILS)
            pipeline_run = PipelineRun()
            pipeline_run.log_details = oci.data_science.models.PipelineRunLogDetails(
                **PIPELINE_RUN_LOG_DETAILS
            )
            pipeline_run.id = PIPELINE_RUN_OCID
            pipeline_run.compartment_id = PIPELINE_PAYLOAD["compartment_id"]
            custom_logging = pipeline_run.custom_logging

            mock_stop_condition.assert_called()
            assert isinstance(custom_logging, OCILog)
            assert custom_logging.id == PIPELINE_RUN_LOG_DETAILS["log_id"]
            assert (
                custom_logging.log_group_id == PIPELINE_RUN_LOG_DETAILS["log_group_id"]
            )
            assert custom_logging.compartment_id == PIPELINE_PAYLOAD["compartment_id"]
            assert custom_logging.annotation == "custom"

    @patch.object(PipelineRun, "_get_service_logging")
    def test_service_logging(self, mock_get_service_logging):
        with patch.object(OCIModelMixin, "deserialize") as mock_deserialize:

            mock_deserialize.return_value = oci.logging.models.Log(**OCI_LOG_DETAILS)

            pipeline_run = PipelineRun()
            pipeline_run.log_details = oci.data_science.models.PipelineRunLogDetails(
                log_group_id=PIPELINE_RUN_LOG_DETAILS["log_group_id"]
            )

            mock_get_service_logging.return_value = OCILog(
                id=PIPELINE_RUN_LOG_DETAILS["log_id"],
                log_group_id=PIPELINE_RUN_LOG_DETAILS["log_group_id"],
                compartment_id=PIPELINE_PAYLOAD["compartment_id"],
                annotation="service",
            )

            service_logging = pipeline_run.service_logging
            assert isinstance(service_logging, OCILog)
            assert service_logging.id == PIPELINE_RUN_LOG_DETAILS["log_id"]
            assert (
                service_logging.log_group_id == PIPELINE_RUN_LOG_DETAILS["log_group_id"]
            )
            assert service_logging.compartment_id == PIPELINE_PAYLOAD["compartment_id"]
            assert service_logging.annotation == "service"

    @patch.object(PipelineRun, "_search_service_logs")
    def test_get_service_logging(self, mock_search_service_logs):
        with patch.object(OCIModelMixin, "deserialize") as mock_deserialize:

            mock_deserialize.return_value = oci.logging.models.Log(**OCI_LOG_DETAILS)
            pipeline_run = PipelineRun()
            pipeline_run.id = PIPELINE_RUN_OCID
            pipeline_run.compartment_id = PIPELINE_PAYLOAD["compartment_id"]
            pipeline_run.log_details = oci.data_science.models.PipelineRunLogDetails(
                log_group_id=PIPELINE_RUN_LOG_DETAILS["log_group_id"]
            )

            mock_search_service_logs.return_value = [
                oci.logging.models.log_summary.LogSummary(
                    id=PIPELINE_RUN_LOG_DETAILS["log_id"]
                )
            ]

            service_logging = pipeline_run._get_service_logging()

            assert isinstance(service_logging, OCILog)
            assert service_logging.id == PIPELINE_RUN_LOG_DETAILS["log_id"]
            assert (
                service_logging.log_group_id == PIPELINE_RUN_LOG_DETAILS["log_group_id"]
            )
            assert service_logging.compartment_id == PIPELINE_PAYLOAD["compartment_id"]
            assert service_logging.annotation == "service"

    @patch.object(PipelineRun, "_search_service_logs")
    def test_get_service_logging_fail(self, mock_search_service_logs):
        pipeline_run = PipelineRun()
        pipeline_run.id = PIPELINE_RUN_OCID
        pipeline_run.compartment_id = PIPELINE_PAYLOAD["compartment_id"]
        pipeline_run.log_details = oci.data_science.models.PipelineRunLogDetails(
            log_group_id=PIPELINE_RUN_LOG_DETAILS["log_group_id"]
        )

        mock_search_service_logs.return_value = []

        with pytest.raises(
            LogNotConfiguredError, match="Service log is not configured for pipeline."
        ):
            service_logging = pipeline_run._get_service_logging()
            assert service_logging == None

        mock_search_service_logs.return_value = None

        with pytest.raises(
            LogNotConfiguredError, match="Service log is not configured for pipeline."
        ):
            service_logging = pipeline_run._get_service_logging()
            assert service_logging == None

    @patch.object(PipelineRun, "_PipelineRun__stream_log")
    @patch.object(PipelineRun, "logs")
    def test_watch_service_log(self, mock_logs, mock_stream_log):
        pipeline_run = PipelineRun()
        mock_logs.return_value = ConsolidatedLog(OCILog(log_type="SERVICE"))

        pipeline_run.watch(log_type="service_log")
        mock_logs.called_with(log_type="service_log")
        mock_stream_log.assert_called_with(mock_logs.return_value, [], 3, "service_log")

    @patch.object(PipelineRun, "_PipelineRun__stream_log")
    @patch.object(PipelineRun, "logs")
    def test_watch_custom_log(self, mock_logs, mock_stream_log):
        pipeline_run = PipelineRun()
        pipeline_run.id = PIPELINE_RUN_OCID
        mock_logs.return_value = ConsolidatedLog(OCILog(log_type="CUSTOM"))

        pipeline_run.watch(log_type="custom_log")
        mock_logs.called_with(log_type="custom_log")
        mock_stream_log.assert_called_with(mock_logs.return_value, [], 3, "custom_log")

    @patch.object(PipelineRun, "_PipelineRun__stream_log")
    @patch.object(PipelineRun, "logs")
    def test_watch_both_log(self, mock_logs, mock_stream_log):
        pipeline_run = PipelineRun()
        pipeline_run.id = PIPELINE_RUN_OCID
        mock_logs.return_value = ConsolidatedLog(
            OCILog(log_type="CUSTOM"),
            OCILog(log_type="SERVICE"),
        )

        pipeline_run.watch()
        mock_logs.called_with(log_type=None)
        mock_stream_log.assert_called_with(mock_logs.return_value, [], 3, None)

    def test_build_filter_expression(self):
        custom_script_step = oci.data_science.models.pipeline_custom_script_step_run.PipelineCustomScriptStepRun(
            **CUSTOM_SCRIPT_STEP_RUN_PAYLOAD
        )
        ml_job_step = (
            oci.data_science.models.pipeline_ml_job_step_run.PipelineMLJobStepRun(
                **ML_JOB_STEP_RUN_PAYLOAD
            )
        )
        ml_job_step_without_job_run_id = (
            oci.data_science.models.pipeline_ml_job_step_run.PipelineMLJobStepRun(
                **ML_JOB_STEP_RUN_NO_ID_PAYLOAD
            )
        )
        pipeline_run = PipelineRun()
        pipeline_run.id = PIPELINE_RUN_OCID
        pipeline_run.step_runs = [
            custom_script_step,
            ml_job_step,
            ml_job_step_without_job_run_id,
        ]

        consolidated_log_expression = pipeline_run._build_filter_expression()

        assert (
            consolidated_log_expression
            == f"(source = '*{PIPELINE_RUN_OCID}' AND (subject = '{custom_script_step.step_name}' OR subject = '{ml_job_step.step_name}')) OR source = '*{ml_job_step.job_run_id}'"
        )

        ml_job_log_expression = pipeline_run._build_filter_expression(
            steps=["TestPipelineMLStepOne"]
        )

        assert (
            ml_job_log_expression
            == f"(source = '*{PIPELINE_RUN_OCID}' AND (subject = '{ml_job_step.step_name}')) OR source = '*{ml_job_step.job_run_id}'"
        )

        ml_job_no_id_log_expression = pipeline_run._build_filter_expression(
            steps=["TestPipelineMLStepTwo"]
        )

        assert ml_job_no_id_log_expression == ""
