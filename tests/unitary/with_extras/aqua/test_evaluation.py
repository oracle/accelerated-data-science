#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import asyncio
import base64
import copy
import json
import os
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, call, PropertyMock, patch

import oci
from parameterized import parameterized

from ads.aqua import utils
from ads.aqua.base import AquaApp
from ads.aqua.evaluation import (
    AquaEvalMetrics,
    AquaEvalReport,
    AquaEvaluationApp,
    AquaEvaluationSummary,
)
from ads.aqua.exception import (
    AquaFileNotFoundError,
    AquaMissingKeyError,
    AquaRuntimeError,
)
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.jobs.ads_job import DataScienceJob, DataScienceJobRun, Job
from ads.aqua.utils import EVALUATION_REPORT_JSON, EVALUATION_REPORT_MD
from ads.model import DataScienceModel
from ads.model.model_version_set import ModelVersionSet

null = None


class TestDataset:
    """Mock service response."""

    model_provenance_object = {
        "git_branch": null,
        "git_commit": null,
        "repository_url": null,
        "script_dir": null,
        "training_id": "ocid1.datasciencejobrun.oc1.iad.<OCID>",
        "training_script": null,
    }

    job_run_object = {
        "compartment_id": "ocid1.compartment.oc1..<OCID>",
        "created_by": "ocid1.datasciencenotebooksession.oc1.iad.<OCID>",
        "defined_tags": {},
        "display_name": "mistral-samsum-evaluation-run-2024-02-10-16:59.57",
        "freeform_tags": {
            "aqua_evaluation": "",
            "evaluation_model_id": "ocid1.datasciencemodel.oc1.iad.<OCID>",
        },
        "id": "ocid1.datasciencejobrun.oc1.iad.<OCID>",
        "job_configuration_override_details": null,
        "job_environment_configuration_override_details": null,
        "job_id": "ocid1.datasciencejob.oc1.iad.<OCID>",
        "job_infrastructure_configuration_details": oci.data_science.models.StandaloneJobInfrastructureConfigurationDetails(
            **{
                "block_storage_size_in_gbs": 512,
                "job_shape_config_details": {"memory_in_gbs": 16.0, "ocpus": 1.0},
                "shape_name": "VM.Standard.E3.Flex",
                "subnet_id": "ocid1.subnet.oc1.iad.<OCID>",
            }
        ),
        "job_log_configuration_override_details": null,
        "job_storage_mount_configuration_details_list": [],
        "lifecycle_details": "",
        "lifecycle_state": "SUCCEEDED",
        "log_details": oci.data_science.models.JobRunLogDetails(
            **{
                "log_group_id": "ocid1.loggroup.oc1.iad.<OCID>",
                "log_id": "ocid1.log.oc1.iad.<OCID>",
            }
        ),
        "project_id": "ocid1.datascienceproject.oc1.iad.<OCID>",
        "time_accepted": "2024-02-10T16:59:58.405000+00:00",
        "time_finished": "2024-02-10T17:18:18.078000+00:00",
        "time_started": "2024-02-10T17:16:33.547000+00:00",
    }

    # job_run_object without logging configurated
    job_run_object_no_logging = {
        "compartment_id": "ocid1.compartment.oc1..<OCID>",
        "created_by": "ocid1.datasciencenotebooksession.oc1.iad.<OCID>",
        "defined_tags": {},
        "display_name": "mistral-samsum-evaluation-run-2024-02-10-16:59.57",
        "freeform_tags": {
            "aqua_evaluation": "",
            "evaluation_model_id": "ocid1.datasciencemodel.oc1.iad.<OCID>",
        },
        "id": "ocid1.datasciencejobrun.oc1.iad.<OCID>",
        "job_configuration_override_details": null,
        "job_environment_configuration_override_details": null,
        "job_id": "ocid1.datasciencejob.oc1.iad.<OCID>",
        "job_infrastructure_configuration_details": oci.data_science.models.StandaloneJobInfrastructureConfigurationDetails(
            **{
                "block_storage_size_in_gbs": 512,
                "job_shape_config_details": {"memory_in_gbs": 16.0, "ocpus": 1.0},
                "shape_name": "VM.Standard.E3.Flex",
                "subnet_id": "ocid1.subnet.oc1.iad.<OCID>",
            }
        ),
        "job_log_configuration_override_details": null,
        "job_storage_mount_configuration_details_list": [],
        "lifecycle_details": "",
        "lifecycle_state": "SUCCEEDED",
        "log_details": null,
        "project_id": "ocid1.datascienceproject.oc1.iad.<OCID>",
        "time_accepted": "2024-02-10T16:59:58.405000+00:00",
        "time_finished": "2024-02-10T17:18:18.078000+00:00",
        "time_started": "2024-02-10T17:16:33.547000+00:00",
    }

    resource_summary_object_loggroup = [
        {
            "additional_details": {"description": null},
            "availability_domain": null,
            "compartment_id": "ocid1.compartment.oc1..<ocid>",
            "defined_tags": {},
            "display_name": "testlogs",
            "freeform_tags": {},
            "identifier": "ocid1.loggroup.oc1.iad.<ocid>",
            "identity_context": {},
            "lifecycle_state": "ACTIVE",
            "resource_type": "LogGroup",
            "search_context": null,
            "system_tags": {},
            "time_created": "2021-05-07T12:51:36.705000+00:00",
        }
    ]

    resource_summary_object_log = [
        {
            "additional_details": {"logGroupId": "ocid1.loggroup.oc1.iad.<ocid>"},
            "availability_domain": null,
            "compartment_id": "ocid1.compartment.oc1..<ocid>",
            "defined_tags": {},
            "display_name": "mylog",
            "freeform_tags": {},
            "identifier": "ocid1.log.oc1.iad.<ocid>",
            "identity_context": {"logGroupId": "ocid1.loggroup.oc1.iad.<ocid>"},
            "lifecycle_state": "ACTIVE",
            "resource_type": "Log",
            "search_context": null,
            "system_tags": {},
            "time_created": "2022-05-11T05:10:29.918000+00:00",
        }
    ]

    resource_summary_object_jobrun = [
        {
            "additional_details": {},
            "availability_domain": null,
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "defined_tags": {},
            "display_name": "mistral-samsum-evaluation-run-2024-02-10-16:59.57",
            "freeform_tags": {
                "aqua_evaluation": "",
                "evaluation_model_id": "ocid1.datasciencemodel.oc1.iad.<OCID>",
            },
            "identifier": "ocid1.datasciencejobrun.oc1.iad.<OCID>",
            "identity_context": {},
            "lifecycle_state": "SUCCEEDED",
            "resource_type": "DataScienceJobRun",
            "search_context": null,
            "system_tags": {},
            "time_created": "2024-02-10T16:59:58.405000+00:00",
        }
    ]

    # A successful evaluation that has all expected fields
    resource_summary_object_eval = [
        {
            "additional_details": {
                "createdBy": "ocid1.datasciencenotebooksession.oc1.iad.<OCID>",
                "description": "This is a dummy MC entry for evaluation.",
                "metadata": [
                    {
                        "category": "other",
                        "description": "",
                        "key": "evaluation_source",
                        "value": "ocid1.datasciencemodeldeployment.oc1.iad.<OCID>",
                    },
                    {
                        "category": "other",
                        "description": "",
                        "key": "evaluation_source_name",
                        "value": "Model 1",
                    },
                    {
                        "category": "other",
                        "description": "",
                        "key": "evaluation_job_id",
                        "value": "ocid1.datasciencejob.oc1.iad.<OCID>",
                    },
                    {
                        "category": "other",
                        "description": "",
                        "key": "evaluation_job_run_id",
                        "value": "ocid1.datasciencejobrun.oc1.iad.<OCID>",
                    },
                    {
                        "category": "other",
                        "description": "",
                        "key": "evaluation_output_path",
                        "value": "oci://mybucket@mytenancy/output",
                    },
                    {
                        "category": "other",
                        "description": "The all necessary permissions to update the record have been granted.",
                        "key": "aqua_evaluate_test_write_access",
                        "value": "PASSED",
                    },
                    {
                        "category": null,
                        "description": null,
                        "key": "Framework",
                        "value": "Other",
                    },
                    {
                        "category": null,
                        "description": null,
                        "key": "Hyperparameters",
                        "value": '{"model_params": {"max_tokens": 500, "top_p": 1, "top_k": 50, "temperature": 0.7, "presence_penalty": 0, "frequency_penalty": 0, "stop": [], "shape": "VM.Standard.E3.Flex", "dataset_path": "oci://mybucket@mytenancy/data.jsonl", "report_path": "oci://mybucket@mytenancy/report"}}',
                    },
                    {
                        "category": null,
                        "description": null,
                        "key": "UseCaseType",
                        "value": "Other",
                    },
                    {
                        "category": null,
                        "description": null,
                        "key": "ArtifactTestResults",
                        "value": null,
                    },
                    {
                        "category": null,
                        "description": null,
                        "key": "FrameworkVersion",
                        "value": null,
                    },
                    {
                        "category": null,
                        "description": null,
                        "key": "Algorithm",
                        "value": null,
                    },
                ],
                "modelVersionSetId": "ocid1.datasciencemodelversionset.oc1.iad.<OCID>",
                "modelVersionSetName": "Experiment 1",
                "projectId": "ocid1.datascienceproject.oc1.iad.<OCID>",
                "versionLabel": null,
            },
            "availability_domain": null,
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "defined_tags": {},
            "display_name": "Eval2",
            "freeform_tags": {
                "aqua_evaluation": "aqua_evaluation",
            },
            "identifier": "ocid1.datasciencemodel.oc1.iad.<OCID>",
            "identity_context": {},
            "lifecycle_state": "ACTIVE",
            "resource_type": "DataScienceModel",
            "search_context": null,
            "system_tags": {},
            "time_created": "2024-02-15T20:18:34.225000+00:00",
        },
    ]

    # evaluation missing metadata
    resource_summary_object_eval_missing_fields = [
        {
            "additional_details": {
                "createdBy": "ocid1.datasciencenotebooksession.oc1.iad.<OCID>",
                "description": "This is a dummy MC entry for evaluation.",
                "metadata": [
                    {
                        "category": null,
                        "description": null,
                        "key": "Framework",
                        "value": "Other",
                    },
                    {
                        "category": null,
                        "description": null,
                        "key": "Hyperparameters",
                        "value": '{"model_params": {"max_tokens": 500, "top_p": 1, "top_k": 50, "temperature": 0.7, "presence_penalty": 0, "frequency_penalty": 0, "stop": [], "shape": "VM.Standard.E3.Flex", "dataset_path": "oci://mybucket@mytenancy/data.jsonl", "report_path": "oci://mybucket@mytenancy/report"}}',
                    },
                    {
                        "category": null,
                        "description": null,
                        "key": "UseCaseType",
                        "value": "Other",
                    },
                    {
                        "category": null,
                        "description": null,
                        "key": "ArtifactTestResults",
                        "value": null,
                    },
                    {
                        "category": null,
                        "description": null,
                        "key": "FrameworkVersion",
                        "value": null,
                    },
                    {
                        "category": null,
                        "description": null,
                        "key": "Algorithm",
                        "value": null,
                    },
                ],
                "modelVersionSetId": "ocid1.datasciencemodelversionset.oc1.iad.<OCID>",
                "modelVersionSetName": "Experiment 1",
                "projectId": "ocid1.datascienceproject.oc1.iad.<OCID>",
                "versionLabel": null,
            },
            "availability_domain": null,
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "defined_tags": {},
            "display_name": "Eval2",
            "freeform_tags": {
                "aqua_evaluation": "aqua_evaluation",
            },
            "identifier": "ocid1.datasciencemodel.oc1.iad.<OCID>",
            "identity_context": {},
            "lifecycle_state": "ACTIVE",
            "resource_type": "DataScienceModel",
            "search_context": null,
            "system_tags": {},
            "time_created": "2024-02-15T20:18:34.225000+00:00",
        }
    ]
    COMPARTMENT_ID = "ocid1.compartment.oc1..<UNIQUE_OCID>"
    EVAL_ID = "ocid1.datasciencemodel.oc1.iad.<OCID>"
    INVALID_EVAL_ID = "ocid1.datasciencemodel.oc1.phx.<OCID>"


class TestAquaEvaluation(unittest.TestCase):
    """Contains unittests for TestAquaEvaluationApp."""

    @classmethod
    def setUpClass(cls):
        utils.is_valid_ocid = MagicMock(return_value=True)

    def setUp(self):
        self.app = AquaEvaluationApp()

        self._query_resources = utils.query_resources
        self._query_resource = utils.query_resource
        utils.query_resources = MagicMock(
            return_value=[
                oci.resource_search.models.ResourceSummary(**item)
                for item in TestDataset.resource_summary_object_eval
            ]
        )
        utils.query_resource = MagicMock(side_effect=self.side_effect_func)

    def tearDown(self) -> None:
        utils.query_resources = self._query_resources
        utils.query_resource = self._query_resource
        self.app._report_cache.clear()
        self.app._metrics_cache.clear()
        self.app._eval_cache.clear()

    @staticmethod
    def side_effect_func(*args, **kwargs):
        if args[0].startswith("ocid1.datasciencemodeldeployment"):
            return oci.resource_search.models.ResourceSummary(
                **TestDataset.resource_summary_object_md[0]
            )
        elif args[0].startswith("ocid1.datasciencemodel"):
            return oci.resource_search.models.ResourceSummary(
                **TestDataset.resource_summary_object_eval[0]
            )
        elif args[0].startswith("ocid1.datasciencejob"):
            return oci.resource_search.models.ResourceSummary(
                **TestDataset.resource_summary_object_jobrun[0]
            )
        elif args[0].startswith("ocid1.loggroup"):
            return oci.resource_search.models.ResourceSummary(
                **TestDataset.resource_summary_object_loggroup[0]
            )
        elif args[0].startswith("ocid1.log"):
            return oci.resource_search.models.ResourceSummary(
                **TestDataset.resource_summary_object_log[0]
            )

    def print_expected_response(self, response, method):
        """Used for visualized expected output."""

        print(f"############ Expected Response For {method} ############")
        if isinstance(response, list):
            response = {"data": response}
        response = json.loads(json.dumps(response, default=AquaAPIhandler.serialize))
        print(response)

    def assert_payload(self, response, response_type):
        """Checks each field is not empty."""

        attributes = response_type.__annotations__.keys()
        rdict = asdict(response)

        for attr in attributes:
            if attr == "lifecycle_details":  # can be empty when jobrun is succeed
                continue
            assert rdict.get(attr), f"{attr} is empty"

    @patch.object(Job, "run")
    @patch("ads.jobs.ads_job.Job.name", new_callable=PropertyMock)
    @patch("ads.jobs.ads_job.Job.id", new_callable=PropertyMock)
    @patch.object(Job, "create")
    @patch("ads.aqua.evaluation.get_container_image")
    @patch.object(DataScienceModel, "create")
    @patch.object(ModelVersionSet, "create")
    @patch.object(DataScienceModel, "from_id")
    def test_create_evaluation(
        self,
        mock_from_id,
        mock_mvs_create,
        mock_ds_model_create,
        mock_get_container_image,
        mock_job_create,
        mock_job_id,
        mock_job_name,
        mock_job_run
    ):
        foundation_model = MagicMock()
        foundation_model.display_name = "test_foundation_model"
        mock_from_id.return_value = foundation_model

        experiment = MagicMock()
        experiment.id = "test_experiment_id"
        mock_mvs_create.return_value = experiment

        evaluation_model = MagicMock()
        evaluation_model.id = "test_evaluation_model_id"
        evaluation_model.display_name = "test_evaluation_model_name"

        oci_dsc_model = MagicMock()
        oci_dsc_model.time_created = "test_time_created"
        evaluation_model.dsc_model = oci_dsc_model
        mock_ds_model_create.return_value = evaluation_model

        mock_get_container_image.return_value = "test_container_image"

        mock_job_id.return_value = "test_evaluation_job_id"
        mock_job_name.return_value = "test_evaluation_job_name"

        evaluation_job_run = MagicMock()
        evaluation_job_run.id = "test_evaluation_job_run_id"
        evaluation_job_run.lifecycle_details = "Job run artifact execution in progress."
        evaluation_job_run.lifecycle_state = "IN_PROGRESS"
        mock_job_run.return_value = evaluation_job_run

        self.app.ds_client.update_model = MagicMock()
        self.app.ds_client.update_model_provenance = MagicMock()

        create_aqua_evaluation_details = dict(
            evaluation_source_id="ocid1.datasciencemodel.oc1.iad.<OCID>",
            evaluation_name="test_evaluation_name",
            dataset_path="oci://dataset_bucket@namespace/prefix/dataset.jsonl",
            report_path="oci://report_bucket@namespace/prefix/",
            model_parameters={},
            shape_name="VM.Standard.E3.Flex",
            block_storage_size=1,
            experiment_name="test_experiment_name",
            memory_in_gbs=1,
            ocpus=1,
        )
        aqua_evaluation_summary = self.app.create(
            **create_aqua_evaluation_details
        )

        assert asdict(aqua_evaluation_summary) == {
            'console_url': f'https://cloud.oracle.com/data-science/models/{evaluation_model.id}?region={self.app.region}',
            'experiment': {
                'id': f'{experiment.id}',
                'name': 'test_experiment_name',
                'url': f'https://cloud.oracle.com/data-science/model-version-sets/{experiment.id}?region={self.app.region}'
            },
            'id': f'{evaluation_model.id}',
            'job': {
                'id': f'{mock_job_id.return_value}',
                'name': f'{mock_job_name.return_value}',
                'url': f'https://cloud.oracle.com/data-science/jobs/{mock_job_id.return_value}?region={self.app.region}'
            },
            'lifecycle_details': f'{evaluation_job_run.lifecycle_details}',
            'lifecycle_state': f'{evaluation_job_run.lifecycle_state}',
            'name': f'{evaluation_model.display_name}',
            'parameters': {
                'dataset_path': '',
                'frequency_penalty': 0.0,
                'max_tokens': '',
                'presence_penalty': 0.0,
                'report_path': '',
                'shape': '',
                'stop': [],
                'temperature': '',
                'top_k': '',
                'top_p': ''
            },
            'source': {
                'id': 'ocid1.datasciencemodel.oc1.iad.<OCID>',
                'name': f'{foundation_model.display_name}',
                'url': f'https://cloud.oracle.com/data-science/models/ocid1.datasciencemodel.oc1.iad.<OCID>?region={self.app.region}'
            },
            'tags': {
                'aqua_evaluation': 'aqua_evaluation',
                'evaluation_experiment_id': f'{experiment.id}',
                'evaluation_job_id': f'{mock_job_id.return_value}',
                'evaluation_source': 'ocid1.datasciencemodel.oc1.iad.<OCID>'
            },
            'time_created': f'{oci_dsc_model.time_created}'
        }

    @parameterized.expand(
        [
            (
                TestDataset.model_provenance_object,
                TestDataset.job_run_object,
            ),
            (
                TestDataset.model_provenance_object,
                TestDataset.job_run_object_no_logging,
            ),
        ]
    )
    def test_get(self, mock_get_model_provenance_response, mock_get_job_run_response):
        """Tests get evaluation details successfully."""
        self.app.ds_client.get_model_provenance = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=oci.data_science.models.ModelProvenance(
                    **mock_get_model_provenance_response
                ),
            )
        )
        self.app.ds_client.get_job_run = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=oci.data_science.models.JobRun(**mock_get_job_run_response),
            )
        )

        response = self.app.get(TestDataset.EVAL_ID)

        self.app.ds_client.get_job_run.assert_called_with(
            TestDataset.model_provenance_object.get("training_id")
        )
        self.print_expected_response(response, "GET EVALUATION")
        # check status return correctly
        assert response.lifecycle_state == "SUCCEEDED"

    @patch.object(utils, "query_resource")
    def test_get_fail(self, mock_query_resource):
        """Tests get evaluation details failed because of invalid eval id."""
        mock_query_resource.return_value = None
        self.app.ds_client.get_model_provenance = MagicMock()
        with self.assertRaises(AquaRuntimeError) as context:
            self.app.get(TestDataset.INVALID_EVAL_ID)

        self.assertTrue(
            f"Failed to retrieve evalution {TestDataset.INVALID_EVAL_ID}."
            in str(context.exception)
        )

    def test_list(self):
        """Tests listing evaluations successfully."""
        self.app._prefetch_resources = MagicMock(return_value={})
        response = self.app.list(TestDataset.COMPARTMENT_ID)
        utils.query_resources.assert_called_with(
            compartment_id=TestDataset.COMPARTMENT_ID,
            resource_type="datasciencemodel",
            tag_list=["aqua_evaluation"],
        )

        utils.query_resource.assert_called_with(
            TestDataset.resource_summary_object_jobrun[0].get("identifier"),
            return_all=False,
        )

        assert len(response) == 1
        self.print_expected_response(response, "LIST EVALUATIONS")
        self.assert_payload(response[0], AquaEvaluationSummary)

        # check status return correctly
        assert response[0].lifecycle_state == "SUCCEEDED"
        assert self.app._eval_cache.currsize == 1

        response1 = self.app.list(TestDataset.COMPARTMENT_ID)
        self.assert_payload(response1[0], AquaEvaluationSummary)
        assert response1 == [
            self.app._eval_cache.get(
                TestDataset.resource_summary_object_eval[0].get("identifier"),
            )
        ]

    @patch.object(DataScienceModel, "download_artifact")
    @patch.object(DataScienceModel, "from_id")
    @patch("tempfile.TemporaryDirectory")
    def test_download_report(
        self, mock_TemporaryDirectory, mock_dsc_model_from_id, mock_download_artifact
    ):
        """Tests download evaluation report successfully."""
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mock_temp_path = os.path.join(curr_dir, "test_data/valid_eval_artifact")
        mock_TemporaryDirectory.return_value.__enter__.return_value = mock_temp_path
        response = self.app.download_report(TestDataset.EVAL_ID)

        mock_dsc_model_from_id.assert_called_with(TestDataset.EVAL_ID)
        self.print_expected_response(response, "DOWNLOAD REPORT")
        self.assert_payload(response, AquaEvalReport)
        read_content = base64.b64decode(response.content)
        assert (
            read_content == b"This is a sample evaluation report.html.\n"
        ), read_content
        assert self.app._report_cache.currsize == 1

        # download from cache
        response1 = self.app.download_report(TestDataset.EVAL_ID)
        assert self.app._report_cache.get(TestDataset.EVAL_ID) == response1

    @patch.object(DataScienceModel, "from_id")
    @patch.object(DataScienceJob, "from_id")
    @patch.object(AquaEvaluationApp, "_delete_job_and_model")
    def test_delete_evaluation(
        self, mock_del_job_model_func, mock_dsc_job, mock_dsc_model_from_id
    ):
        mock_dsc_model_from_id.return_value = MagicMock(
            provenance_data={
                "training_id": TestDataset.model_provenance_object.get("training_id"),
            }
        )
        mock_dsc_job.return_value = MagicMock(lifecycle_state="ACCEPTED")
        mock_del_job_model_func.return_value = None
        result = self.app.delete(TestDataset.EVAL_ID)
        assert result["id"] == TestDataset.EVAL_ID
        assert result["lifecycle_state"] == "DELETING"

        mock_del_job_model_func.assert_called_once()

    @patch.object(DataScienceModel, "from_id")
    @patch.object(DataScienceJobRun, "from_ocid")
    @patch.object(AquaEvaluationApp, "_cancel_job_run")
    def test_cancel_evaluation(
        self, mock_cancel_jr_func, mock_dsc_job_run, mock_dsc_model_from_id
    ):
        mock_dsc_model_from_id.return_value = MagicMock(
            provenance_data={
                "training_id": TestDataset.model_provenance_object.get("training_id")
            }
        )
        mock_dsc_job_run.return_value = MagicMock(lifecycle_state="ACCEPTED")
        mock_cancel_jr_func.return_value = None

        result = self.app.cancel(TestDataset.EVAL_ID)

        assert result["id"] == TestDataset.EVAL_ID
        assert result["lifecycle_state"] == "CANCELING"
        mock_cancel_jr_func.assert_called_once()

    @parameterized.expand(
        [
            (None, AquaRuntimeError),
            # (
            #     DataScienceModel(),
            #     AquaMissingKeyError,
            # ),
        ]
    )
    @patch.object(DataScienceModel, "from_id")
    def test_cancel_and_delete_failed(
        self, dsc_model, expect_error, mock_dsc_model_from_id
    ):
        """Tests error raised in cancel/delete."""

        mock_dsc_model_from_id.return_value = dsc_model

        with self.assertRaises(expect_error):
            self.app.cancel(TestDataset.INVALID_EVAL_ID)

        with self.assertRaises(expect_error):
            self.app.delete(TestDataset.INVALID_EVAL_ID)

    @patch.object(DataScienceModel, "download_artifact")
    @patch.object(DataScienceModel, "from_id")
    @patch("tempfile.TemporaryDirectory")
    def test_load_metrics(
        self, mock_TemporaryDirectory, mock_dsc_model_from_id, mock_download_artifact
    ):
        """Tests loading evaluation metrics successfully."""
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mock_temp_path = os.path.join(curr_dir, "test_data/valid_eval_artifact")
        mock_TemporaryDirectory.return_value.__enter__.return_value = mock_temp_path
        response = self.app.load_metrics(TestDataset.EVAL_ID)

        mock_dsc_model_from_id.assert_called_with(TestDataset.EVAL_ID)
        self.print_expected_response(response, "LOAD METRICS")
        self.assert_payload(response, AquaEvalMetrics)
        assert len(response.metric_results) == 1
        assert len(response.metric_summary_result) == 1
        assert self.app._metrics_cache.currsize == 1

        response1 = self.app.load_metrics(TestDataset.EVAL_ID)
        assert response1 == self.app._metrics_cache.get(TestDataset.EVAL_ID)

    @patch.object(DataScienceModel, "download_artifact")
    @patch.object(DataScienceModel, "from_id")
    def test_load_metrics_fail(self, mock_dsc_model_from_id, mock_download_artifact):
        """Tests loading metrics failed when missing `report.md` in artifact."""
        with self.assertRaises(AquaFileNotFoundError):
            self.app.load_metrics(TestDataset.EVAL_ID)

    @patch.object(DataScienceModel, "download_artifact")
    @patch.object(DataScienceModel, "from_id")
    def test_load_metrics_missing_json(
        self, mock_dsc_model_from_id, mock_download_artifact
    ):
        """Tests loading metrics still work when missing `report.json` in artifact."""

        def read_side_effect_func(*args):
            if args[2] == EVALUATION_REPORT_MD:
                return b"This is a test"
            elif args[2] == EVALUATION_REPORT_JSON:
                return AquaFileNotFoundError(f"{args[2]} not found.")

        self.app._read_from_artifact = MagicMock(side_effect=read_side_effect_func)

        response = self.app.load_metrics(TestDataset.EVAL_ID)
        assert len(response.metric_results) == 0

    def test_get_status(self):
        """Tests getting evaluation status successfully."""
        self.app.ds_client.get_model_provenance = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=oci.data_science.models.ModelProvenance(
                    **TestDataset.model_provenance_object
                ),
            )
        )
        self.app.ds_client.get_job_run = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=oci.data_science.models.JobRun(**TestDataset.job_run_object),
            )
        )
        response = self.app.get_status(TestDataset.EVAL_ID)
        self.print_expected_response(response, "GET STATUS")
        assert response.get("lifecycle_state") == "SUCCEEDED"

    @parameterized.expand(
        [
            (
                dict(
                    return_value=oci.response.Response(
                        status=200, request=MagicMock(), headers=MagicMock(), data=None
                    )
                ),
                "SUCCEEDED",
            ),
            (
                dict(
                    side_effect=oci.exceptions.ServiceError(
                        status=404, code=None, message="error test msg", headers={}
                    )
                ),
                "FAILED",
            ),
        ]
    )
    def test_get_status_when_missing_jobrun(
        self, mock_head_model_artifact_response, expected_output
    ):
        """Tests getting evaluation status correctly when missing jobrun association."""
        self.app.ds_client.get_model_provenance = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=oci.data_science.models.ModelProvenance(
                    **TestDataset.model_provenance_object
                ),
            )
        )
        self.app._fetch_jobrun = MagicMock(return_value=None)

        self.app.ds_client.head_model_artifact = MagicMock(
            side_effect=mock_head_model_artifact_response.get("side_effect", None),
            return_value=mock_head_model_artifact_response.get("return_value", None),
        )

        response = self.app.get_status(TestDataset.EVAL_ID)
        self.app.ds_client.head_model_artifact.assert_called_with(
            model_id=TestDataset.EVAL_ID
        )
        actual_status = response.get("lifecycle_state")
        assert (
            actual_status == expected_output
        ), f"expected status is {expected_output}, actual status is {actual_status}"

    @patch.object(utils, "query_resource")
    def test_get_status_failed(self, mock_query_resource):
        """Tests when no correct evaluation found."""
        mock_query_resource.return_value = None
        self.app.ds_client.get_model_provenance = MagicMock()
        with self.assertRaises(AquaRuntimeError) as context:
            self.app.get_status(TestDataset.INVALID_EVAL_ID)

        self.assertTrue(
            f"Failed to retrieve evalution {TestDataset.INVALID_EVAL_ID}."
            in str(context.exception)
        )

    @parameterized.expand(
        [
            (
                "Job run artifact execution failed with exit code 16",
                "Validation errors in the evaluation config. Exit code: 16.",
            ),
            ("Job completed successfully.", "Job completed successfully."),
        ]
    )
    def test_extract_job_lifecycle_details(self, input, expect_output):
        """Tests extracting job lifecycle details."""
        msg = self.app._extract_job_lifecycle_details(input)
        assert msg == expect_output, msg

    def test_get_supported_metrics(self):
        """Tests getting a list of supported metrics for evaluation.
        This method currently hardcoded the return value.
        """
        from .utils import SupportMetricsFormat as metric_schema
        from .utils import check

        response = self.app.get_supported_metrics()
        assert isinstance(response, list)
        for metric in response:
            assert check(metric_schema, metric)

    def test_load_evaluation_config(self):
        """Tests loading default config for evaluation.
        This method currently hardcoded the return value.
        """
        from .utils import EvaluationConfigFormat as config_schema
        from .utils import check

        response = self.app.load_evaluation_config(eval_id=TestDataset.EVAL_ID)
        assert isinstance(response, dict)
        assert check(config_schema, response)


class TestAquaEvaluationList(unittest.TestCase):
    """More tests related to list function."""

    def setUp(self):
        self.app = AquaEvaluationApp()

    def tearDown(self) -> None:
        self.app._eval_cache.clear()

    @patch("ads.aqua.utils.query_resource")
    @patch("ads.aqua.utils.query_resources")
    def test_skipping_fetch_jobrun(self, mock_query_resources, mock_query_resource):
        """Tests listing evalution."""
        mock_query_resources.return_value = [
            oci.resource_search.models.ResourceSummary(**item)
            for item in TestDataset.resource_summary_object_eval
        ]
        self.app._prefetch_resources = MagicMock(
            return_value={
                TestDataset.resource_summary_object_jobrun[0].get(
                    "identifier"
                ): oci.resource_search.models.ResourceSummary(
                    **TestDataset.resource_summary_object_jobrun[0]
                )
            }
        )
        # self.app._process_evaluation_summary = MagicMock(return_value=MagicMock())

        self.app.list(TestDataset.COMPARTMENT_ID)
        mock_query_resources.assert_called_once()
        mock_query_resource.assert_not_called()

    @patch("ads.aqua.utils.query_resource")
    @patch("ads.aqua.utils.query_resources")
    def test_error_in_fetch_job(self, mock_query_resources, mock_query_resource):
        """Tests when fetching job encounters error."""
        mock_query_resources.return_value = [
            oci.resource_search.models.ResourceSummary(**item)
            for item in TestDataset.resource_summary_object_eval
        ]
        mock_query_resource.side_effect = Exception()
        self.app._process_evaluation_summary = MagicMock(return_value=MagicMock())
        self.app.list(TestDataset.COMPARTMENT_ID)

        mock_query_resource.assert_called_once()
        self.app._process_evaluation_summary.assert_called_with(
            model=oci.resource_search.models.ResourceSummary(
                **TestDataset.resource_summary_object_eval[0]
            ),
            jobrun=None,
        )

    @patch("ads.aqua.utils.query_resources")
    def test_missing_info_in_custometadata(self, mock_query_resources):
        """Tests missing info in evaluation custom metadata."""
        eval_without_meta = copy.deepcopy(TestDataset.resource_summary_object_eval[0])
        eval_without_meta.get("additional_details").update(dict(metadata=[]))
        mock_query_resources.return_value = [
            oci.resource_search.models.ResourceSummary(**eval_without_meta)
        ]
        self.app.ds_client.head_model_artifact = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=MagicMock(),
            ),
        )
        response = self.app.list(TestDataset.COMPARTMENT_ID)

        assert len(response) == 1


class TestCancelDeleteEvaluation(unittest.IsolatedAsyncioTestCase):
    """More test cases for cancel and delete evaluation."""

    def setUp(self):
        self.app = AquaEvaluationApp()
        self.mock_model = DataScienceModel(id="model456")

    @patch.object(DataScienceJobRun, "cancel")
    @patch("ads.aqua.evaluation.logger")
    async def test_cancel(self, mock_logger, mock_cancel):
        await self.app._cancel_job_run(DataScienceJobRun(), self.mock_model)

        mock_cancel.assert_called_once()
        mock_logger.info.assert_called_once()

    @patch("ads.aqua.evaluation.logger")
    async def test_cancel_exception(self, mock_logger):
        mock_cancel = MagicMock(
            side_effect=oci.exceptions.ServiceError(
                status=404, code=None, headers={}, message="error test msg"
            )
        )
        mock_run = DataScienceJobRun()
        mock_run.cancel = mock_cancel

        await self.app._cancel_job_run(mock_run, self.mock_model)

        mock_cancel.assert_called_once()
        mock_logger.error.assert_called_once()

    @patch("ads.aqua.evaluation.logger")
    async def test_delete(self, mock_logger):
        mock_job = DataScienceJob()
        mock_job.dsc_job.delete = MagicMock()
        self.mock_model.delete = MagicMock()

        await self.app._delete_job_and_model(mock_job, self.mock_model)

        mock_job.dsc_job.delete.assert_called_once()
        self.mock_model.delete.assert_called_once()
        mock_logger.info.assert_called()

    @patch("ads.aqua.evaluation.logger")
    async def test_delete_exception(self, mock_logger):
        mock_job = DataScienceJob()
        mock_job.dsc_job.delete = MagicMock(
            side_effect=oci.exceptions.ServiceError(
                status=404, code=None, headers={}, message="error test msg"
            )
        )

        await self.app._delete_job_and_model(mock_job, self.mock_model)

        mock_logger.error.assert_called_once()
