#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import base64
import json
import os
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import oci

from ads.aqua import utils
from ads.aqua.evaluation import (
    AquaEvalMetrics,
    AquaEvalReport,
    AquaEvaluationApp,
    AquaEvaluationSummary,
)
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.model import DataScienceModel
from ads.jobs.ads_job import DataScienceJobRun, DataScienceJob

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

    resource_summary_object_eval = [
        {
            "additional_details": {
                "createdBy": "ocid1.datasciencenotebooksession.oc1.iad.<OCID>",
                "description": "This is a dummy MC entry for evaluation.",
                "metadata": [
                    {
                        "category": "other",
                        "description": "The model that was evaluated.",
                        "key": "evaluation_source",
                        "value": "ocid1.datasciencemodeldeployment.oc1.iad.<OCID>",
                    },
                    {
                        "category": "other",
                        "description": "The model that was evaluated.",
                        "key": "evaluation_source_name",
                        "value": "Model 1",
                    },
                    {
                        "category": "other",
                        "description": "The JOB OCID associated with the Evaluation MC entry.",
                        "key": "evaluation_job_id",
                        "value": "ocid1.datasciencejob.oc1.iad.<OCID>",
                    },
                    {
                        "category": "other",
                        "description": "The JOB RUN OCID associated with the Evaluation MC entry.",
                        "key": "evaluation_job_run_id",
                        "value": "ocid1.datasciencejobrun.oc1.iad.<OCID>",
                    },
                    {
                        "category": "other",
                        "description": "The OS path to store the result of the evaluation.",
                        "key": "evaluation_output_path",
                        "value": "oci://ming-dev@ociodscdev/output",
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
                        "value": '{"model_params": {"shape": "BM.A10.2", "max_tokens": 100, "temperature": 100, "top_p": 1, "top_k": 1}, "model_config": {"gpu_memory": "0.9"}}',
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

    COMPARTMENT_ID = "ocid1.compartment.oc1..<UNIQUE_OCID>"
    EVAL_ID = "ocid1.datasciencemodel.oc1.iad.<OCID>"


class TestAquaModel(unittest.TestCase):
    """Contains unittests for AquaModelApp."""

    @classmethod
    def setUpClass(cls):
        utils.is_valid_ocid = MagicMock(return_value=True)

    def setUp(self):
        self.app = AquaEvaluationApp()
        utils.query_resources = MagicMock(
            return_value=[
                oci.resource_search.models.ResourceSummary(**item)
                for item in TestDataset.resource_summary_object_eval
            ]
        )
        utils.query_resource = MagicMock(side_effect=self.side_effect_func)

    def side_effect_func(*args, **kwargs):
        if args[1].startswith("ocid1.datasciencemodeldeployment"):
            return oci.resource_search.models.ResourceSummary(
                **TestDataset.resource_summary_object_md[0]
            )
        elif args[1].startswith("ocid1.datasciencemodel"):
            return oci.resource_search.models.ResourceSummary(
                **TestDataset.resource_summary_object_eval[0]
            )
        elif args[1].startswith("ocid1.datasciencejob"):
            return oci.resource_search.models.ResourceSummary(
                **TestDataset.resource_summary_object_jobrun[0]
            )

    def print_expected_response(self, response, method):
        """Used for manually check expected output."""
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
            assert rdict.get(attr)

    def test_get(self):
        """Tests get evaluation details successfully."""
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

        response = self.app.get(TestDataset.EVAL_ID)

        utils.query_resource.assert_called_with(TestDataset.EVAL_ID)

        self.app.ds_client.get_job_run.assert_called_with(
            TestDataset.model_provenance_object.get("training_id")
        )
        self.print_expected_response(response, "GET EVALUATION")
        self.assert_payload(response, AquaEvaluationSummary)

    def test_list(self):
        """Tests list evaluations successfully."""
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

    @patch.object(DataScienceModel, "from_id")
    @patch.object(DataScienceJob, "from_id")
    def test_delete_evaluation(self, mock_dsc_job, mock_dsc_model_from_id):
        mock_dsc_model_delete = MagicMock()
        mock_dsc_model_from_id.return_value = MagicMock(
            provenance_data={
                "training_id": TestDataset.model_provenance_object.get("training_id"),
            },
            delete=mock_dsc_model_delete,
        )
        mock_dsc_job_delete = MagicMock()
        mock_dsc_job.return_value = MagicMock(
            lifecycle_state="ACCEPTED", delete=mock_dsc_job_delete
        )
        mock_dsc_model_delete.return_value = None

        result = self.app.delete(TestDataset.EVAL_ID)

        assert result["id"] == TestDataset.EVAL_ID
        assert result["lifecycle_state"] == "DELETING"

        mock_dsc_job_delete.assert_called_once()
        mock_dsc_model_delete.assert_called_once()

    @patch.object(DataScienceModel, "from_id")
    @patch.object(DataScienceJobRun, "from_ocid")
    def test_cancel_evaluation(self, mock_dsc_job_run, mock_dsc_model_from_id):
        mock_dsc_model_from_id.return_value = MagicMock(
            provenance_data={
                "training_id": TestDataset.model_provenance_object.get("training_id")
            }
        )
        mock_dsc_job_run_cancel = MagicMock()
        mock_dsc_job_run.return_value = MagicMock(
            lifecycle_state="ACCEPTED", cancel=mock_dsc_job_run_cancel
        )

        result = self.app.cancel(TestDataset.EVAL_ID)

        assert result["id"] == TestDataset.EVAL_ID
        assert result["lifecycle_state"] == "CANCELING"
        mock_dsc_job_run_cancel.assert_called_once()

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

    def test_get_status(self):
        """Tests getting evaluation status successfully."""
        # TODO: add test for difference cases.
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
        response = self.app.get_status(TestDataset.EVAL_ID)
        self.print_expected_response(response, "GET STATUS")
        assert response.get("lifecycle_state") == "SUCCEEDED"
