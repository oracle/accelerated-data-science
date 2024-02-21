#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import unittest
from unittest.mock import MagicMock, call

import oci

from ads.aqua import utils
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.extension.base_handler import AquaAPIhandler

null = None


class TestDataset:
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

    resource_summary_object_md = [
        {
            "additional_details": {},
            "availability_domain": null,
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "defined_tags": {},
            "display_name": "datasciencemodeldeployment_Mixtral-8x7B-v0.1_20240220",
            "freeform_tags": {"OCI_AQUA": ""},
            "identifier": "ocid1.datasciencemodeldeployment.oc1.iad.<OCID>",
            "identity_context": {},
            "lifecycle_state": "CREATING",
            "resource_type": "DataScienceModelDeployment",
            "search_context": null,
            "system_tags": {},
            "time_created": "2024-02-20T09:00:50.489000+00:00",
        }
    ]

    resource_summary_object_job = [
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
                        "value": '{"model_params": {"max_tokens": 100, "temperature": 100, "top_p": 1, "top_k": 1}, "model_config": {"gpu_memory": "0.9"}}',
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
                        "value": '{"bert_score": {"precision": {"0.25": "0.345", "0.5": "0.3453", "0.75": "0.66553"}, "recall": {"0.25": "453", "0.5": "0.4345", "0.75": "0.53435"}, "f1": {"0.25": "453", "0.5": "0.4345", "0.75": "0.53435"}}}',
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
                "evaluation_job_id": "ocid1.datasciencejobrun.oc1.iad.<OCID>",
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
                **TestDataset.resource_summary_object_job[0]
            )

    def assert_payload(self, response, method):
        print(f"############ Expected Response For {method} ############")
        if isinstance(response, list):
            response = {"data": response}
        response = json.loads(json.dumps(response, default=AquaAPIhandler.serialize))
        print(response)

    def test_get(self):
        """Tests get evaluation details successfully."""

        self.app.ds_client.get_job_run = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=oci.data_science.models.JobRun(**TestDataset.job_run_object),
            )
        )
        response = self.app.get(TestDataset.EVAL_ID)
        calls = [
            call(TestDataset.EVAL_ID),
            call(
                TestDataset.resource_summary_object_md[0].get("identifier"),
                return_all=False,
            ),
        ]
        utils.query_resource.assert_has_calls(calls)

        self.app.ds_client.get_job_run.assert_called_with(
            TestDataset.resource_summary_object_eval[0]
            .get("freeform_tags")
            .get("evaluation_job_id")
        )
        self.assert_payload(response, "GET EVALUATION")

    def test_list(self):
        """Tests list evaluations successfully."""

        response = self.app.list(TestDataset.COMPARTMENT_ID)

        utils.query_resources.assert_called_with(
            compartment_id=TestDataset.COMPARTMENT_ID, resource_type="datasciencemodel"
        )

        calls = [
            call(
                TestDataset.resource_summary_object_eval[0]
                .get("freeform_tags")
                .get("evaluation_job_id"),
                return_all=False,
            ),
            call(
                TestDataset.resource_summary_object_md[0].get("identifier"),
                return_all=False,
            ),
        ]
        utils.query_resource.assert_has_calls(calls)

        assert len(response) == 1
        self.assert_payload(response, "LIST EVALUATIONS")

    def test_missing_tags(self):
        pass

    def test_missing_metadata(self):
        pass
