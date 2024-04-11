#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest
from dataclasses import asdict
from importlib import reload
from unittest.mock import MagicMock

from mock import patch
import oci
from parameterized import parameterized

import ads.aqua.model
import ads.config
from ads.aqua.model import AquaModelApp, AquaModelSummary
from ads.model.datascience_model import DataScienceModel
from ads.model.model_metadata import ModelCustomMetadata, ModelProvenanceMetadata, ModelTaxonomyMetadata


class TestDataset:
    model_summary_objects = [
        {
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "created_by": "ocid1.datasciencenotebooksession.oc1.iad.<OCID>",
            "defined_tags": {},
            "display_name": "Model1",
            "freeform_tags": {
                "OCI_AQUA": "",
                "aqua_service_model": "ocid1.datasciencemodel.oc1.iad.<OCID>#Model1",
                "license": "UPL",
                "organization": "Oracle AI",
                "task": "text_generation",
            },
            "id": "ocid1.datasciencemodel.oc1.iad.<OCID>",
            "lifecycle_state": "ACTIVE",
            "project_id": "ocid1.datascienceproject.oc1.iad.<OCID>",
            "time_created": "2024-01-19T17:57:39.158000+00:00",
        },
    ]

    resource_summary_objects = [
        {
            "additional_details": {},
            "availability_domain": "",
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "defined_tags": {},
            "display_name": "Model1-Fine-Tuned",
            "freeform_tags": {
                "OCI_AQUA": "",
                "aqua_fine_tuned_model": "ocid1.datasciencemodel.oc1.iad.<OCID>#Model1",
                "license": "UPL",
                "organization": "Oracle AI",
                "task": "text_generation",
            },
            "identifier": "ocid1.datasciencemodel.oc1.iad.<OCID>",
            "identity_context": {},
            "lifecycle_state": "ACTIVE",
            "resource_type": "DataScienceModel",
            "search_context": "",
            "system_tags": {},
            "time_created": "2024-01-19T19:33:58.078000+00:00",
        },
    ]

    SERVICE_COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"
    COMPARTMENT_ID = "ocid1.compartment.oc1..<UNIQUE_OCID>"


class TestAquaModel(unittest.TestCase):
    """Contains unittests for AquaModelApp."""

    def setUp(self):
        self.app = AquaModelApp()

    @classmethod
    def setUpClass(cls):
        os.environ["CONDA_BUCKET_NS"] = "test-namespace"
        os.environ["ODSC_MODEL_COMPARTMENT_OCID"] = TestDataset.SERVICE_COMPARTMENT_ID
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.model)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("CONDA_BUCKET_NS", None)
        os.environ.pop("ODSC_MODEL_COMPARTMENT_OCID", None)
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.model)

    @patch.object(DataScienceModel, "create")
    @patch("ads.model.datascience_model.validate")
    @patch.object(DataScienceModel, "from_id")
    def test_create_model(self, mock_from_id, mock_validate, mock_create):
        service_model = MagicMock()
        service_model.model_file_description = {"test_key":"test_value"}
        service_model.display_name = "test_display_name"
        service_model.description = "test_description"
        service_model.freeform_tags = {"test_key":"test_value"}
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            key="test_metadata_item_key",
            value="test_metadata_item_value"
        )
        service_model.custom_metadata_list = custom_metadata_list
        service_model.provenance_metadata = ModelProvenanceMetadata(
            training_id="test_training_id"
        )
        mock_from_id.return_value = service_model

        # will not copy service model 
        self.app.create(
            model_id="test_model_id",
            project_id="test_project_id",
            compartment_id="test_compartment_id",
        )

        mock_from_id.assert_called_with("test_model_id")
        mock_validate.assert_not_called()
        mock_create.assert_not_called()

        service_model.compartment_id = TestDataset.SERVICE_COMPARTMENT_ID
        mock_from_id.return_value = service_model

        # will copy service model
        self.app.create(
            model_id="test_model_id",
            project_id="test_project_id",
            compartment_id="test_compartment_id"
        )

        mock_from_id.assert_called_with("test_model_id")
        mock_validate.assert_called()
        mock_create.assert_called_with(
            model_by_reference=True
        )

    @patch("ads.aqua.model.read_file")
    @patch.object(DataScienceModel, "from_id")
    def test_get_model_not_fine_tuned(self, mock_from_id, mock_read_file):
        ds_model = MagicMock()
        ds_model.id = "test_id"
        ds_model.compartment_id = "test_compartment_id"
        ds_model.project_id = "test_project_id"
        ds_model.display_name = "test_display_name"
        ds_model.description = "test_description"
        ds_model.freeform_tags = {
            "OCI_AQUA":"ACTIVE",
            "license":"test_license",
            "organization":"test_organization",
            "task":"test_task"
        }
        ds_model.time_created = "2024-01-19T17:57:39.158000+00:00"
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            key="artifact_location",
            value="oci://bucket@namespace/prefix/"
        )
        ds_model.custom_metadata_list = custom_metadata_list

        mock_from_id.return_value = ds_model
        mock_read_file.return_value = "test_model_card"

        aqua_model = self.app.get(model_id="test_model_id")

        mock_from_id.assert_called_with("test_model_id")
        mock_read_file.assert_called_with(
            file_path="oci://bucket@namespace/prefix/README.md",
            auth=self.app._auth,
        )

        assert asdict(aqua_model) == {
            'compartment_id': f'{ds_model.compartment_id}',
            'console_link': (
                f'https://cloud.oracle.com/data-science/models/{ds_model.id}?region={self.app.region}',
            ),
            'icon': '',
            'id': f'{ds_model.id}',
            'is_fine_tuned_model': False,
            'license': f'{ds_model.freeform_tags["license"]}',
            'model_card': f'{mock_read_file.return_value}',
            'name': f'{ds_model.display_name}',
            'organization': f'{ds_model.freeform_tags["organization"]}',
            'project_id': f'{ds_model.project_id}',
            'ready_to_deploy': True,
            'ready_to_finetune': False,
            'search_text': 'ACTIVE,test_license,test_organization,test_task',
            'tags': ds_model.freeform_tags,
            'task': f'{ds_model.freeform_tags["task"]}',
            'time_created': f'{ds_model.time_created}'
        }

    @patch("ads.aqua.utils.query_resource")
    @patch("ads.aqua.model.read_file")
    @patch.object(DataScienceModel, "from_id")
    def test_get_model_fine_tuned(self, mock_from_id, mock_read_file, mock_query_resource):
        ds_model = MagicMock()
        ds_model.id = "test_id"
        ds_model.compartment_id = "test_model_compartment_id"
        ds_model.project_id = "test_project_id"
        ds_model.display_name = "test_display_name"
        ds_model.description = "test_description"
        ds_model.model_version_set_id = "test_model_version_set_id"
        ds_model.model_version_set_name = "test_model_version_set_name"
        ds_model.freeform_tags = {
            "OCI_AQUA":"ACTIVE",
            "license":"test_license",
            "organization":"test_organization",
            "task":"test_task",
            "aqua_fine_tuned_model":"test_finetuned_model"
        }
        ds_model.time_created = "2024-01-19T17:57:39.158000+00:00"
        ds_model.lifecycle_state = "ACTIVE"
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            key="artifact_location",
            value="oci://bucket@namespace/prefix/"
        )
        custom_metadata_list.add(
            key="fine_tune_source",
            value="test_fine_tuned_source_id"
        )
        custom_metadata_list.add(
            key="fine_tune_source_name",
            value="test_fine_tuned_source_name"
        )
        ds_model.custom_metadata_list = custom_metadata_list
        defined_metadata_list = ModelTaxonomyMetadata()
        defined_metadata_list["Hyperparameters"].value = {
            "training_data" : "test_training_data",
            "val_set_size" : "test_val_set_size"
        }
        ds_model.defined_metadata_list = defined_metadata_list
        ds_model.provenance_metadata = ModelProvenanceMetadata(
            training_id="test_training_job_run_id"
        )

        mock_from_id.return_value = ds_model
        mock_read_file.return_value = "test_model_card"

        response = MagicMock()
        job_run = MagicMock()
        job_run.id = "test_job_run_id"
        job_run.lifecycle_state = "SUCCEEDED"
        job_run.lifecycle_details = "test lifecycle details"
        job_run.identifier = "test_job_id",
        job_run.display_name = "test_job_name"
        job_run.compartment_id = "test_job_run_compartment_id"
        job_infrastructure_configuration_details = MagicMock()
        job_infrastructure_configuration_details.shape_name = "test_shape_name"

        job_configuration_override_details = MagicMock()
        job_configuration_override_details.environment_variables = {
            "NODE_COUNT" : 1
        }
        job_run.job_infrastructure_configuration_details = job_infrastructure_configuration_details
        job_run.job_configuration_override_details = job_configuration_override_details
        log_details = MagicMock()
        log_details.log_id = "test_log_id"
        log_details.log_group_id = "test_log_group_id"
        job_run.log_details = log_details
        response.data = job_run
        self.app.ds_client.get_job_run = MagicMock(
            return_value = response
        )

        query_resource = MagicMock()
        query_resource.display_name = "test_display_name"
        mock_query_resource.return_value = query_resource

        model = self.app.get(model_id="test_model_id")

        mock_from_id.assert_called_with("test_model_id")
        mock_read_file.assert_called_with(
            file_path="oci://bucket@namespace/prefix/README.md",
            auth=self.app._auth,
        )
        mock_query_resource.assert_called()

        assert asdict(model) == {
            'compartment_id': f'{ds_model.compartment_id}',
            'console_link': (
                f'https://cloud.oracle.com/data-science/models/{ds_model.id}?region={self.app.region}',
            ),
            'dataset': 'test_training_data',
            'experiment': {'id': '', 'name': '', 'url': ''},
            'icon': '',
            'id': f'{ds_model.id}',
            'is_fine_tuned_model': True,
            'job': {'id': '', 'name': '', 'url': ''},
            'license': 'test_license',
            'lifecycle_details': f'{job_run.lifecycle_details}',
            'lifecycle_state': f'{ds_model.lifecycle_state}',
            'log': {
                'id': f'{log_details.log_id}',
                'name': f'{query_resource.display_name}',
                'url': 'https://cloud.oracle.com/logging/search?searchQuery=search '
                    f'"{job_run.compartment_id}/{log_details.log_group_id}/{log_details.log_id}" | '
                    f"source='{job_run.id}' | sort by datetime desc&regions={self.app.region}"
                },
            'log_group': {
                'id': f'{log_details.log_group_id}',
                'name': f'{query_resource.display_name}',
                'url': f'https://cloud.oracle.com/logging/log-groups/{log_details.log_group_id}?region={self.app.region}'
                },
            'metrics': [
                {
                    'category': 'validation',
                    'name': 'validation_metrics',
                    'scores': []
                },
                {
                    'category': 'training',
                    'name': 'training_metrics',
                    'scores': []
                },
                {
                    'category': 'validation',
                    'name': 'validation_metrics_final',
                    'scores': []
                },
                {
                    'category': 'training',
                    'name': 'training_metrics_final',
                    'scores': []
                }
            ],
            'model_card': f'{mock_read_file.return_value}',
            'name': f'{ds_model.display_name}',
            'organization': 'test_organization',
            'project_id': f'{ds_model.project_id}',
            'ready_to_deploy': True,
            'ready_to_finetune': False,
            'search_text': 'ACTIVE,test_license,test_organization,test_task,test_finetuned_model',
            'shape_info': {
                'instance_shape': f'{job_infrastructure_configuration_details.shape_name}',
                'replica': 1,
            },
            'source': {'id': '', 'name': '', 'url': ''},
            'tags': ds_model.freeform_tags,
            'task': 'test_task',
            'time_created': f'{ds_model.time_created}',
            'validation': {
                'type': 'Automatic split',
                'value': 'test_val_set_size'
            }
        }

    @patch("ads.aqua.model.read_file")
    @patch("ads.aqua.model.get_artifact_path")
    def test_load_license(self, mock_get_artifact_path, mock_read_file):
        self.app.ds_client.get_model = MagicMock()
        mock_get_artifact_path.return_value = "oci://bucket@namespace/prefix/config/LICENSE.txt"
        mock_read_file.return_value = "test_license"

        license = self.app.load_license(model_id="test_model_id")

        mock_get_artifact_path.assert_called()
        mock_read_file.assert_called()

        assert asdict(license) == {
            'id': 'test_model_id', 'license': 'test_license'
        }

    def test_list_service_models(self):
        """Tests listing service models succesfully."""

        self.app.list_resource = MagicMock(
            return_value=[
                oci.data_science.models.ModelSummary(**item)
                for item in TestDataset.model_summary_objects
            ]
        )

        results = self.app.list()

        received_args = self.app.list_resource.call_args.kwargs
        assert received_args.get("compartment_id") == TestDataset.SERVICE_COMPARTMENT_ID

        assert len(results) == 1

        attributes = AquaModelSummary.__annotations__.keys()
        for r in results:
            rdict = asdict(r)
            print("############ Expected Response ############")
            print(rdict)

            for attr in attributes:
                assert rdict.get(attr) is not None

    def test_list_custom_models(self):
        """Tests list custom models succesfully."""

        self.app._rqs = MagicMock(
            return_value=[
                oci.resource_search.models.ResourceSummary(**item)
                for item in TestDataset.resource_summary_objects
            ]
        )

        results = self.app.list(TestDataset.COMPARTMENT_ID)

        self.app._rqs.assert_called_with(TestDataset.COMPARTMENT_ID)

        assert len(results) == 1

        attributes = AquaModelSummary.__annotations__.keys()
        for r in results:
            rdict = asdict(r)
            print("############ Expected Response ############")
            print(rdict)

            for attr in attributes:
                assert rdict.get(attr) is not None

    @parameterized.expand(
        [
            (
                None,
                {"license": "UPL", "org": "Oracle", "task": "text_generation"},
                "UPL,Oracle,text_generation",
            ),
            (
                "This is a description.",
                {"license": "UPL", "org": "Oracle", "task": "text_generation"},
                "This is a description. UPL,Oracle,text_generation",
            ),
        ]
    )
    def test_build_search_text(self, description, tags, expected_output):
        assert (
            self.app._build_search_text(tags=tags, description=description)
            == expected_output
        )
