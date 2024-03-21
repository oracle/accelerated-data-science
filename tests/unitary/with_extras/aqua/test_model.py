#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest
from dataclasses import asdict
from importlib import reload
from unittest.mock import MagicMock

import oci
from parameterized import parameterized

import ads.aqua.model
import ads.config
from ads.aqua.model import AquaModelApp, AquaModelSummary


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
        reload(ads.aqua.model)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("CONDA_BUCKET_NS", None)
        os.environ.pop("ODSC_MODEL_COMPARTMENT_OCID", None)
        reload(ads.config)
        reload(ads.aqua.model)

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
