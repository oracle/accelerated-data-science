#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
from collections import namedtuple
from dataclasses import asdict
from unittest.mock import ANY, MagicMock, patch

from oci.data_science.models import ModelSummary

from ads.aqua.model import AquaModelApp, AquaModelSummary


class TestDataset:
    Response = namedtuple("Response", ["data", "status"])
    DataList = namedtuple("DataList", ["objects"])

    model_summary_objects = [
        {
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "created_by": "ocid1.datasciencenotebooksession.oc1.iad.<OCID>",
            "defined_tags": {},
            "display_name": "Model1-Fine-Tuned",
            "freeform_tags": {
                "OCI_AQUA": "",
                "aqua_fine_tuned_model": "ocid1.datasciencemodel.oc1.iad.<OCID>#Model1",
                "license": "UPL",
                "organization": "Oracle AI",
                "task": "text_generation",
            },
            "id": "ocid1.datasciencemodel.oc1.iad.<OCID>",
            "lifecycle_state": "ACTIVE",
            "project_id": "ocid1.datascienceproject.oc1.iad.<OCID>",
            "time_created": "2024-01-19T19:33:58.078000+00:00",
        },
        {
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "created_by": "ocid1.datasciencenotebooksession.oc1.iad.<OCID>",
            "defined_tags": {},
            "display_name": "Model1-Copy",
            "freeform_tags": {
                "OCI_AQUA": "",
                "license": "UPL",
                "organization": "Oracle AI",
                "task": "text_generation",
            },
            "id": "ocid1.datasciencemodel.oc1.iad.<OCID>",
            "lifecycle_state": "ACTIVE",
            "project_id": "ocid1.datascienceproject.oc1.iad.<OCID>",
            "time_created": "2024-01-19T19:30:39.452000+00:00",
        },
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

    model_object = {
        "compartment_id": "ocid1.compartment.oc1.<OCID>",
        "created_by": "ocid1.datasciencenotebooksession.oc1.iad.<OCID>",
        "custom_metadata_list": [
            {
                "category": "other",
                "description": "model by reference storage path",
                "key": "Object Storage Path",
                "value": "oci://mybucket@mytenancy/prefix/artifact",
            },
            {
                "category": "other",
                "description": "The model file name.",
                "key": "ModelFileName",
                "value": "model.pkl",
            },
        ],
        "display_name": "Model1-Fine-Tuned",
        "freeform_tags": {
            "OCI_AQUA": "",
            "aqua_fine_tuned_model": "ocid1.datasciencemodel.oc1.iad.<OCID>#Model1",
            "license": "UPL",
            "organization": "Oracle AI",
            "task": "text_generation",
        },
        "id": "ocid1.datasciencemodel.oc1.iad.<OCID>",
        "project_id": "ocid1.datascienceproject.<OCID>",
        "time_created": "2024-01-19T19:33:58.078000+00:00",
    }

    COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"
    MOCK_ICON = "data:image/svg+xml;base64,########"


class TestAquaModel(unittest.TestCase):
    """Contains unittests for AquaModelApp."""

    def setUp(self):
        self.app = AquaModelApp()  # Replace with actual instantiation

    @patch("ads.common.auth.default_signer")
    def test_list(self, mock_auth):
        """Tests list models succesfully."""
        self.app.list_resource = MagicMock(
            return_value=[
                ModelSummary(**item) for item in TestDataset.model_summary_objects
            ]
        )

        results = self.app.list(TestDataset.COMPARTMENT_ID)

        assert len(results) == 2

        attributes = AquaModelSummary.__annotations__.keys()
        for r in results:
            rdict = asdict(r)
            for attr in attributes:
                assert rdict.get(attr) is not None
