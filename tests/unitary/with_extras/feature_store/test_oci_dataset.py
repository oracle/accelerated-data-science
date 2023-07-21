#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import MagicMock, patch

import pytest
from oci.feature_store.models import Dataset
from oci.response import Response

from ads.common.oci_mixin import OCIModelMixin
from ads.feature_store.service.oci_dataset import (
    OCIDataset,
)

DATASET_OCID = "ocid1.dataset.oc1.iad.xxx"

OCI_DATASET_PAYLOAD = {
    "id": DATASET_OCID,
    "compartment_id": "ocid1.compartment.oc1..xxx",
    "feature_store_id": "ocid1.featurestore.oc1.iad.xxx",
    "query": "SELECT feature_gr_1.name FROM feature_gr_1",
    "entity_id": "ocid1.entity.oc1.iad.xxx",
    "name": "dataset name",
    "description": "The dataset description",
    "lifecycle_state": "ACTIVE",
    "created_by": "ocid1.user.oc1..xxx",
    "time_created": "2022-08-24T17:07:39.200000Z",
}


class TestOCIDataset:
    def setup_class(cls):
        # Mock delete model response
        cls.mock_delete_dataset_response = Response(
            data=None, status=None, headers=None, request=None
        )

        # Mock create/update model response
        cls.mock_create_dataset_response = Response(
            data=Dataset(**OCI_DATASET_PAYLOAD),
            status=None,
            headers=None,
            request=None,
        )
        cls.mock_update_dataset_response = Response(
            data=Dataset(**OCI_DATASET_PAYLOAD),
            status=None,
            headers=None,
            request=None,
        )

    def setup_method(self):
        self.mock_dataset = OCIDataset(**OCI_DATASET_PAYLOAD)

    @pytest.fixture(scope="class")
    def mock_client(self):
        mock_client = MagicMock()
        mock_client.create_dataset = MagicMock(
            return_value=self.mock_create_dataset_response
        )
        mock_client.update_dataset = MagicMock(
            return_value=self.mock_update_dataset_response
        )
        mock_client.delete_dataset = MagicMock(
            return_value=self.mock_delete_dataset_response
        )
        return mock_client

    def test_create_fail(self):
        """Ensures creating model fails in case of wrong input params."""
        with pytest.raises(
            ValueError,
            match="compartment_id` must be specified for the dataset.",
        ):
            OCIDataset().create()

    def test_create_success(self, mock_client):
        """Ensures creating model passes in case of valid input params."""
        with patch.object(OCIDataset, "client", mock_client):
            with patch.object(OCIDataset, "to_oci_model") as mock_to_oci_model:
                with patch.object(
                    OCIDataset, "update_from_oci_model"
                ) as mock_update_from_oci_model:
                    mock_update_from_oci_model.return_value = self.mock_dataset
                    mock_oci_dataset = Dataset(**OCI_DATASET_PAYLOAD)
                    mock_to_oci_model.return_value = mock_oci_dataset
                    result = self.mock_dataset.create()
                    mock_client.create_dataset.assert_called_with(mock_oci_dataset)
                    assert result == self.mock_dataset

    @patch.object(OCIModelMixin, "from_ocid")
    def test_from_id(self, mock_from_ocid):
        """Tests getting a model by OCID."""
        OCIDataset.from_id(DATASET_OCID)
        mock_from_ocid.assert_called_with(DATASET_OCID)
