#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import MagicMock, patch

import pytest
from oci.feature_store.models import FeatureStore
from oci.response import Response

from ads.common.oci_mixin import OCIModelMixin
from ads.feature_store.service.oci_feature_store import (
    OCIFeatureStore,
)

FEATURE_STORE_OCID = "ocid1.feature_store.oc1.iad.xxx"

OCI_FEATURE_STORE_PAYLOAD = {
    "id": FEATURE_STORE_OCID,
    "compartment_id": "ocid1.compartment.oc1..xxx",
    "display_name": "feature_store name",
    "description": "The feature_store description",
    "lifecycle_state": "ACTIVE",
    "created_by": "ocid1.user.oc1..xxx",
    "freeform_tags": {"key1": "value1"},
    "defined_tags": {"key1": {"skey1": "value1"}},
    "time_created": "2022-08-24T17:07:39.200000Z",
}


class TestOCIFeatureStore:
    def setup_class(cls):
        # Mock delete model response
        cls.mock_delete_feature_store_response = Response(
            data=None, status=None, headers=None, request=None
        )

        # Mock create/update model response
        cls.mock_create_feature_store_response = Response(
            data=FeatureStore(**OCI_FEATURE_STORE_PAYLOAD),
            status=None,
            headers=None,
            request=None,
        )
        cls.mock_update_feature_store_response = Response(
            data=FeatureStore(**OCI_FEATURE_STORE_PAYLOAD),
            status=None,
            headers=None,
            request=None,
        )

    def setup_method(self):
        self.mock_feature_store = OCIFeatureStore(**OCI_FEATURE_STORE_PAYLOAD)

    @pytest.fixture(scope="class")
    def mock_client(self):
        mock_client = MagicMock()
        mock_client.create_feature_store = MagicMock(
            return_value=self.mock_create_feature_store_response
        )
        mock_client.update_feature_store = MagicMock(
            return_value=self.mock_update_feature_store_response
        )
        mock_client.delete_feature_store = MagicMock(
            return_value=self.mock_delete_feature_store_response
        )
        return mock_client

    def test_create_fail(self):
        """Ensures creating model fails in case of wrong input params."""
        with pytest.raises(
            ValueError,
            match="`compartment_id` must be specified for the feature store.",
        ):
            OCIFeatureStore().create()

    def test_create_success(self, mock_client):
        """Ensures creating model passes in case of valid input params."""
        with patch.object(OCIFeatureStore, "client", mock_client):
            with patch.object(OCIFeatureStore, "to_oci_model") as mock_to_oci_model:
                with patch.object(
                    OCIFeatureStore, "update_from_oci_model"
                ) as mock_update_from_oci_model:
                    mock_update_from_oci_model.return_value = self.mock_feature_store
                    mock_oci_feature_store = FeatureStore(**OCI_FEATURE_STORE_PAYLOAD)
                    mock_to_oci_model.return_value = mock_oci_feature_store
                    result = self.mock_feature_store.create()
                    mock_client.create_feature_store.assert_called_with(
                        mock_oci_feature_store
                    )
                    assert result == self.mock_feature_store

    def test_update(self, mock_client):
        """Tests updating datascience feature_store."""
        with patch.object(OCIFeatureStore, "client", mock_client):
            with patch.object(OCIFeatureStore, "to_oci_model") as mock_to_oci_model:
                mock_oci_feature_store = FeatureStore(**OCI_FEATURE_STORE_PAYLOAD)
                mock_to_oci_model.return_value = mock_oci_feature_store
                with patch.object(
                    OCIFeatureStore, "update_from_oci_model"
                ) as mock_update_from_oci_model:
                    mock_update_from_oci_model.return_value = self.mock_feature_store
                    result = self.mock_feature_store.update()
                    mock_client.update_feature_store.assert_called_with(
                        self.mock_feature_store.id, mock_oci_feature_store
                    )
                    assert result == self.mock_feature_store

    @patch.object(OCIModelMixin, "from_ocid")
    def test_from_id(self, mock_from_ocid):
        """Tests getting a model by OCID."""
        OCIFeatureStore.from_id(FEATURE_STORE_OCID)
        mock_from_ocid.assert_called_with(FEATURE_STORE_OCID)
