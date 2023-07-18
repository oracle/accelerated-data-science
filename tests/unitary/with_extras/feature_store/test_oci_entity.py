#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import MagicMock, patch

import pytest
from oci.feature_store.models import Entity
from oci.response import Response

from ads.common.oci_mixin import OCIModelMixin
from ads.feature_store.service.oci_entity import (
    OCIEntity,
)

ENTITY_OCID = "ocid1.entity.oc1.iad.xxx"

OCI_ENTITY_PAYLOAD = {
    "id": ENTITY_OCID,
    "compartment_id": "ocid1.compartment.oc1..xxx",
    "feature_store_id": "ocid1.featurestore.oc1.iad.xxx",
    "name": "entity name",
    "description": "The entity description",
    "lifecycle_state": "ACTIVE",
    "created_by": "ocid1.user.oc1..xxx",
    "freeform_tags": {"key1": "value1"},
    "defined_tags": {"key1": {"skey1": "value1"}},
    "time_created": "2022-08-24T17:07:39.200000Z",
}


class TestOCIEntity:
    def setup_class(cls):
        # Mock delete model response
        cls.mock_delete_entity_response = Response(
            data=None, status=None, headers=None, request=None
        )

        # Mock create/update model response
        cls.mock_create_entity_response = Response(
            data=Entity(**OCI_ENTITY_PAYLOAD),
            status=None,
            headers=None,
            request=None,
        )
        cls.mock_update_entity_response = Response(
            data=Entity(**OCI_ENTITY_PAYLOAD),
            status=None,
            headers=None,
            request=None,
        )

    def setup_method(self):
        self.mock_entity = OCIEntity(**OCI_ENTITY_PAYLOAD)

    @pytest.fixture(scope="class")
    def mock_client(self):
        mock_client = MagicMock()
        mock_client.create_entity = MagicMock(
            return_value=self.mock_create_entity_response
        )
        mock_client.update_entity = MagicMock(
            return_value=self.mock_update_entity_response
        )
        mock_client.delete_entity = MagicMock(
            return_value=self.mock_delete_entity_response
        )
        return mock_client

    def test_create_fail(self):
        """Ensures creating model fails in case of wrong input params."""
        with pytest.raises(
            ValueError,
            match="The `compartment_id` must be specified.",
        ):
            OCIEntity().create()

    def test_create_success(self, mock_client):
        """Ensures creating model passes in case of valid input params."""
        with patch.object(OCIEntity, "client", mock_client):
            with patch.object(OCIEntity, "to_oci_model") as mock_to_oci_model:
                with patch.object(
                    OCIEntity, "update_from_oci_model"
                ) as mock_update_from_oci_model:
                    mock_update_from_oci_model.return_value = self.mock_entity
                    mock_oci_entity = Entity(**OCI_ENTITY_PAYLOAD)
                    mock_to_oci_model.return_value = mock_oci_entity
                    result = self.mock_entity.create()
                    mock_client.create_entity.assert_called_with(mock_oci_entity)
                    assert result == self.mock_entity

    def test_update(self, mock_client):
        """Tests updating datascience entity."""
        with patch.object(OCIEntity, "client", mock_client):
            with patch.object(OCIEntity, "to_oci_model") as mock_to_oci_model:
                mock_oci_entity = Entity(**OCI_ENTITY_PAYLOAD)
                mock_to_oci_model.return_value = mock_oci_entity
                with patch.object(
                    OCIEntity, "update_from_oci_model"
                ) as mock_update_from_oci_model:
                    mock_update_from_oci_model.return_value = self.mock_entity
                    result = self.mock_entity.update()
                    mock_client.update_entity.assert_called_with(
                        self.mock_entity.id, mock_oci_entity
                    )
                    assert result == self.mock_entity

    @patch.object(OCIModelMixin, "from_ocid")
    def test_from_id(self, mock_from_ocid):
        """Tests getting a model by OCID."""
        OCIEntity.from_id(ENTITY_OCID)
        mock_from_ocid.assert_called_with(ENTITY_OCID)
