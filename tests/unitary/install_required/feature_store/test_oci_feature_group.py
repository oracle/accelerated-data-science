#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import MagicMock, patch

import pytest
from oci.feature_store.models import FeatureGroup
from oci.response import Response

from ads.common.oci_mixin import OCIModelMixin
from ads.feature_store.service.oci_feature_group import (
    OCIFeatureGroup,
)

FEATURE_GROUP_OCID = "ocid1.featuregroup.oc1.iad.xxx"

OCI_FEATURE_GROUP_PAYLOAD = {
    "id": FEATURE_GROUP_OCID,
    "name": "feature_group",
    "entity_id": "ocid1.entity.oc1.iad.xxx",
    "description": "feature group description",
    "primary_keys": {"items": []},
    "feature_store_id": "ocid1.featurestore.oc1.iad.xxx",
    "compartment_id": "ocid1.compartment.oc1.iad.xxx",
    "input_feature_details": [
        {"name": "cc_num", "feature_type": "STRING", "order_number": 1},
        {"name": "provider", "feature_type": "STRING", "order_number": 2},
        {"name": "expires", "feature_type": "STRING", "order_number": 3},
    ],
    "lifecycle_state": "ACTIVE",
    "created_by": "ocid1.user.oc1..xxx",
    "time_created": "2022-08-24T17:07:39.200000Z",
}


class TestOCIFeatureGroup:
    def setup_class(cls):
        # Mock delete model response
        cls.mock_delete_featuregroup_response = Response(
            data=None, status=None, headers=None, request=None
        )

        # Mock create/update model response
        cls.mock_create_featuregroup_response = Response(
            data=FeatureGroup(**OCI_FEATURE_GROUP_PAYLOAD),
            status=None,
            headers=None,
            request=None,
        )
        cls.mock_update_featuregroup_response = Response(
            data=FeatureGroup(**OCI_FEATURE_GROUP_PAYLOAD),
            status=None,
            headers=None,
            request=None,
        )

    def setup_method(self):
        self.mock_featuregroup = OCIFeatureGroup(**OCI_FEATURE_GROUP_PAYLOAD)

    @pytest.fixture(scope="class")
    def mock_client(self):
        mock_client = MagicMock()
        mock_client.create_featuregroup = MagicMock(
            return_value=self.mock_create_featuregroup_response
        )
        mock_client.update_featuregroup = MagicMock(
            return_value=self.mock_update_featuregroup_response
        )
        mock_client.delete_featuregroup = MagicMock(
            return_value=self.mock_delete_featuregroup_response
        )
        return mock_client

    def test_create_fail(self):
        """Ensures creating model fails in case of wrong input params."""
        with pytest.raises(
            ValueError,
            match="`compartment_id` must be specified for the feature group.",
        ):
            OCIFeatureGroup().create()

    def test_create_success(self, mock_client):
        """Ensures creating model passes in case of valid input params."""
        with patch.object(OCIFeatureGroup, "client", mock_client):
            with patch.object(OCIFeatureGroup, "to_oci_model") as mock_to_oci_model:
                with patch.object(
                    OCIFeatureGroup, "update_from_oci_model"
                ) as mock_update_from_oci_model:
                    mock_update_from_oci_model.return_value = self.mock_featuregroup
                    mock_oci_featuregroup = FeatureGroup(**OCI_FEATURE_GROUP_PAYLOAD)
                    mock_to_oci_model.return_value = mock_oci_featuregroup
                    result = self.mock_featuregroup.create()
                    mock_client.create_feature_group.assert_called_with(
                        mock_oci_featuregroup
                    )
                    assert result == self.mock_featuregroup

    def test_update(self, mock_client):
        """Tests updating datascience featuregroup."""
        with patch.object(OCIFeatureGroup, "client", mock_client):
            with patch.object(OCIFeatureGroup, "to_oci_model") as mock_to_oci_model:
                mock_oci_featuregroup = FeatureGroup(**OCI_FEATURE_GROUP_PAYLOAD)
                mock_to_oci_model.return_value = mock_oci_featuregroup
                with patch.object(
                    OCIFeatureGroup, "update_from_oci_model"
                ) as mock_update_from_oci_model:
                    mock_update_from_oci_model.return_value = self.mock_featuregroup
                    result = self.mock_featuregroup.update()
                    mock_client.update_feature_group.assert_called_with(
                        mock_oci_featuregroup, self.mock_featuregroup.id
                    )
                    assert result == self.mock_featuregroup

    @patch.object(OCIModelMixin, "from_ocid")
    def test_from_id(self, mock_from_ocid):
        """Tests getting a model by OCID."""
        OCIFeatureGroup.from_id(FEATURE_GROUP_OCID)
        mock_from_ocid.assert_called_with(FEATURE_GROUP_OCID)
