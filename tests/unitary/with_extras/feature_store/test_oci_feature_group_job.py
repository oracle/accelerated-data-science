#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import MagicMock, patch

import pytest
from oci.feature_store.models import FeatureGroupJob
from oci.response import Response

from ads.common.oci_mixin import OCIModelMixin
from ads.feature_store.service.oci_feature_group_job import (
    OCIFeatureGroupJob,
)

FEATURE_GROUP_JOB_OCID = "ocid1.feature_group_job.oc1.iad.xxx"

OCI_FEATURE_GROUP_JOB_PAYLOAD = {
    "compartment_id": "compartmentId",
    "feature_group_id": "ocid1.feature_group.oc1.iad.xxx",
    "ingestion_mode": "OVERWRITE",
}


class TestOCIFeatureGroupJob:
    def setup_class(cls):
        # Mock create/update model response
        cls.mock_create_entity_response = Response(
            data=FeatureGroupJob(**OCI_FEATURE_GROUP_JOB_PAYLOAD),
            status=None,
            headers=None,
            request=None,
        )

    def setup_method(self):
        self.mock_entity = OCIFeatureGroupJob(**OCI_FEATURE_GROUP_JOB_PAYLOAD)

    @pytest.fixture(scope="class")
    def mock_client(self):
        mock_client = MagicMock()
        mock_client.create_entity = MagicMock(
            return_value=self.mock_create_entity_response
        )
        return mock_client

    def test_create_fail(self):
        """Ensures creating model fails in case of wrong input params."""
        with pytest.raises(
            ValueError,
            match="The `compartment_id` must be specified.",
        ):
            OCIFeatureGroupJob().create()

    def test_create_success(self, mock_client):
        """Ensures creating model passes in case of valid input params."""
        with patch.object(OCIFeatureGroupJob, "client", mock_client):
            with patch.object(OCIFeatureGroupJob, "to_oci_model") as mock_to_oci_model:
                with patch.object(
                    OCIFeatureGroupJob, "update_from_oci_model"
                ) as mock_update_from_oci_model:
                    mock_update_from_oci_model.return_value = self.mock_entity
                    mock_oci_entity = FeatureGroupJob(**OCI_FEATURE_GROUP_JOB_PAYLOAD)
                    mock_to_oci_model.return_value = mock_oci_entity
                    result = self.mock_entity.create()
                    mock_client.create_feature_group_job.assert_called_with(
                        mock_oci_entity
                    )
                    assert result == self.mock_entity

    @patch.object(OCIModelMixin, "from_ocid")
    def test_from_id(self, mock_from_ocid):
        """Tests getting a model by OCID."""
        OCIFeatureGroupJob.from_id(FEATURE_GROUP_JOB_OCID)
        mock_from_ocid.assert_called_with(FEATURE_GROUP_JOB_OCID)
