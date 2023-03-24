#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import oci
from unittest.mock import MagicMock, patch
from ads.common.oci_resource import OCIResource, SEARCH_TYPE


class TestOCIResource:
    """Contains test cases for ads.common.oci_resource.py"""

    model_id = "ocid1.modelcatalog.oc1.<unique_ocid>"
    model_deployment_list = [
        oci.resource_search.models.resource_summary.ResourceSummary(
            compartment_id="ocid1.compartment.<unique_ocid>",
            display_name="RBH-Demo-Deployment",
            identifier="ocid1.datasciencemodeldeployment.<unique_ocid>",
            lifecycle_state="ACTIVE",
            resource_type="DataScienceModelDeployment",
            time_created="2021-06-01T18:40:38.586000+00:00",
        )
    ]
    model_deployment_data = oci.resource_search.models.ResourceSummaryCollection(
        items=model_deployment_list
    )

    @patch.object(oci.resource_search.ResourceSearchClient, "search_resources")
    def test_search_structured(self, mock_search_resources):
        """Test OCIResource.search."""
        mock_search_resources.return_value = MagicMock()
        mock_search_resources.return_value.data = self.model_deployment_data
        query = f"query datasciencemodeldeployment resources where ModelId='{self.model_id}'"
        model_deployments = OCIResource.search(query, type=SEARCH_TYPE.STRUCTURED)
        assert len(model_deployments) > 0
        assert isinstance(model_deployments, list)
        assert model_deployments[0].resource_type == "DataScienceModelDeployment"
