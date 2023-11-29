#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from unittest import mock

from ads.model.model_properties import ModelProperties

mock_env_variables = {
    "PROJECT_OCID": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
    "JOB_RUN_OCID": "ocid1.datasciencejobrun.oc1.iad.<unique_ocid>",
    "NB_SESSION_OCID": "ocid1.datasciencenotebook.oc1.iad.<unique_ocid>",
    "NB_SESSION_COMPARTMENT_OCID": "ocid1.compartment.oc1..<unique_ocid>",
}


class TestModelProperties:
    @mock.patch.dict(os.environ, mock_env_variables, clear=True)
    def test__adjust_with_env(self):
        """Tests adjustment env variables."""
        model_properties = ModelProperties()
        assert model_properties.project_id == None
        assert model_properties.training_resource_id == None
        assert model_properties.compartment_id == None
        model_properties._adjust_with_env()
        assert model_properties.project_id == mock_env_variables["PROJECT_OCID"]
        assert (
            model_properties.training_resource_id == mock_env_variables["JOB_RUN_OCID"]
        )
        assert (
            model_properties.compartment_id
            == mock_env_variables["NB_SESSION_COMPARTMENT_OCID"]
        )
