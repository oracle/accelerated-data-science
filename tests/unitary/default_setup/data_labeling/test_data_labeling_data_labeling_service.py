#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
from unittest.mock import patch

import pytest
from ads.common.object_storage_details import InvalidObjectStoragePath
from ads.data_labeling.data_labeling_service import DataLabeling
from ads.common import auth, oci_client


class TestDataLabelingUnitTest(unittest.TestCase):

    BUCKET_NAME = "testbucket"
    NAMESPACE = "testnamespace"
    PREFIX = "testprefix"
    PREFIX_JSON = "/testprefix/testfile.json"
    compartment_id = "ocid1.compartment.oc1..<unique_ocid>"
    with patch.object(auth, "default_signer"):
        with patch.object(oci_client, "OCIClientFactory"):
            dls = DataLabeling(compartment_id=compartment_id)
    dataset_id = "ocid1.datalabelingdataset.oc1.iad.<unique_ocid>"

    @patch("ads.data_labeling.data_labeling_service.NB_SESSION_COMPARTMENT_OCID", None)
    def test_data_labeling_init_missing_compartment_id(self):

        with pytest.raises(ValueError):
            DataLabeling()

    def test_data_labeling_init(self):
        assert self.dls.compartment_id == self.compartment_id

    def test_export_missing_bucket(self):
        oci_path = "oci://@testnamespace"
        with pytest.raises(
            InvalidObjectStoragePath, match=r".*oci://<bucket_name>@<namespace>/key*"
        ):
            self.dls.export(self.dataset_id, oci_path)

    def test_export_missing_namespace(self):
        oci_path = "oci://testbucket@/path"
        with pytest.raises(
            InvalidObjectStoragePath, match=r".*oci://<bucket_name>@<namespace>/key*"
        ):
            self.dls.export(self.dataset_id, oci_path)
