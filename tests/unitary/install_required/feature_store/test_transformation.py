#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import hashlib
import json
from copy import deepcopy
from unittest.mock import patch

import pytest

from ads.feature_store.transformation import Transformation
from ads.feature_store.service.oci_transformation import OCITransformation


TRANSFORMATION_OCID = "ocid1.transformation.oc1.iad.xxx"

TRANSFORMATION_PAYLOAD = {
    "displayName": "transformation",
    "compartmentId": "ocid1.compartment.oc1.iad.xxx",
    "description": "transformation description",
    "featureStoreId": "ocid1.featurestore.oc1.iad.xxx",
}


class TestTransformation:
    DEFAULT_PROPERTIES_PAYLOAD = {
        "compartmentId": TRANSFORMATION_PAYLOAD["compartmentId"],
        "displayName": TRANSFORMATION_PAYLOAD["displayName"],
    }

    def setup_class(cls):
        cls.mock_date = datetime.datetime(3022, 7, 1)

    def setup_method(self):
        self.payload = deepcopy(TRANSFORMATION_PAYLOAD)
        self.mock_dsc_transformation = Transformation(**self.payload)

    def prepare_dict(self, data):
        return data

    def hash_dict(self, data):
        return hashlib.sha1(
            json.dumps(self.prepare_dict(data), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def compare_dict(self, dict1, dict2):
        print(
            f"dict1_hash: {self.hash_dict(dict1)}; dict2_hash: {self.hash_dict(dict2)}"
        )
        return self.hash_dict(dict1) == self.hash_dict(dict2)

    @patch.object(
        Transformation,
        "_load_default_properties",
        return_value=DEFAULT_PROPERTIES_PAYLOAD,
    )
    def test__init__default_properties(self, mock_load_default_properties):
        dsc_transformation = Transformation()
        assert dsc_transformation.to_dict()["spec"] == self.DEFAULT_PROPERTIES_PAYLOAD

    @patch.object(Transformation, "_load_default_properties", return_value={})
    def test__init__(self, mock_load_default_properties):
        dsc_transformation = Transformation(**self.payload)
        assert self.prepare_dict(
            dsc_transformation.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    @patch.object(Transformation, "_load_default_properties", return_value={})
    def test_with_methods_1(self, mock_load_default_properties):
        """Tests all with methods."""
        dsc_transformation = (
            Transformation()
            .with_description("transformation description")
            .with_compartment_id(self.payload["compartmentId"])
            .with_display_name(self.payload["displayName"])
            .with_feature_store_id(self.payload["featureStoreId"])
        )
        assert self.prepare_dict(
            dsc_transformation.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    def test_with_methods_2(self):
        """Tests all with methods."""
        dsc_transformation = (
            Transformation()
            .with_description("transformation description")
            .with_compartment_id(self.payload["compartmentId"])
            .with_display_name(self.payload["displayName"])
            .with_feature_store_id(self.payload["featureStoreId"])
        )
        assert self.prepare_dict(
            dsc_transformation.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    @patch.object(OCITransformation, "delete")
    def test_delete(self, mock_delete):
        """Tests deleting transformation from transformation."""
        self.mock_dsc_transformation.delete()

    @patch.object(Transformation, "_update_from_oci_fs_transformation_model")
    @patch.object(OCITransformation, "list_resource")
    def test_list(self, mock_list_resource, mock__update_from_oci_fs_model):
        """Tests listing transformation in a given compartment."""
        mock_list_resource.return_value = [OCITransformation(**TRANSFORMATION_PAYLOAD)]
        mock__update_from_oci_fs_model.return_value = Transformation(**self.payload)
        result = Transformation.list(
            compartment_id="test_compartment_id",
            extra_tag="test_cvalue",
        )
        mock_list_resource.assert_called_with(
            "test_compartment_id",
            **{"extra_tag": "test_cvalue"},
        )
        assert len(result) == 1
        assert self.prepare_dict(result[0].to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(Transformation, "_update_from_oci_fs_transformation_model")
    @patch.object(OCITransformation, "from_id")
    def test_from_id(self, mock_oci_from_id, mock__update_from_oci_fs_model):
        """Tests getting an existing model by OCID."""
        mock_oci_model = OCITransformation(**TRANSFORMATION_PAYLOAD)
        mock_oci_from_id.return_value = mock_oci_model
        mock__update_from_oci_fs_model.return_value = Transformation(**self.payload)
        result = Transformation.from_id(TRANSFORMATION_OCID)

        mock_oci_from_id.assert_called_with(TRANSFORMATION_OCID)
        mock__update_from_oci_fs_model.assert_called_with(mock_oci_model)
        assert self.prepare_dict(result.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(OCITransformation, "create")
    def test_create_success(
        self,
        mock_oci_dsc_model_create,
    ):
        """Tests creating datascience transformation."""
        oci_dsc_model = OCITransformation(**TRANSFORMATION_PAYLOAD)
        mock_oci_dsc_model_create.return_value = oci_dsc_model

        # to check random display name
        self.mock_dsc_transformation.with_display_name("")
        result = self.mock_dsc_transformation.create()
        mock_oci_dsc_model_create.assert_called()

    @patch.object(Transformation, "_load_default_properties", return_value={})
    def test_create_fail(self, mock__load_default_properties):
        """Tests creating datascience transformation."""
        dsc_transformation = Transformation()
        with pytest.raises(ValueError, match="Specify compartment OCID."):
            dsc_transformation.create()

    def test_to_dict(self):
        """Tests serializing transformation to a dictionary."""
        test_dict = self.mock_dsc_transformation.to_dict()
        assert self.prepare_dict(test_dict["spec"]) == self.prepare_dict(self.payload)
        assert test_dict["kind"] == self.mock_dsc_transformation.kind
        assert test_dict["type"] == self.mock_dsc_transformation.type

    def test_from_dict(self):
        """Tests loading transformation instance from a dictionary of configurations."""
        assert self.prepare_dict(
            self.mock_dsc_transformation.to_dict()["spec"]
        ) == self.prepare_dict(
            Transformation.from_dict({"spec": self.payload}).to_dict()["spec"]
        )

    @patch("ads.common.utils.get_random_name_for_resource", return_value="test_name")
    def test__random_display_name(self, mock_get_random_name_for_resource):
        """Tests generating a random display name."""
        expected_result = f"{self.mock_dsc_transformation._PREFIX}-test_name"
        assert self.mock_dsc_transformation._random_display_name() == expected_result
