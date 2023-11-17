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

from ads.feature_store.feature_store import FeatureStore
from ads.feature_store.service.oci_feature_store import OCIFeatureStore

FEATURE_STORE_OCID = "ocid1.featurestore.oc1.iad.xxx"

FEATURE_STORE_PAYLOAD = {
    "displayName": "feature_store_name",
    "description": "desc",
    "compartmentId": "compartmentId",
    "offlineConfig": {"metastoreId": "ocid1.datacatalogmetastore.oc1.iad.xxx"},
}
COMPARTMENT_ID_ERROR = "Specify compartment OCID."

class TestFeatureStore:
    test_config = {
        "tenancy": "ocid1.tenancy.oc1..xxx",  # must be a real-like ocid
        "user": "ocid1.user.oc1..xxx",  # must be a real-like ocid
        "fingerprint": "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
        "key_file": "<path>/<to>/<key_file>",
        "region": "test_region",
    }

    DEFAULT_PROPERTIES_PAYLOAD = {
        "compartmentId": FEATURE_STORE_PAYLOAD["compartmentId"],
        "displayName": FEATURE_STORE_PAYLOAD["displayName"],
    }

    def setup_class(cls):
        cls.mock_date = datetime.datetime(3022, 7, 1)

    def setup_method(self):
        self.payload = deepcopy(FEATURE_STORE_PAYLOAD)
        self.mock_dsc_feature_store = FeatureStore(**self.payload)

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
        FeatureStore,
        "_load_default_properties",
        return_value=DEFAULT_PROPERTIES_PAYLOAD,
    )
    def test__init__default_properties(self, mock_load_default_properties):
        dsc_feature_store = FeatureStore()
        assert dsc_feature_store.to_dict()["spec"] == self.DEFAULT_PROPERTIES_PAYLOAD

    @patch.object(FeatureStore, "_load_default_properties", return_value={})
    def test__init__(self, mock_load_default_properties):
        dsc_feature_store = FeatureStore(**self.payload)
        assert self.prepare_dict(
            dsc_feature_store.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    @patch.object(FeatureStore, "_load_default_properties", return_value={})
    def test_with_methods_1(self, mock_load_default_properties):
        """Tests all with methods."""
        dsc_feature_store = (
            FeatureStore()
            .with_compartment_id(self.payload["compartmentId"])
            .with_display_name(self.payload["displayName"])
            .with_description(self.payload["description"])
            .with_offline_config(self.payload["offlineConfig"].get("metastoreId") or {})
        )
        assert self.prepare_dict(
            dsc_feature_store.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    def test_with_methods_2(self):
        """Tests all with methods."""
        dsc_feature_store = (
            FeatureStore()
            .with_compartment_id(self.payload["compartmentId"])
            .with_display_name(self.payload["displayName"])
            .with_description(self.payload["description"])
            .with_offline_config(self.payload["offlineConfig"].get("metastoreId") or {})
        )
        assert self.prepare_dict(
            dsc_feature_store.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    @patch.object(OCIFeatureStore, "delete")
    def test_delete(self, mock_delete):
        """Tests deleting feature store from feature store."""
        self.mock_dsc_feature_store.delete()

    @patch.object(FeatureStore, "_update_from_oci_fs_model")
    @patch.object(OCIFeatureStore, "list_resource")
    def test_list(self, mock_list_resource, mock__update_from_oci_fs_model):
        """Tests listing feature store in a given compartment."""
        mock_list_resource.return_value = [OCIFeatureStore(**FEATURE_STORE_PAYLOAD)]
        mock__update_from_oci_fs_model.return_value = FeatureStore(**self.payload)
        result = FeatureStore.list(
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

    @patch.object(FeatureStore, "_update_from_oci_fs_model")
    @patch.object(OCIFeatureStore, "from_id")
    def test_from_id(self, mock_oci_from_id, mock__update_from_oci_fs_model):
        """Tests getting an existing model by OCID."""
        mock_oci_model = OCIFeatureStore(**FEATURE_STORE_PAYLOAD)
        mock_oci_from_id.return_value = mock_oci_model
        mock__update_from_oci_fs_model.return_value = FeatureStore(**self.payload)
        result = FeatureStore.from_id(FEATURE_STORE_OCID)

        mock_oci_from_id.assert_called_with(FEATURE_STORE_OCID)
        mock__update_from_oci_fs_model.assert_called_with(mock_oci_model)
        assert self.prepare_dict(result.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(OCIFeatureStore, "create")
    def test_create_success(
        self,
        mock_oci_dsc_model_create,
    ):
        """Tests creating datascience feature_store."""
        oci_dsc_model = OCIFeatureStore(**FEATURE_STORE_PAYLOAD)
        mock_oci_dsc_model_create.return_value = oci_dsc_model

        # to check rundom display name
        self.mock_dsc_feature_store.with_display_name("")
        result = self.mock_dsc_feature_store.create()
        mock_oci_dsc_model_create.assert_called()

    @patch.object(FeatureStore, "_load_default_properties", return_value={})
    def test_create_fail(self, mock__load_default_properties):
        """Tests creating datascience feature_store."""
        dsc_feature_store = FeatureStore()
        with pytest.raises(ValueError, match=COMPARTMENT_ID_ERROR):
            dsc_feature_store.create()

    def test_to_dict(self):
        """Tests serializing feature store to a dictionary."""
        test_dict = self.mock_dsc_feature_store.to_dict()
        assert self.prepare_dict(test_dict["spec"]) == self.prepare_dict(self.payload)
        assert test_dict["kind"] == self.mock_dsc_feature_store.kind
        assert test_dict["type"] == self.mock_dsc_feature_store.type

    def test_from_dict(self):
        """Tests loading feature store instance from a dictionary of configurations."""
        assert self.prepare_dict(
            self.mock_dsc_feature_store.to_dict()["spec"]
        ) == self.prepare_dict(
            FeatureStore.from_dict({"spec": self.payload}).to_dict()["spec"]
        )

    @patch("ads.common.utils.get_random_name_for_resource", return_value="test_name")
    def test__random_display_name(self, mock_get_random_name_for_resource):
        """Tests generating a random display name."""
        expected_result = f"{self.mock_dsc_feature_store._PREFIX}-test_name"
        assert self.mock_dsc_feature_store._random_display_name() == expected_result

    @patch("oci.config.from_file", return_value=test_config)
    @patch("oci.signer.load_private_key_from_file")
    def test__to_oci_fs_entity(self, mock_load_key_file, mock_config_from_file):
        """Tests creating an `OCIFeatureStore` instance from the  `FeatureStore`."""
        with patch.object(OCIFeatureStore, "sync"):
            test_oci_dsc_feature_store = OCIFeatureStore(**FEATURE_STORE_PAYLOAD)
            test_oci_dsc_feature_store.id = None
            test_oci_dsc_feature_store.lifecycle_state = None
            test_oci_dsc_feature_store.created_by = None
            test_oci_dsc_feature_store.time_created = None

            assert self.prepare_dict(
                test_oci_dsc_feature_store.to_dict()
            ) == self.prepare_dict(self.mock_dsc_feature_store._to_oci_fs().to_dict())

            test_oci_dsc_feature_store.display_name = "new_name"
            assert self.prepare_dict(
                test_oci_dsc_feature_store.to_dict()
            ) == self.prepare_dict(
                self.mock_dsc_feature_store._to_oci_fs(
                    display_name="new_name"
                ).to_dict()
            )
