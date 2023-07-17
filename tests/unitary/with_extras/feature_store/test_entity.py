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

from ads.feature_store.entity import Entity
from ads.feature_store.service.oci_entity import OCIEntity

ENTITY_OCID = "ocid1.entity.oc1.iad.xxx"

ENTITY_PAYLOAD = {
    "name": "entity",
    "compartmentId": "ocid1.compartment.oc1.iad.xxx",
    "description": "entity description",
    "featureStoreId": "ocid1.featurestore.oc1.iad.xxx",
}


class TestEntity:
    test_config = {
        "tenancy": "ocid1.tenancy.oc1..xxx",  # must be a real-like ocid
        "user": "ocid1.user.oc1..xxx",  # must be a real-like ocid
        "fingerprint": "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
        "key_file": "<path>/<to>/<key_file>",
        "region": "test_region",
    }

    DEFAULT_PROPERTIES_PAYLOAD = {
        "compartmentId": ENTITY_PAYLOAD["compartmentId"],
        "name": ENTITY_PAYLOAD["name"],
    }

    def setup_class(cls):
        cls.mock_date = datetime.datetime(3022, 7, 1)

    def setup_method(self):
        self.payload = deepcopy(ENTITY_PAYLOAD)
        self.mock_dsc_entity = Entity(**self.payload)

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
        Entity,
        "_load_default_properties",
        return_value=DEFAULT_PROPERTIES_PAYLOAD,
    )
    def test__init__default_properties(self, mock_load_default_properties):
        dsc_entity = Entity()
        assert dsc_entity.to_dict()["spec"] == self.DEFAULT_PROPERTIES_PAYLOAD

    @patch.object(Entity, "_load_default_properties", return_value={})
    def test__init__(self, mock_load_default_properties):
        dsc_entity = Entity(**self.payload)
        assert self.prepare_dict(dsc_entity.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(Entity, "_load_default_properties", return_value={})
    def test_with_methods_1(self, mock_load_default_properties):
        """Tests all with methods."""
        dsc_entity = (
            Entity()
            .with_description("entity description")
            .with_compartment_id(self.payload["compartmentId"])
            .with_name(self.payload["name"])
            .with_feature_store_id(self.payload["featureStoreId"])
        )
        assert self.prepare_dict(dsc_entity.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    def test_with_methods_2(self):
        """Tests all with methods."""
        dsc_entity = (
            Entity()
            .with_description("entity description")
            .with_compartment_id(self.payload["compartmentId"])
            .with_name(self.payload["name"])
            .with_feature_store_id(self.payload["featureStoreId"])
        )
        assert self.prepare_dict(dsc_entity.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(OCIEntity, "delete")
    def test_delete(self, mock_delete):
        """Tests deleting entity from entity."""
        self.mock_dsc_entity.delete()

    @patch.object(Entity, "_update_from_oci_fs_entity_model")
    @patch.object(OCIEntity, "list_resource")
    def test_list(self, mock_list_resource, mock__update_from_oci_fs_model):
        """Tests listing entity in a given compartment."""
        mock_list_resource.return_value = [OCIEntity(**ENTITY_PAYLOAD)]
        mock__update_from_oci_fs_model.return_value = Entity(**self.payload)
        result = Entity.list(
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

    @patch.object(Entity, "_update_from_oci_fs_entity_model")
    @patch.object(OCIEntity, "from_id")
    def test_from_id(self, mock_oci_from_id, mock__update_from_oci_fs_model):
        """Tests getting an existing model by OCID."""
        mock_oci_model = OCIEntity(**ENTITY_PAYLOAD)
        mock_oci_from_id.return_value = mock_oci_model
        mock__update_from_oci_fs_model.return_value = Entity(**self.payload)
        result = Entity.from_id(ENTITY_OCID)

        mock_oci_from_id.assert_called_with(ENTITY_OCID)
        mock__update_from_oci_fs_model.assert_called_with(mock_oci_model)
        assert self.prepare_dict(result.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(OCIEntity, "create")
    def test_create_success(
        self,
        mock_oci_dsc_model_create,
    ):
        """Tests creating datascience entity."""
        oci_dsc_model = OCIEntity(**ENTITY_PAYLOAD)
        mock_oci_dsc_model_create.return_value = oci_dsc_model

        # to check random display name
        self.mock_dsc_entity.with_name("")
        result = self.mock_dsc_entity.create()
        mock_oci_dsc_model_create.assert_called()

    @patch.object(Entity, "_load_default_properties", return_value={})
    def test_create_fail(self, mock__load_default_properties):
        """Tests creating datascience entity."""
        dsc_entity = Entity()
        with pytest.raises(ValueError, match="Specify compartment OCID."):
            dsc_entity.create()

    def test_to_dict(self):
        """Tests serializing entity to a dictionary."""
        test_dict = self.mock_dsc_entity.to_dict()
        assert self.prepare_dict(test_dict["spec"]) == self.prepare_dict(self.payload)
        assert test_dict["kind"] == self.mock_dsc_entity.kind
        assert test_dict["type"] == self.mock_dsc_entity.type

    def test_from_dict(self):
        """Tests loading entity instance from a dictionary of configurations."""
        assert self.prepare_dict(
            self.mock_dsc_entity.to_dict()["spec"]
        ) == self.prepare_dict(
            Entity.from_dict({"spec": self.payload}).to_dict()["spec"]
        )

    @patch("ads.common.utils.get_random_name_for_resource", return_value="test_name")
    def test__random_name(self, mock_get_random_name_for_resource):
        """Tests generating a random display name."""
        expected_result = f"{self.mock_dsc_entity._PREFIX}-test_name"
        assert self.mock_dsc_entity._random_display_name() == expected_result

    @patch("oci.config.from_file", return_value=test_config)
    @patch("oci.signer.load_private_key_from_file")
    def test__to_oci_fs_entity(self, mock_load_key_file, mock_config_from_file):
        """Tests creating an `OCIEntity` instance from the  `Entity`."""
        with patch.object(OCIEntity, "sync"):
            test_oci_dsc_entity = OCIEntity(**ENTITY_PAYLOAD)
            test_oci_dsc_entity.id = None
            test_oci_dsc_entity.lifecycle_state = None
            test_oci_dsc_entity.created_by = None
            test_oci_dsc_entity.time_created = None

            assert self.prepare_dict(
                test_oci_dsc_entity.to_dict()
            ) == self.prepare_dict(self.mock_dsc_entity._to_oci_fs_entity().to_dict())

            test_oci_dsc_entity.name = "new_name"
            assert self.prepare_dict(
                test_oci_dsc_entity.to_dict()
            ) == self.prepare_dict(
                self.mock_dsc_entity._to_oci_fs_entity(name="new_name").to_dict()
            )
