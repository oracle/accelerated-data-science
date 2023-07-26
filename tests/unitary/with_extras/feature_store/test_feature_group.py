#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import hashlib
import json
from copy import deepcopy
from unittest.mock import patch

import pandas
import pytest

from ads.feature_store.common.spark_session_singleton import SparkSessionSingleton
from ads.feature_store.execution_strategy.engine.spark_engine import SparkEngine
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.feature_group_job import FeatureGroupJob
from ads.feature_store.feature_store import FeatureStore
from ads.feature_store.input_feature_detail import FeatureDetail, FeatureType
from ads.feature_store.service.oci_feature_group import OCIFeatureGroup
from ads.feature_store.service.oci_feature_store import OCIFeatureStore
from tests.unitary.with_extras.feature_store.test_feature_group_job import (
    FEATURE_GROUP_JOB_PAYLOAD,
)

FEATURE_GROUP_OCID = "ocid1.featuregroup.oc1.iad.xxx"

FEATURE_GROUP_PAYLOAD = {
    "name": "feature_group",
    "entityId": "ocid1.entity.oc1.iad.xxx",
    "description": "feature group description",
    "primaryKeys": {"items": []},
    "partitionKeys": {"items": []},
    "transformationParameters": "e30=",
    "featureStoreId": "ocid1.featurestore.oc1.iad.xxx",
    "compartmentId": "ocid1.compartment.oc1.iad.xxx",
    "inputFeatureDetails": [
        {"featureType": "STRING", "name": "cc_num"},
        {"featureType": "STRING", "name": "provider"},
        {"featureType": "STRING", "name": "expires"},
    ],
    "isInferSchema": False,
}


@pytest.fixture(autouse=True)
def dataframe_fixture_basic():
    data = {
        "primary_key": [1, 2, 3, 4],
        "event_date": [
            datetime.datetime(2022, 7, 3).date(),
            datetime.datetime(2022, 1, 5).date(),
            datetime.datetime(2022, 1, 6).date(),
            datetime.datetime(2022, 1, 7).date(),
        ],
        "state": ["nevada", None, "nevada", None],
        "measurement": [12.4, 32.5, 342.6, 43.7],
    }

    return pandas.DataFrame(data)


class TestFeatureGroup:
    test_config = {
        "tenancy": "ocid1.tenancy.oc1..xxx",  # must be a real-like ocid
        "user": "ocid1.user.oc1..xxx",  # must be a real-like ocid
        "fingerprint": "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
        "key_file": "<path>/<to>/<key_file>",
        "region": "test_region",
    }

    DEFAULT_PROPERTIES_PAYLOAD = {
        "compartmentId": FEATURE_GROUP_PAYLOAD["compartmentId"],
        "name": FEATURE_GROUP_PAYLOAD["name"],
    }

    def setup_class(cls):
        cls.mock_date = datetime.datetime(3022, 7, 1)

    def setup_method(self):
        self.payload = deepcopy(FEATURE_GROUP_PAYLOAD)
        self.payload_feature_group_job = deepcopy(FEATURE_GROUP_JOB_PAYLOAD)
        self.mock_dsc_feature_group = FeatureGroup(**self.payload)
        self.mock_dsc_feature_group_job = FeatureGroupJob(
            **self.payload_feature_group_job
        )

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
        FeatureGroup,
        "_load_default_properties",
        return_value=DEFAULT_PROPERTIES_PAYLOAD,
    )
    def test__init__default_properties(self, mock_load_default_properties):
        dsc_feature_group = FeatureGroup()
        assert dsc_feature_group.to_dict()["spec"] == self.DEFAULT_PROPERTIES_PAYLOAD

    @patch.object(FeatureGroup, "_load_default_properties", return_value={})
    def test__init__(self, mock_load_default_properties):
        dsc_feature_group = FeatureGroup(**self.payload)
        assert self.prepare_dict(
            dsc_feature_group.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    @patch.object(FeatureGroup, "_load_default_properties", return_value={})
    def test_with_methods_1(self, mock_load_default_properties):
        """Tests all with methods."""

        input_feature_details = [
            FeatureDetail("cc_num").with_feature_type(FeatureType.STRING),
            FeatureDetail("provider").with_feature_type(FeatureType.STRING),
            FeatureDetail("expires").with_feature_type(FeatureType.STRING),
        ]

        dsc_feature_group = (
            FeatureGroup()
            .with_description("feature group description")
            .with_compartment_id(self.payload["compartmentId"])
            .with_name(self.payload["name"])
            .with_entity_id(self.payload["entityId"])
            .with_feature_store_id(self.payload["featureStoreId"])
            .with_primary_keys([])
            .with_partition_keys([])
            .with_transformation_kwargs({})
            .with_input_feature_details(input_feature_details)
        )
        assert self.prepare_dict(
            dsc_feature_group.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    def test_with_methods_2(self):
        """Tests all with methods."""
        input_feature_details = [
            FeatureDetail("cc_num").with_feature_type(FeatureType.STRING),
            FeatureDetail("provider").with_feature_type(FeatureType.STRING),
            FeatureDetail("expires").with_feature_type(FeatureType.STRING),
        ]
        dsc_feature_group = (
            FeatureGroup()
            .with_description("feature group description")
            .with_compartment_id(self.payload["compartmentId"])
            .with_name(self.payload["name"])
            .with_entity_id(self.payload["entityId"])
            .with_feature_store_id(self.payload["featureStoreId"])
            .with_primary_keys([])
            .with_partition_keys([])
            .with_transformation_kwargs({})
            .with_input_feature_details(input_feature_details)
        )
        assert self.prepare_dict(
            dsc_feature_group.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    @patch.object(OCIFeatureGroup, "delete")
    @patch.object(SparkSessionSingleton, "__init__", return_value=None)
    @patch.object(SparkSessionSingleton, "get_spark_session")
    def test_delete(self, spark, get_spark_session, mock_delete):
        """Tests deleting feature group from feature group."""
        with patch.object(FeatureGroupJob, "create"):
            with patch.object(FeatureStore, "from_id"):
                self.mock_dsc_feature_group.delete()

    @patch.object(FeatureGroup, "_update_from_oci_feature_group_model")
    @patch.object(OCIFeatureGroup, "list_resource")
    def test_list(self, mock_list_resource, mock__update_from_oci_fs_model):
        """Tests listing feature group in a given compartment."""
        mock_list_resource.return_value = [OCIFeatureGroup(**FEATURE_GROUP_PAYLOAD)]
        mock__update_from_oci_fs_model.return_value = FeatureGroup(**self.payload)
        result = FeatureGroup.list(
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

    @patch.object(FeatureGroup, "_update_from_oci_feature_group_model")
    @patch.object(OCIFeatureGroup, "from_id")
    def test_from_id(self, mock_oci_from_id, mock__update_from_oci_fs_model):
        """Tests getting an existing model by OCID."""
        mock_oci_model = OCIFeatureGroup(**FEATURE_GROUP_PAYLOAD)
        mock_oci_from_id.return_value = mock_oci_model
        mock__update_from_oci_fs_model.return_value = FeatureGroup(**self.payload)
        result = FeatureGroup.from_id(FEATURE_GROUP_OCID)

        mock_oci_from_id.assert_called_with(FEATURE_GROUP_OCID)
        mock__update_from_oci_fs_model.assert_called_with(mock_oci_model)
        assert self.prepare_dict(result.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(OCIFeatureGroup, "create")
    def test_create_success(
            self,
            mock_oci_dsc_model_create,
    ):
        """Tests creating datascience feature_group."""
        oci_dsc_model = OCIFeatureGroup(**FEATURE_GROUP_PAYLOAD)
        mock_oci_dsc_model_create.return_value = oci_dsc_model

        # to check rundom display name
        self.mock_dsc_feature_group.with_name("")
        result = self.mock_dsc_feature_group.create()
        mock_oci_dsc_model_create.assert_called()

    @patch.object(FeatureGroup, "_load_default_properties", return_value={})
    def test_create_fail(self, mock__load_default_properties):
        """Tests creating datascience feature_group."""
        dsc_feature_group = FeatureGroup()
        with pytest.raises(ValueError, match="Specify compartment OCID."):
            dsc_feature_group.create()

    def test_to_dict(self):
        """Tests serializing feature group to a dictionary."""
        test_dict = self.mock_dsc_feature_group.to_dict()
        assert self.prepare_dict(test_dict["spec"]) == self.prepare_dict(self.payload)
        assert test_dict["kind"] == self.mock_dsc_feature_group.kind
        assert test_dict["type"] == self.mock_dsc_feature_group.type

    def test_from_dict(self):
        """Tests loading feature group instance from a dictionary of configurations."""
        assert self.prepare_dict(
            self.mock_dsc_feature_group.to_dict()["spec"]
        ) == self.prepare_dict(
            FeatureGroup.from_dict({"spec": self.payload}).to_dict()["spec"]
        )

    @patch("ads.common.utils.get_random_name_for_resource", return_value="test_name")
    def test__random_name(self, mock_get_random_name_for_resource):
        """Tests generating a random display name."""
        expected_result = f"{self.mock_dsc_feature_group._PREFIX}-test_name"
        assert self.mock_dsc_feature_group._random_display_name() == expected_result

    @patch("oci.config.from_file", return_value=test_config)
    @patch("oci.signer.load_private_key_from_file")
    def test__to_oci_fs_entity(self, mock_load_key_file, mock_config_from_file):
        """Tests creating an `OCIFeatureGroup` instance from the  `FeatureGroup`."""
        with patch.object(OCIFeatureGroup, "sync"):
            test_oci_dsc_feature_group = OCIFeatureGroup(**FEATURE_GROUP_PAYLOAD)
            test_oci_dsc_feature_group.id = None
            test_oci_dsc_feature_group.lifecycle_state = None
            test_oci_dsc_feature_group.created_by = None
            test_oci_dsc_feature_group.time_created = None

            assert self.prepare_dict(
                test_oci_dsc_feature_group.to_dict()
            ) == self.prepare_dict(
                self.mock_dsc_feature_group._to_oci_feature_group().to_dict()
            )

            test_oci_dsc_feature_group.name = "new_name"
            assert self.prepare_dict(
                test_oci_dsc_feature_group.to_dict()
            ) == self.prepare_dict(
                self.mock_dsc_feature_group._to_oci_feature_group(
                    name="new_name"
                ).to_dict()
            )

    @patch.object(OCIFeatureGroup, "update")
    @patch.object(SparkSessionSingleton, "__init__", return_value=None)
    @patch.object(SparkSessionSingleton, "get_spark_session")
    def test_materialise(
            self, spark_session, get_spark_session, mocke_update, dataframe_fixture_basic
    ):
        with patch.object(FeatureGroupJob, "create") as mock_feature_group_job:
            with patch.object(FeatureStore, "from_id"):
                with patch.object(FeatureGroupJob, "_mark_job_complete"):
                    mock_feature_group_job.return_value = (
                        self.mock_dsc_feature_group_job
                    )
                    self.mock_dsc_feature_group.with_id(FEATURE_GROUP_OCID)
                    self.mock_dsc_feature_group.materialise(dataframe_fixture_basic)

    @patch.object(SparkSessionSingleton, "__init__", return_value=None)
    @patch.object(SparkSessionSingleton, "get_spark_session")
    @patch.object(OCIFeatureStore, "from_id")
    def test_preview(self, feature_store, spark_session, get_spark_session):
        with patch.object(SparkEngine, "sql") as mock_execution_strategy:
            mock_execution_strategy.return_value = None
            self.mock_dsc_feature_group.preview()
            mock_execution_strategy.assert_called_once()

    @patch.object(SparkSessionSingleton, "__init__", return_value=None)
    @patch.object(SparkSessionSingleton, "get_spark_session")
    @patch.object(OCIFeatureStore, "from_id")
    def test_profile(self, feature_store, spark_session, get_spark_session):
        with patch.object(SparkEngine, "sql") as mock_execution_strategy:
            mock_execution_strategy.return_value = None
            self.mock_dsc_feature_group.profile()
            mock_execution_strategy.assert_called_once()

    @patch.object(SparkSessionSingleton, "__init__", return_value=None)
    @patch.object(SparkSessionSingleton, "get_spark_session")
    @patch.object(OCIFeatureStore, "from_id")
    def test_history(self, feature_store, spark_session, get_spark_session):
        with patch.object(SparkEngine, "sql") as mock_execution_strategy:
            mock_execution_strategy.return_value = None
            self.mock_dsc_feature_group.history()
            mock_execution_strategy.assert_called_once()

    @patch.object(OCIFeatureGroup, "update")
    @patch.object(SparkSessionSingleton, "__init__", return_value=None)
    @patch.object(SparkSessionSingleton, "get_spark_session")
    @patch.object(OCIFeatureStore, "from_id")
    def test_restore(
            self, feature_store, spark_session, get_spark_session, mock_update
    ):
        with patch.object(SparkEngine, "sql") as mock_execution_strategy:
            mock_execution_strategy.return_value = None
            self.mock_dsc_feature_group.with_id(FEATURE_GROUP_OCID)
            self.mock_dsc_feature_group.restore(1)
            mock_execution_strategy.assert_called_once()
