#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from copy import deepcopy
from unittest.mock import patch

from ads.feature_store.common.feature_store_singleton import SparkSessionSingleton
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.feature_store import FeatureStore
from ads.feature_store.query.query import Query
import pytest

FEATURE_GROUP_OCID_1 = "ocid1.featuregroup1.oc1.iad.xxx"

FEATURE_GROUP_PAYLOAD_1 = {
    "id": FEATURE_GROUP_OCID_1,
    "name": "feature_group1",
    "entityId": "ocid1.entity.oc1.iad.xxx",
    "description": "feature group description",
    "primaryKeys": {"items": []},
    "featureStoreId": "ocid1.featurestore.oc1.iad.xxx",
    "compartmentId": "ocid1.compartment.oc1.iad.xxx",
    "inputFeatureDetails": {
        "items": [
            {"featureType": "STRING", "name": "cc_num"},
            {"featureType": "STRING", "name": "provider"},
            {"featureType": "STRING", "name": "expires"},
        ]
    },
    "outputFeatureDetails": {
        "items": [
            {
                "featureType": "STRING",
                "name": "cc_num",
                "featureGroupId": FEATURE_GROUP_OCID_1,
            },
            {
                "featureType": "STRING",
                "name": "provider",
                "featureGroupId": FEATURE_GROUP_OCID_1,
            },
            {
                "featureType": "STRING",
                "name": "expires",
                "featureGroupId": FEATURE_GROUP_OCID_1,
            },
        ]
    },
}

FEATURE_GROUP_OCID_2 = "ocid1.featuregroup1.oc1.iad.xxx"

FEATURE_GROUP_PAYLOAD_2 = {
    "id": FEATURE_GROUP_OCID_2,
    "name": "feature_group2",
    "entityId": "ocid1.entity.oc1.iad.xxx",
    "description": "feature group description",
    "primaryKeys": {"items": []},
    "featureStoreId": "ocid1.featurestore.oc1.iad.xxx",
    "compartmentId": "ocid1.compartment.oc1.iad.xxx",
    "inputFeatureDetails": {
        "items": [
            {"featureType": "STRING", "name": "cc_num"},
            {"featureType": "STRING", "name": "provider"},
            {"featureType": "STRING", "name": "expires"},
        ]
    },
    "outputFeatureDetails": {
        "items": [
            {
                "featureType": "STRING",
                "name": "cc_num",
                "featureGroupId": FEATURE_GROUP_OCID_2,
            },
            {
                "featureType": "STRING",
                "name": "provider",
                "featureGroupId": FEATURE_GROUP_OCID_2,
            },
            {
                "featureType": "STRING",
                "name": "expires",
                "featureGroupId": FEATURE_GROUP_OCID_2,
            },
        ],
    },
}

FEATURE_GROUP_OCID_3 = "ocid1.featuregroup1.oc1.iad.xxx"

FEATURE_GROUP_PAYLOAD_3 = {
    "id": FEATURE_GROUP_OCID_3,
    "name": "feature_group3",
    "entityId": "ocid1.entity.oc1.iad.xxx",
    "description": "feature group description",
    "primaryKeys": {"items": []},
    "featureStoreId": "ocid1.featurestore.oc1.iad.xxx",
    "compartmentId": "ocid1.compartment.oc1.iad.xxx",
    "inputFeatureDetails": {
        "items": [
            {"featureType": "STRING", "name": "X"},
            {"featureType": "STRING", "name": "Y"},
            {"featureType": "STRING", "name": "Z"},
        ]
    },
    "outputFeatureDetails": {
        "items": [
            {
                "featureType": "STRING",
                "name": "X",
                "featureGroupId": FEATURE_GROUP_OCID_3,
            },
            {
                "featureType": "STRING",
                "name": "Y",
                "featureGroupId": FEATURE_GROUP_OCID_3,
            },
            {
                "featureType": "STRING",
                "name": "Z",
                "featureGroupId": FEATURE_GROUP_OCID_3,
            },
        ],
    },
}


class TestQuery:
    def setup_method(self):
        self.payload_1 = deepcopy(FEATURE_GROUP_PAYLOAD_1)
        self.mock_dsc_feature_group_1 = FeatureGroup(**self.payload_1)
        self.payload_2 = deepcopy(FEATURE_GROUP_PAYLOAD_2)
        self.mock_dsc_feature_group_2 = FeatureGroup(**self.payload_2)
        self.payload_3 = deepcopy(FEATURE_GROUP_PAYLOAD_3)
        self.mock_dsc_feature_group_3 = FeatureGroup(**self.payload_3)
        self.feature_store_id = "ocid1.featurestore.oc1.iad.xxx"
        self.entity_id = "ocid1.entity.oc1.iad.xxx"

    @patch.object(SparkSessionSingleton, "get_spark_session")
    @patch.object(FeatureStore, "from_id")
    def test_select_subset_query(self, spark_session, mock_feature_store):
        """Tests with simple select query."""
        dsc_query = Query(
            left_feature_group=self.mock_dsc_feature_group_1,
            left_features=["cc_num"],
            feature_store_id=self.feature_store_id,
            entity_id=self.entity_id,
        )
        assert (
            dsc_query.to_string()
            == "SELECT fg_0.cc_num cc_num FROM `ocid1.entity.oc1.iad.xxx`.feature_group1 fg_0"
        )

    @patch.object(SparkSessionSingleton, "get_spark_session")
    @patch.object(FeatureStore, "from_id")
    def test_select_all_query(self, spark_session, mock_feature_store):
        """Tests with simple select all query."""
        dsc_query = Query(
            left_feature_group=self.mock_dsc_feature_group_1,
            left_features=[],
            feature_store_id=self.feature_store_id,
            entity_id=self.entity_id,
        )
        assert (
            dsc_query.to_string()
            == "SELECT fg_0.cc_num cc_num, fg_0.provider provider, fg_0.expires expires "
            "FROM `ocid1.entity.oc1.iad.xxx`.feature_group1 fg_0"
        )

    @patch.object(SparkSessionSingleton, "get_spark_session")
    @patch.object(FeatureStore, "from_id")
    def test_select_subset_two_columns_query(self, spark_session, mock_feature_store):
        """Tests with simple select query."""
        dsc_query = Query(
            left_feature_group=self.mock_dsc_feature_group_1,
            left_features=["cc_num", "provider"],
            feature_store_id=self.feature_store_id,
            entity_id=self.entity_id,
        )
        assert (
            dsc_query.to_string()
            == "SELECT fg_0.cc_num cc_num, fg_0.provider provider "
            "FROM `ocid1.entity.oc1.iad.xxx`.feature_group1 fg_0"
        )

    @patch.object(SparkSessionSingleton, "__init__", return_value=None)
    @patch.object(SparkSessionSingleton, "get_spark_session")
    @patch.object(FeatureStore, "from_id")
    def test_select_subset_with_no_matching_column(
        self, spark_session, get_spark_session, mock_feature_store
    ):
        """Tests with select and join query with not matching column"""
        with pytest.raises(
            ValueError,
            match="Cannot join feature group '"
            + self.mock_dsc_feature_group_1.name
            + "' on 'X', as it is not present in the feature group. ",
        ):
            dsc_query = Query(
                left_feature_group=self.mock_dsc_feature_group_1,
                left_features=["cc_num", "provider"],
                feature_store_id=self.feature_store_id,
                entity_id=self.entity_id,
            ).join(
                Query(
                    left_feature_group=self.mock_dsc_feature_group_3,
                    left_features=["X", "Y"],
                    feature_store_id=self.feature_store_id,
                    entity_id=self.entity_id,
                ),
                on=["X"],
            )
        # assert dsc_query.to_string() == (
        #     "SELECT fg_1.cc_num cc_num, fg_1.provider provider, fg_0.X X, fg_0.Y Y FROM "
        #     "`ocid1.entity.oc1.iad.xxx`.feature_group1 fg_1 INNER JOIN "
        #     "`ocid1.entity.oc1.iad.xxx`.feature_group3 fg_0 ON fg_1.X = fg_0.X"
        # )

    @patch.object(SparkSessionSingleton, "__init__", return_value=None)
    @patch.object(SparkSessionSingleton, "get_spark_session")
    @patch.object(FeatureStore, "from_id")
    def test_select_subset_with_no_matching_column_without_on(
        self, spark_session, get_spark_session, mock_feature_store
    ):
        """Tests with select and join query with not matching column"""
        with pytest.raises(
            ValueError,
            match="Cannot join feature groups '"
            + self.mock_dsc_feature_group_1.name
            + "' and '"
            + self.mock_dsc_feature_group_3.name
            + "', as no matching primary keys were found.",
        ):
            dsc_query = Query(
                left_feature_group=self.mock_dsc_feature_group_1,
                left_features=["cc_num", "provider"],
                feature_store_id=self.feature_store_id,
                entity_id=self.entity_id,
            ).join(
                Query(
                    left_feature_group=self.mock_dsc_feature_group_3,
                    left_features=["X", "Y"],
                    feature_store_id=self.feature_store_id,
                    entity_id=self.entity_id,
                )
            )

    @patch.object(SparkSessionSingleton, "__init__", return_value=None)
    @patch.object(SparkSessionSingleton, "get_spark_session")
    @patch.object(FeatureStore, "from_id")
    def test_select_subset_two_columns_and_join_query(
        self, spark_session, get_spark_session, mock_feature_store
    ):
        """Tests with select and join query."""
        dsc_query = Query(
            left_feature_group=self.mock_dsc_feature_group_1,
            left_features=["cc_num", "provider"],
            feature_store_id=self.feature_store_id,
            entity_id=self.entity_id,
        ).join(
            Query(
                left_feature_group=self.mock_dsc_feature_group_2,
                left_features=["cc_num", "provider"],
                feature_store_id=self.feature_store_id,
                entity_id=self.entity_id,
            ),
            on=["cc_num"],
        )
        print(dsc_query.to_string())
        assert (
            dsc_query.to_string()
            == "SELECT fg_1.cc_num cc_num, fg_1.provider provider FROM "
            "`ocid1.entity.oc1.iad.xxx`.feature_group1 fg_1 INNER JOIN "
            "`ocid1.entity.oc1.iad.xxx`.feature_group2 fg_0 ON fg_1.cc_num = fg_0.cc_num"
        )

    @patch.object(SparkSessionSingleton, "__init__", return_value=None)
    @patch.object(SparkSessionSingleton, "get_spark_session")
    @patch.object(FeatureStore, "from_id")
    def test_select_subset_and_multi_join_query(
        self, spark_session, get_spark_session, mock_feature_store
    ):
        """Tests with select and join query."""
        dsc_query = (
            Query(
                left_feature_group=self.mock_dsc_feature_group_1,
                left_features=["cc_num", "provider"],
                feature_store_id=self.feature_store_id,
                entity_id=self.entity_id,
            )
            .join(
                Query(
                    left_feature_group=self.mock_dsc_feature_group_2,
                    left_features=["cc_num", "provider"],
                    feature_store_id=self.feature_store_id,
                    entity_id=self.entity_id,
                ),
                on=["cc_num"],
            )
            .join(
                Query(
                    left_feature_group=self.mock_dsc_feature_group_3,
                    left_features=["X", "Y"],
                    feature_store_id=self.feature_store_id,
                    entity_id=self.entity_id,
                ),
                left_on=["cc_num"],
                right_on=["X"],
            )
        )
        assert dsc_query.to_string() == (
            "SELECT fg_2.cc_num cc_num, fg_2.provider provider, fg_1.X X, fg_1.Y Y "
            "FROM `ocid1.entity.oc1.iad.xxx`.feature_group1 fg_2 "
            "INNER JOIN `ocid1.entity.oc1.iad.xxx`.feature_group2 fg_0 "
            "ON fg_2.cc_num = fg_0.cc_num "
            "INNER JOIN `ocid1.entity.oc1.iad.xxx`.feature_group3 fg_1 "
            "ON fg_0.cc_num = fg_1.X"
        )

    @patch.object(SparkSessionSingleton, "get_spark_session")
    @patch.object(FeatureStore, "from_id")
    def test_validation_of_features(self, spark_session, mock_feature_store):
        """Tests with select and join query."""
        dsc_query = Query(
            left_feature_group=self.mock_dsc_feature_group_1,
            left_features=["cc_num", "provider"],
            feature_store_id=self.feature_store_id,
            entity_id=self.entity_id,
        ).join(
            Query(
                left_feature_group=self.mock_dsc_feature_group_2,
                left_features=["NOT_PRESENT", "NOT_PRESENT"],
                feature_store_id=self.feature_store_id,
                entity_id=self.entity_id,
            ),
            on=["cc_num"],
        )
        # TODO(fixme): Do not form the query in such cases
        assert dsc_query.to_string() == (
            "SELECT fg_1.cc_num cc_num, fg_1.provider provider, fg_0.NOT_PRESENT "
            "NOT_PRESENT FROM `ocid1.entity.oc1.iad.xxx`.feature_group1 fg_1 INNER JOIN "
            "`ocid1.entity.oc1.iad.xxx`.feature_group2 fg_0 ON fg_1.cc_num = fg_0.cc_num"
        )
