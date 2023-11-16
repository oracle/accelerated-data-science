#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import hashlib
import json
from copy import deepcopy
from unittest import mock
from unittest.mock import patch

import pandas
import pytest
import ads.feature_store.common.utils.utility

import logging
import os
from collections import defaultdict
from typing import Callable, Dict, List, Union, TypeVar, Generic, Type, Optional

import fsspec
from oci.resource_search.models import (
    ResourceSummaryCollection,
    StructuredSearchDetails,
)

from ads.common.oci_client import OCIClientFactory

from ads.dataset.progress import TqdmProgressBar
from ads.feature_store.common.enums import FeatureType, TransformationMode

from ads.feature_store.entity import Entity
from ads.feature_store.feature_store import FeatureStore
from ads.feature_store.dataset import Dataset
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.feature_store_registrar import FeatureStoreRegistrar
from ads.feature_store.input_feature_detail import FeatureDetail
from ads.feature_store.transformation import Transformation
from ads.feature_store.service.oci_feature_store import OCIFeatureStore
from ads.feature_store.service.oci_entity import OCIEntity
from ads.feature_store.service.oci_transformation import OCITransformation
from ads.feature_store.service.oci_feature_group import OCIFeatureGroup
from ads.feature_store.service.oci_dataset import OCIDataset
import yaml

from ads.common import utils, auth

try:
    from yaml import CSafeLoader as Loader
except:
    from yaml import SafeLoader as Loader

TEST_COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaayzpqrqrtpn5ddo62rgdo5vhs236p7l2qex5skynqpmregtqiblea"
TEST_METASTORE_ID = "ocid1.datacatalogmetastore.oc1.iad.amaaaaaaqc2qulqav5pzijun724nglvsent3634hqrc2ybu5vfi3fu35tkyq"
TEST_FILE_PATH = "../../../data/feature_store_minimal.yaml"

TEST_PAYLOAD = {
    "apiVersion": "v1",
    "kind": "featureStore",
    "spec": {
        "displayName": "FeatureStore Resource Minimal",
        "compartmentId": "ocid1.compartment.oc1..aaaaaaaayzpqrqrtpn5ddo62rgdo5vhs236p7l2qex5skynqpmregtqiblea",
        "offlineConfig": {
            "metastoreId": "ocid1.datacatalogmetastore.oc1.iad.amaaaaaaqc2qulqav5pzijun724nglvsent3634hqrc2ybu5vfi3fu35tkyq"
        },
        "entity": [{"kind": "entity", "spec": {"name": "credit_card_entity_40"}}],
        "transformation": [
            {
                "kind": "transformation",
                "spec": {
                    "displayName": "transactions_df",
                    "transformationMode": "SPARK_SQL",
                    "sourceCode": ' def transactions_df(transactions_batch): sql_query = f""" SELECT tid, datetime, cc_num, category, amount, latitude FROM {transactions_batch} """ return sql_query ',
                },
            }
        ],
        "featureGroup": [
            {
                "kind": "featureGroup",
                "spec": {
                    "entity": [
                        {"kind": "entity", "spec": {"name": "credit_card_entity_40"}}
                    ],
                    "name": "profile_feature_group_name_40",
                    "primaryKeys": ["name"],
                    "inputFeatureDetails": [
                        {
                            "name": "name",
                            "featureType": "STRING",
                            "orderNumber": 1,
                            "cast": "STRING",
                        },
                        {
                            "name": "sex",
                            "featureType": "STRING",
                            "orderNumber": 2,
                            "cast": "STRING",
                        },
                        {
                            "name": "mail",
                            "featureType": "STRING",
                            "orderNumber": 3,
                            "cast": "STRING",
                        },
                    ],
                },
            }
        ],
    },
}


class TestFeatureStoreRegistrar:
    test_config = {
        "tenancy": "ocid1.tenancy.oc1..xxx",  # must be a real-like ocid
        "user": "ocid1.user.oc1..xxx",  # must be a real-like ocid
        "fingerprint": "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
        "key_file": "<path>/<to>/<key_file>",
        "region": "test_region",
    }

    def setup_class(cls):
        cls.mock_date = datetime.datetime(3022, 7, 1)

    def setup_method(self):
        pass

    @pytest.fixture(autouse=False, scope="module")
    def feature_store_model(self):
        return (
            FeatureStore()
            .with_compartment_id(TEST_COMPARTMENT_ID)
            .with_display_name("FeatureStore Test Resource")
            .with_offline_config(metastore_id=TEST_METASTORE_ID)
        )

    def test_create_failure(self):
        pass

    def test_from_dict_invalid(self):
        with pytest.raises(ValueError):
            FeatureStoreRegistrar.from_dict([])

    def test_from_dict_empty(self):
        with pytest.raises(ValueError):
            FeatureStoreRegistrar.from_dict({})

    def test_from_dict_valid(self):
        registrar = FeatureStoreRegistrar.from_dict(TEST_PAYLOAD)
        assert registrar is not None

    def test_generate_yaml(self):
        FeatureStoreRegistrar.generate_yaml()
        pwd = os.getcwd()
        file_path = os.path.join(pwd, "feature_store.yaml")
        assert os.path.isfile(file_path) == True
        os.remove(file_path)

    def test_from_yaml_with_uri(self):
        registrar = FeatureStoreRegistrar.from_yaml(uri=TEST_FILE_PATH)
        assert registrar is not None
        assert len(registrar.feature_group_entity_map) == 1
        assert len(registrar.feature_group_transformation_map) == 0
        assert len(registrar.dataset_entity_map) == 1

    def test_from_yaml_with_string(self):
        with open(TEST_FILE_PATH, "r") as yaml_file:
            yaml_string = yaml_file.read()
        registrar = FeatureStoreRegistrar.from_yaml(yaml_string=yaml_string)
        assert registrar is not None
        assert len(registrar.feature_group_entity_map) == 1
        assert len(registrar.feature_group_transformation_map) == 0
        assert len(registrar.dataset_entity_map) == 1

    def test_from_yaml_without_args_failure(self):
        with pytest.raises(ValueError):
            FeatureStoreRegistrar.from_yaml()

    def test__init__(self, feature_store_model):
        entity = (
            Entity()
            .with_compartment_id(TEST_COMPARTMENT_ID)
            .with_name("flight_entity_test")
            .with_feature_store_id(feature_store_model.id)
        )
        flight_input_feature_details = [
            FeatureDetail("YEAR", FeatureType.INTEGER, 1),
            FeatureDetail("MONTH", FeatureType.INTEGER, 2),
            FeatureDetail("DAY", FeatureType.INTEGER, 3),
            FeatureDetail("FLIGHT_NUMBER", FeatureType.INTEGER, 4),
        ]
        feature_group = (
            FeatureGroup()
            .with_compartment_id(TEST_COMPARTMENT_ID)
            .with_name("FeatureGroup")
            .with_entity_id(entity.id)
            .with_feature_store_id(feature_store_model.id)
            .with_primary_keys(["FLIGHT_NUMBER"])
            .with_input_feature_details(flight_input_feature_details)
        )
        registrar = FeatureStoreRegistrar(
            feature_store_model, [entity], [], [feature_group], []
        )
        assert registrar is not None
        assert registrar.feature_group_transformation_map is not None
        assert registrar.feature_group_entity_map is not None
        assert registrar.dataset_entity_map is not None

    @patch.object(OCIDataset, "create")
    @patch.object(OCIFeatureGroup, "create")
    @patch.object(OCITransformation, "create")
    @patch.object(OCIEntity, "create")
    @patch.object(OCIFeatureStore, "create")
    def test_create(
        self,
        mock_fs_create,
        mock_entity_create,
        mock_trans_create,
        mock_fg_create,
        mock_dataset_create,
    ):
        registrar = FeatureStoreRegistrar.from_yaml(uri=TEST_FILE_PATH)
        registrar.create()
        mock_fs_create.assert_called()
        mock_entity_create.assert_called()
        mock_trans_create.assert_called()
        mock_fg_create.assert_called()
        mock_dataset_create.assert_called()
