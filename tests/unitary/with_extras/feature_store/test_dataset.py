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

from ads.feature_store.common.feature_store_singleton import FeatureStoreSingleton
from ads.feature_store.dataset import Dataset
from ads.feature_store.dataset_job import DatasetJob
from ads.feature_store.execution_strategy.engine.spark_engine import SparkEngine

from ads.feature_store.service.oci_dataset import OCIDataset
from ads.feature_store.service.oci_feature_store import OCIFeatureStore
from ads.feature_store.feature_store import FeatureStore
from tests.unitary.with_extras.feature_store.test_feature_store import (
    FEATURE_STORE_PAYLOAD,
)

DATASET_OCID = "ocid1.dataset.oc1.iad.xxx"

DATASET_PAYLOAD = {
    "name": "dataset_name",
    "compartmentId": "compartmentId",
    "entityId": "ocid1.entity.oc1.iad.xxx",
    "description": "dataset description",
    "query": "SELECT feature_gr_1.name FROM feature_gr_1",
    "featureStoreId": "ocid1.featurestore.oc1.iad.xxx",
}

DATASET_JOB_PAYLOAD = {
    "jobConfigurationDetails": {"jobConfigurationType": "SPARK_BATCH_AUTOMATIC"},
    "compartmentId": "compartmentId",
    "featureGroupId": "ocid1.feature_group.oc1.iad.xxx",
    "ingestionMode": "OVERWRITE",
}

DATASET_JOB_RESPONSE_PAYLOAD = {
    "compartmentId": "ocid1.compartment.oc1.iad.xxx",
    "datasetId": "861AA4E9C8E811A79D74C464A01CDF42",
    "id": "d40265b7-d66e-49a3-ae26-699012e0df5d",
    "ingestionMode": "OVERWRITE",
    "lifecycleState": "SUCCEEDED",
}


@pytest.fixture
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


class TestDataset:
    test_config = {
        "tenancy": "ocid1.tenancy.oc1..xxx",  # must be a real-like ocid
        "user": "ocid1.user.oc1..xxx",  # must be a real-like ocid
        "fingerprint": "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
        "key_file": "<path>/<to>/<key_file>",
        "region": "test_region",
    }

    DEFAULT_PROPERTIES_PAYLOAD = {
        "compartmentId": DATASET_PAYLOAD["compartmentId"],
        "name": DATASET_PAYLOAD["name"],
    }

    def setup_class(cls):
        cls.mock_date = datetime.datetime(3022, 7, 1)

    def setup_method(self):
        self.payload = deepcopy(DATASET_PAYLOAD)
        self.mock_dsc_dataset = Dataset(**self.payload)
        self.payload_dataset_job = deepcopy(DATASET_JOB_PAYLOAD)
        self.mock_dsc_dataset_job = DatasetJob(**self.payload_dataset_job)

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
        Dataset,
        "_load_default_properties",
        return_value=DEFAULT_PROPERTIES_PAYLOAD,
    )
    def test__init__default_properties(self, mock_load_default_properties):
        dsc_dataset = Dataset()
        assert dsc_dataset.to_dict()["spec"] == self.DEFAULT_PROPERTIES_PAYLOAD

    @patch.object(Dataset, "_load_default_properties", return_value={})
    def test__init__(self, mock_load_default_properties):
        dsc_dataset = Dataset(**self.payload)
        assert self.prepare_dict(dsc_dataset.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(Dataset, "_load_default_properties", return_value={})
    def test_with_methods_1(self, mock_load_default_properties):
        """Tests all with methods."""
        dsc_dataset = (
            Dataset()
            .with_description("dataset description")
            .with_compartment_id(self.payload["compartmentId"])
            .with_name(self.payload["name"])
            .with_entity_id(self.payload["entityId"])
            .with_feature_store_id(self.payload["featureStoreId"])
            .with_query("SELECT feature_gr_1.name FROM feature_gr_1")
        )
        assert self.prepare_dict(dsc_dataset.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    def test_with_methods_2(self):
        """Tests all with methods."""
        dsc_dataset = (
            Dataset()
            .with_description("dataset description")
            .with_compartment_id(self.payload["compartmentId"])
            .with_name(self.payload["name"])
            .with_entity_id(self.payload["entityId"])
            .with_feature_store_id(self.payload["featureStoreId"])
            .with_query("SELECT feature_gr_1.name FROM feature_gr_1")
        )
        assert self.prepare_dict(dsc_dataset.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(OCIDataset, "update")
    @patch.object(OCIDataset, "delete")
    @patch.object(FeatureStoreSingleton, "__init__", return_value=None)
    @patch.object(FeatureStoreSingleton, "get_spark_session")
    def test_delete(self, spark, get_spark_session, mock_delete, mock_update):
        """Tests deleting dataset from dataset."""
        with patch.object(DatasetJob, "create"):
            with patch.object(FeatureStore, "from_id"):
                with patch.object(DatasetJob, "_mark_job_complete"):
                    self.mock_dsc_dataset.with_id(DATASET_OCID)
                    self.mock_dsc_dataset.delete()

    @patch.object(Dataset, "_update_from_oci_dataset_model")
    @patch.object(OCIDataset, "list_resource")
    def test_list(self, mock_list_resource, mock__update_from_oci_dataset_model):
        """Tests listing dataset in a given compartment."""
        mock_list_resource.return_value = [OCIDataset(**DATASET_PAYLOAD)]
        mock__update_from_oci_dataset_model.return_value = Dataset(**self.payload)
        result = Dataset.list(
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

    @patch.object(Dataset, "_update_from_oci_dataset_model")
    @patch.object(OCIDataset, "from_id")
    def test_from_id(self, mock_oci_from_id, mock__update_from_oci_dataset_model):
        """Tests getting an existing model by OCID."""
        mock_oci_model = OCIDataset(**DATASET_PAYLOAD)
        mock_oci_from_id.return_value = mock_oci_model
        mock__update_from_oci_dataset_model.return_value = Dataset(**self.payload)
        result = Dataset.from_id(DATASET_OCID)

        mock_oci_from_id.assert_called_with(DATASET_OCID)
        mock__update_from_oci_dataset_model.assert_called_with(mock_oci_model)
        assert self.prepare_dict(result.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(OCIDataset, "create")
    def test_create_success(
        self,
        mock_oci_dsc_model_create,
    ):
        """Tests creating datascience dataset."""
        oci_dsc_model = OCIDataset(**DATASET_PAYLOAD)
        mock_oci_dsc_model_create.return_value = oci_dsc_model

        # to check rundom display name
        self.mock_dsc_dataset.with_name("")
        result = self.mock_dsc_dataset.create()
        mock_oci_dsc_model_create.assert_called()

    @patch.object(Dataset, "_load_default_properties", return_value={})
    def test_create_fail(self, mock__load_default_properties):
        """Tests creating datascience dataset."""
        dsc_dataset = Dataset()
        with pytest.raises(ValueError, match="Specify compartment OCID."):
            dsc_dataset.create()

    def test_to_dict(self):
        """Tests serializing dataset to a dictionary."""
        test_dict = self.mock_dsc_dataset.to_dict()
        assert self.prepare_dict(test_dict["spec"]) == self.prepare_dict(self.payload)
        assert test_dict["kind"] == self.mock_dsc_dataset.kind
        assert test_dict["type"] == self.mock_dsc_dataset.type

    def test_from_dict(self):
        """Tests loading dataset instance from a dictionary of configurations."""
        assert self.prepare_dict(
            self.mock_dsc_dataset.to_dict()["spec"]
        ) == self.prepare_dict(
            Dataset.from_dict({"spec": self.payload}).to_dict()["spec"]
        )

    @patch("ads.common.utils.get_random_name_for_resource", return_value="test_name")
    def test__random_display_name(self, mock_get_random_name_for_resource):
        """Tests generating a random display name."""
        expected_result = f"{self.mock_dsc_dataset._PREFIX}-test_name"
        assert self.mock_dsc_dataset._random_display_name() == expected_result

    @patch("oci.config.from_file", return_value=test_config)
    @patch("oci.signer.load_private_key_from_file")
    def test__to_oci_fs_entity(self, mock_load_key_file, mock_config_from_file):
        """Tests creating an `OCIDataset` instance from the  `Dataset`."""
        with patch.object(OCIDataset, "sync"):
            test_oci_dsc_dataset = OCIDataset(**DATASET_PAYLOAD)
            test_oci_dsc_dataset.id = None
            test_oci_dsc_dataset.lifecycle_state = None
            test_oci_dsc_dataset.created_by = None
            test_oci_dsc_dataset.time_created = None

            assert self.prepare_dict(
                test_oci_dsc_dataset.to_dict()
            ) == self.prepare_dict(self.mock_dsc_dataset._to_oci_dataset().to_dict())

            test_oci_dsc_dataset.name = "new_name"
            assert self.prepare_dict(
                test_oci_dsc_dataset.to_dict()
            ) == self.prepare_dict(
                self.mock_dsc_dataset._to_oci_dataset(name="new_name").to_dict()
            )

    @patch.object(OCIDataset, "update")
    @patch.object(FeatureStoreSingleton, "__init__", return_value=None)
    @patch.object(FeatureStoreSingleton, "get_spark_session")
    def test_materialise(self, spark, get_spark_session, mock_update):
        with patch.object(DatasetJob, "create") as mock_dataset_job:
            with patch.object(FeatureStore, "from_id"):
                with patch.object(DatasetJob, "_mark_job_complete"):
                    mock_dataset_job.return_value = self.mock_dsc_dataset_job
                    self.mock_dsc_dataset.with_id(DATASET_OCID)
                    self.mock_dsc_dataset.materialise()

    @patch.object(FeatureStoreSingleton, "__init__", return_value=None)
    @patch.object(FeatureStoreSingleton, "get_spark_session")
    @patch.object(OCIFeatureStore, "from_id")
    def test_preview(self, feature_store, spark, get_spark_session):
        with patch.object(SparkEngine, "sql") as mock_execution_strategy:
            feature_store.return_value = OCIFeatureStore(**FEATURE_STORE_PAYLOAD)
            mock_execution_strategy.return_value = None
            self.mock_dsc_dataset.preview()
            mock_execution_strategy.assert_called_once()

    @patch.object(FeatureStoreSingleton, "__init__", return_value=None)
    @patch.object(FeatureStoreSingleton, "get_spark_session")
    @patch.object(OCIFeatureStore, "from_id")
    def test_profile(self, spark, get_spark_session, feature_store):
        with patch.object(SparkEngine, "sql") as mock_execution_strategy:
            feature_store.return_value = OCIFeatureStore(**FEATURE_STORE_PAYLOAD)
            mock_execution_strategy.return_value = None
            self.mock_dsc_dataset.profile()
            mock_execution_strategy.assert_called_once()

    @patch.object(FeatureStoreSingleton, "__init__", return_value=None)
    @patch.object(FeatureStoreSingleton, "get_spark_session")
    @patch.object(OCIFeatureStore, "from_id")
    def test_history(self, spark, get_spark_session, feature_store):
        with patch.object(SparkEngine, "sql") as mock_execution_strategy:
            feature_store.return_value = OCIFeatureStore(**FEATURE_STORE_PAYLOAD)
            mock_execution_strategy.return_value = None
            self.mock_dsc_dataset.history()
            mock_execution_strategy.assert_called_once()

    @patch.object(OCIDataset, "update")
    @patch.object(FeatureStoreSingleton, "__init__", return_value=None)
    @patch.object(FeatureStoreSingleton, "get_spark_session")
    @patch.object(OCIFeatureStore, "from_id")
    def test_restore(self, spark, get_spark_session, feature_store, mock_update):
        with patch.object(SparkEngine, "sql") as mock_execution_strategy:
            mock_execution_strategy.return_value = None
            self.mock_dsc_dataset.with_id(DATASET_OCID)
            self.mock_dsc_dataset.restore(1)
            mock_execution_strategy.assert_called_once()

    def test_get_last_job(self):
        """Tests getting most recent dataset job for a dataset."""
        with patch.object(DatasetJob, "list") as mock_dataset_job:
            self.mock_dsc_dataset.with_id(DATASET_OCID)
            mock_dataset_job.return_value = [
                DatasetJob.from_dict({"spec": DATASET_JOB_RESPONSE_PAYLOAD})
            ]
            ds_job = self.mock_dsc_dataset.get_last_job()
            assert ds_job is not None
