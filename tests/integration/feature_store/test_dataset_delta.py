#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
import unittest

from ads.feature_store.feature_option_details import FeatureOptionDetails
from tests.integration.feature_store.test_base import FeatureStoreTestCase

try:
    from ads.feature_store.common.enums import IngestionMode
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest(
        "OCI Feature Store is not available. Skipping the Feature Store tests."
    )


class TestDatasetDelta(FeatureStoreTestCase):
    """Contains integration tests for Dataset Delta changes."""

    @pytest.fixture()
    def feature_store(self):
        feature_store = self.define_feature_store_resource().create()
        yield feature_store
        self.clean_up_feature_store(feature_store)

    @pytest.fixture()
    def entity(self, feature_store):
        entity = self.create_entity_resource(feature_store)
        yield entity
        self.clean_up_entity(entity)

    @pytest.fixture()
    def feature_group(self, entity, feature_store):
        feature_group = self.define_feature_group_resource(
            entity.oci_fs_entity.id, feature_store.oci_fs.id
        ).create()
        yield feature_group
        self.clean_up_feature_group(feature_group)

    @pytest.fixture()
    def dataset(self, entity, feature_store, feature_group):
        dataset = self.define_dataset_resource(
            entity.oci_fs_entity.id,
            feature_store.oci_fs.id,
            feature_group.oci_feature_group.name,
        ).create()
        yield dataset
        self.clean_up_dataset(dataset)

    # @pytest.mark.skip(reason="will be redundant after others tests are added")
    def test_create_dataset_entities(
        self, feature_store, entity, feature_group, dataset
    ):
        """Tests creating feature store entities."""

        assert feature_store.oci_fs.id
        assert feature_store.oci_fs.lifecycle_state == "ACTIVE"
        assert feature_store.oci_fs.display_name == self.get_name("FeatureStore1")

        assert entity.oci_fs_entity.id
        assert entity.oci_fs_entity.lifecycle_state == "ACTIVE"

        assert feature_group.oci_feature_group.id
        assert feature_group.oci_feature_group.name == self.get_name("petals1")
        assert feature_group.oci_feature_group.lifecycle_state == "ACTIVE"

        assert dataset.oci_dataset.id
        assert dataset.oci_dataset.name == self.get_name("petals_ds")
        assert dataset.oci_dataset.lifecycle_state == "ACTIVE"

    def test_dataset_delta_operations(self, feature_group, dataset):
        """Tests delta operations of dataset."""

        feature_group.materialise(self.data)
        dataset.materialise()

        feature_group.materialise(self.data2, ingestion_mode=IngestionMode.OVERWRITE)
        dataset.materialise()

        preview = dataset.preview(row_count=5)
        assert preview.count() == 5

        restore = dataset.restore(version_number=0)
        assert restore.count() == 1

        history = dataset.history()
        assert history.count() == 3

        profile = dataset.profile()
        assert profile.count() == 1

    def test_dataset_materialise_append(self, feature_group, dataset):
        """Tests dataset append."""

        feature_group.materialise(self.data)
        dataset.materialise()

        df = dataset.preview(row_count=50)
        assert df.count() == 14

        feature_group.materialise(
            self.data2,
            ingestion_mode=IngestionMode.APPEND,
        )
        dataset.materialise(ingestion_mode=IngestionMode.APPEND)

        df = dataset.preview(row_count=50)
        assert df.count() == 42

    def test_dataset_materialise_overwrite(self, feature_group, dataset):
        """Tests dataset overwrite."""

        feature_group.materialise(self.data)
        dataset.materialise()

        df = dataset.preview(row_count=50)
        assert df.count() == 14
        assert len(df.columns) == 6

        feature_group.materialise(self.data2, ingestion_mode=IngestionMode.OVERWRITE)
        dataset.materialise(ingestion_mode=IngestionMode.OVERWRITE)

        df = dataset.preview(row_count=50)
        assert df.count() == 14
        assert len(df.columns) == 6

    def test_dataset_materialise_overwrite_schema(self, feature_group, dataset):
        """Tests dataset overwrite schema."""

        feature_group.materialise(self.data)
        dataset.materialise()

        df = dataset.preview(row_count=50)
        assert df.count() == 14
        assert len(df.columns) == 6

        options = FeatureOptionDetails().with_feature_option_write_config_details(
            overwrite_schema=True
        )
        feature_group.materialise(
            self.data3,
            feature_option_details=options,
            ingestion_mode=IngestionMode.OVERWRITE,
        )
        dataset.materialise(
            ingestion_mode=IngestionMode.OVERWRITE, feature_option_details=options
        )

        df = dataset.preview(row_count=50)
        assert df.count() == 14
        assert len(df.columns) == 5

    def test_dataset_materialise_merge_schema(self, feature_group, dataset):
        """Tests dataset merge schema."""

        feature_group.materialise(self.data3)
        dataset.materialise()

        df = dataset.preview(row_count=50)
        assert df.count() == 14
        assert len(df.columns) == 5

        options = FeatureOptionDetails().with_feature_option_write_config_details(
            merge_schema=True
        )
        feature_group.materialise(
            self.data,
            feature_option_details=options,
            ingestion_mode=IngestionMode.OVERWRITE,
        )
        dataset.materialise(
            ingestion_mode=IngestionMode.OVERWRITE, feature_option_details=options
        )

        df = dataset.preview(row_count=50)
        assert df.count() == 14
        assert len(df.columns) == 6
