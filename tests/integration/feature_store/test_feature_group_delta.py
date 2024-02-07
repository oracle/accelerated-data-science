#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
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


class TestFeatureGroupDelta(FeatureStoreTestCase):
    """Contains integration tests for Feature Group Delta changes."""

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

    # @pytest.mark.skip(reason="will be redundant after others tests are added")
    def test_create_feature_group_entities(self, feature_store, entity, feature_group):
        """Tests creating feature store entities."""

        assert feature_store.oci_fs.id
        assert feature_store.oci_fs.lifecycle_state == "ACTIVE"
        assert feature_store.oci_fs.name == self.get_name("FeatureStore1")

        assert entity.oci_fs_entity.id
        assert entity.oci_fs_entity.lifecycle_state == "ACTIVE"

        assert feature_group.oci_feature_group.id
        assert feature_group.oci_feature_group.name == self.get_name("petals1")
        assert feature_group.oci_feature_group.lifecycle_state == "ACTIVE"

    def test_feature_group_delta_operations(self, feature_store, entity, feature_group):
        """Tests delta operations of feature group."""

        feature_group.materialise(self.data)
        feature_group.materialise(self.data2, ingestion_mode=IngestionMode.OVERWRITE)

        df = feature_group.select()
        assert df.read().count() == 14

        preview = feature_group.preview(row_count=5)
        assert preview.count() == 5

        restore = feature_group.restore(version_number=0)
        assert restore.count() == 1

        history = feature_group.history()
        assert history.count() == 3

        profile = feature_group.profile()
        assert profile.count() == 1

    def test_feature_group_materialise_append(self, feature_group):
        """Tests feature group append."""

        feature_group.materialise(self.data)

        df = feature_group.select()
        assert df.read().count() == 14

        feature_group.materialise(
            self.data2,
            ingestion_mode=IngestionMode.APPEND,
        )

        df = feature_group.select()
        assert df.read().count() == 28

    def test_feature_group_materialise_overwrite(self, feature_group):
        """Tests feature group overwrite."""

        feature_group.materialise(self.data)

        df = feature_group.select().read()
        assert df.count() == 14
        assert len(df.columns) == 6

        feature_group.materialise(self.data2, ingestion_mode=IngestionMode.OVERWRITE)

        df = feature_group.select().read()
        assert df.count() == 14
        assert len(df.columns) == 6

    def test_feature_group_materialise_upsert(self, feature_group):
        """Tests feature group upsert."""

        feature_group.materialise(self.data)

        df = feature_group.select().read()
        assert df.count() == 14
        assert len(df.columns) == 6

        feature_group.materialise(self.data2, ingestion_mode=IngestionMode.UPSERT)

        df = feature_group.select().read()
        assert df.count() == 14
        assert len(df.columns) == 6

    def test_feature_group_materialise_overwrite_schema(self, feature_group):
        """Tests feature group overwrite schema."""

        feature_group.materialise(self.data)

        df = feature_group.select().read()
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

        df = feature_group.select().read()
        assert df.count() == 14
        assert len(df.columns) == 5

    def test_feature_group_materialise_merge_schema(self, feature_group):
        """Tests feature group merge schema."""

        feature_group.materialise(self.data3)

        df = feature_group.select().read()
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

        df = feature_group.select().read()
        assert df.count() == 14
        assert len(df.columns) == 6
