#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from ads.feature_store.feature_group import FeatureGroup
from tests.integration.feature_store.test_base import FeatureStoreTestCase


class TestFeatureGroupStatistics(FeatureStoreTestCase):
    """Contains integration tests for Feature Group Statistics computation."""

    def define_feature_group_resource_with_stats_disabled(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_resource = (
            FeatureGroup()
            .with_description("feature group with statistics disabled")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("petals2"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.INPUT_FEATURE_DETAILS)
            .with_statistics_config(False)
        )
        return feature_group_resource

    def define_feature_group_resource_with_default_config(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_resource = (
            FeatureGroup()
            .with_description("feature group with default stats config")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("petals3"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.INPUT_FEATURE_DETAILS)
        )
        return feature_group_resource

    def test_feature_group_statistics_with_default_config(self):
        """Tests statistics computation operation for feature group."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource_with_default_config(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()
        assert fg.oci_feature_group.id

        fg.materialise(self.data)

        stat_obj = fg.get_statistics()
        assert stat_obj.content is not None
        assert len(stat_obj.to_pandas().columns) == 6

        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_statistics_disabled(self):
        """Tests statistics computation operation for feature group."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource_with_stats_disabled(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()
        assert fg.oci_feature_group.id

        fg.materialise(self.data)

        stat_obj = fg.get_statistics()
        assert stat_obj.content is None

        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_statistics(self):
        """Tests statistics computation operation for feature group."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()
        assert fg.oci_feature_group.id

        fg.materialise(self.data)

        stat_obj = fg.get_statistics()
        assert stat_obj.content is not None
        assert len(stat_obj.to_pandas().columns) == 2

        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)
