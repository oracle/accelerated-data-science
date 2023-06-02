#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from ads.feature_store.dataset import Dataset
from tests.integration.feature_store.test_base import FeatureStoreTestCase


class TestDatasetStatistics(FeatureStoreTestCase):
    """Contains integration tests for Dataset Statistics computation."""

    def define_dataset_resource_with_stats_disabled(
        self, entity_id, feature_store_id, feature_group_name
    ) -> "Dataset":
        name = self.get_name("petals1")
        dataset_resource = (
            Dataset()
            .with_description("dataset with statistics disabled")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("petals_ds_stat_disabled"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_query(f"SELECT * FROM `{entity_id}`.{feature_group_name}")
            .with_statistics_config(False)
        )
        return dataset_resource

    def define_dataset_resource_with_default_config(
        self, entity_id, feature_store_id, feature_group_name
    ) -> "Dataset":
        name = self.get_name("petals1")
        dataset_resource = (
            Dataset()
            .with_description("dataset with default statistics configuration")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("petals_ds_default_stat"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_query(f"SELECT * FROM `{entity_id}`.{feature_group_name}")
        )
        return dataset_resource

    def test_dataset_statistics_with_default_config(self):
        """Tests statistics computation operation for dataset."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()
        assert fg.oci_feature_group.id

        fg.materialise(self.data)

        dataset = self.define_dataset_resource_with_default_config(
            entity.oci_fs_entity.id, fs.oci_fs.id, fg.oci_feature_group.name
        ).create()
        assert dataset.oci_dataset.id

        dataset.materialise()
        stat_obj = dataset.get_statistics()
        assert stat_obj is not None
        assert len(stat_obj.to_pandas().columns) == 6

        self.clean_up_dataset(dataset)
        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_dataset_statistics_disabled(self):
        """Tests statistics computation operation for dataset."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()
        assert fg.oci_feature_group.id

        fg.materialise(self.data)

        dataset = self.define_dataset_resource_with_stats_disabled(
            entity.oci_fs_entity.id, fs.oci_fs.id, fg.oci_feature_group.name
        ).create()
        assert dataset.oci_dataset.id

        dataset.materialise()
        stat_obj = dataset.get_statistics()
        assert stat_obj.content is None

        self.clean_up_dataset(dataset)
        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_dataset_statistics(self):
        """Tests statistics computation operation for dataset."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()
        assert fg.oci_feature_group.id

        fg.materialise(self.data)

        dataset = self.define_dataset_resource(
            entity.oci_fs_entity.id, fs.oci_fs.id, fg.oci_feature_group.name
        ).create()
        assert dataset.oci_dataset.id

        dataset.materialise()
        stat_obj = dataset.get_statistics()
        assert stat_obj.content is not None
        assert len(stat_obj.to_pandas().columns) == 2

        self.clean_up_dataset(dataset)
        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)
