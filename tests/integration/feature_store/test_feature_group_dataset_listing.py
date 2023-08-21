#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from ads.feature_store.feature_group_job import FeatureGroupJob

from ads.feature_store.dataset import Dataset
from ads.feature_store.feature_group import FeatureGroup
from tests.integration.feature_store.test_base import FeatureStoreTestCase


class TestFeatureGroupDatasetListing(FeatureStoreTestCase):
    """Contains integration tests for Feature Group and Dataset Listing."""

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

    def define_dataset_resource_with_stats_disabled(
        self, entity_id, feature_store_id, feature_group_name
    ) -> "Dataset":
        name = self.get_name("petals4")
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

    def test_feature_group_listing_without_limit(self):
        """Tests listing of feature group resources with user defined limit."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg1 = self.define_feature_group_resource_with_default_config(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()
        assert fg1.oci_feature_group.id
        fg1.materialise(self.data)
        fg1.materialise(self.data2)

        fg1_job_list = FeatureGroupJob.list(
            compartment_id=self.COMPARTMENT_ID, feature_group_id=fg1.id
        )
        assert fg1_job_list is not None
        assert len(fg1_job_list) == 2

        fg2 = self.define_feature_group_resource_with_stats_disabled(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()
        assert fg2.oci_feature_group.id
        fg2.materialise(self.data3)

        fg_list = FeatureGroup.list(
            compartment_id=self.COMPARTMENT_ID, feature_store_id=fs.id
        )
        assert fg_list is not None
        assert len(fg_list) == 2

        self.clean_up_feature_group(fg1)
        self.clean_up_feature_group(fg2)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_listing_with_limit(self):
        """Tests listing of feature group resources with user defined limit."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg1 = self.define_feature_group_resource_with_default_config(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()
        assert fg1.oci_feature_group.id
        fg1.materialise(self.data)
        fg1.materialise(self.data2)

        fg1_job_list = FeatureGroupJob.list(
            compartment_id=self.COMPARTMENT_ID,
            feature_group_id=fg1.id,
            sort_by="timeCreated",
            limit="1",
        )
        assert fg1_job_list is not None
        assert len(fg1_job_list) == 1

        fg2 = self.define_feature_group_resource_with_stats_disabled(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()
        assert fg2.oci_feature_group.id
        fg2.materialise(self.data3)

        fg_list = FeatureGroup.list(
            compartment_id=self.COMPARTMENT_ID,
            sort_by="timeCreated",
            limit="1",
        )
        assert fg_list is not None
        assert len(fg_list) == 1

        self.clean_up_feature_group(fg1)
        self.clean_up_feature_group(fg2)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_dataset_listing_without_limit(self):
        """Tests listing of dataset resources without any limit."""
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
        ds_list = Dataset.list(
            compartment_id=self.COMPARTMENT_ID, feature_store_id=fs.id
        )
        assert ds_list is not None
        assert len(ds_list) == 1

        self.clean_up_dataset(dataset)
        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)
