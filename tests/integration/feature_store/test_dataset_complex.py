#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
import unittest

from ads.feature_store.dataset import Dataset

from ads.feature_store.entity import Entity

from ads.feature_store.feature_store import FeatureStore

from ads.feature_store.feature_group import FeatureGroup

from ads.feature_store.feature_option_details import FeatureOptionDetails
from tests.integration.feature_store.test_base import FeatureStoreTestCase


class TestDatasetComplex(FeatureStoreTestCase):
    """Contains integration tests for Dataset Delta changes."""

    @pytest.fixture()
    def feature_store(self) -> FeatureStore:
        feature_store = self.define_feature_store_resource().create()
        yield feature_store
        # self.clean_up_feature_store(feature_store)

    @pytest.fixture()
    def entity(self, feature_store: FeatureStore):
        entity = self.create_entity_resource(feature_store)
        yield entity
        # self.clean_up_entity(entity)

    @pytest.fixture()
    def feature_group(self, entity, feature_store) -> "FeatureGroup":
        feature_group = self.define_feature_group_resource(
            entity.oci_fs_entity.id, feature_store.oci_fs.id
        ).create()
        yield feature_group
        # self.clean_up_feature_group(feature_group)

    def test_manual_dataset(
        self,
        feature_store: FeatureStore,
        entity: Entity,
        feature_group: FeatureGroup,
    ):
        query = """
             (SELECT
                name, games, goals
                FROM tblMadrid WHERE name = 'ronaldo')
             UNION
             (SELECT
                name, games, goals
                FROM tblBarcelona WHERE name = 'messi')
            ORDER BY goals"""
        name = self.get_name("fireside_football_debate")
        dataset_resource = (
            Dataset()
            .with_description("dataset description")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(name)
            .with_entity_id(entity_id=entity.id)
            .with_feature_store_id(feature_store_id=feature_store.id)
            .with_query(query)
            .with_feature_groups([feature_group])
        ).create()
        assert len(dataset_resource.feature_groups) == 1
        assert dataset_resource.feature_groups[0].id == feature_group.id
        assert dataset_resource.get_spec(
            Dataset.CONST_FEATURE_GROUP
        ).is_manual_association
        return dataset_resource
