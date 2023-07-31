#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest

from ads.feature_store.dataset import Dataset
from ads.feature_store.feature_group_expectation import ExpectationType
from ads.feature_store.model_details import ModelDetails
from tests.integration.feature_store.test_base import FeatureStoreTestCase


class TestFeatureGroupValidation(FeatureStoreTestCase):
    """Contains integration tests for Feature Group validations changes."""

    def define_dataset_resource1(
        self, entity_id, feature_store_id, feature_group_name
    ) -> "Dataset":
        dataset_resource1 = (
            Dataset()
            .with_description("dataset description")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("petals_ds"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_query(f"SELECT * FROM `{entity_id}`.{feature_group_name}")
            .with_statistics_config(False)
            .with_expectation_suite(
                expectation_suite=self.define_expectation_suite_single(),
                expectation_type=ExpectationType.STRICT,
            )
        )
        return dataset_resource1

    def test_dataset_validation_operations(self):
        """Tests validation operations of feature group."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()

        assert fg.oci_feature_group.id
        fg.materialise(self.data)

        dataset = self.define_dataset_resource1(
            entity.oci_fs_entity.id, fs.oci_fs.id, fg.oci_feature_group.name
        ).create()
        assert dataset.oci_dataset.id

        dataset.materialise()
        df = dataset.get_validation_output().to_pandas().T
        assert df is not None
        assert "success" in df.columns
        assert True in df["success"].values

        df = dataset.get_validation_output().to_summary().T
        assert df is not None
        assert "success" in df.columns
        assert True in df["success"].values

        self.clean_up_dataset(dataset)
        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_dataset_model_details(self):
        """Tests validation operations of feature group."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()

        assert fg.oci_feature_group.id
        fg.materialise(self.data)

        dataset = self.define_dataset_resource1(
            entity.oci_fs_entity.id, fs.oci_fs.id, fg.oci_feature_group.name
        ).create()
        assert dataset.oci_dataset.id

        dataset.materialise()
        dataset.add_models(ModelDetails().with_items(["model_ocid_invalid"]))
        assert len(dataset.model_details.get("items")) == 0
        self.clean_up_dataset(dataset)
        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_dataset_without_validation(self):
        """Tests validation operations of feature group."""
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
        df = dataset.get_validation_output().to_pandas()
        assert df is None

        self.clean_up_dataset(dataset)
        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)
