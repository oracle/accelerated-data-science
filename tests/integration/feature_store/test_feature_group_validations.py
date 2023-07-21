#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.feature_group_expectation import ExpectationType

from tests.integration.feature_store.test_base import FeatureStoreTestCase


class TestFeatureGroupValidation(FeatureStoreTestCase):
    """Contains integration tests for Feature Group validations changes."""

    def define_feature_group_resource1(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_resource1 = (
            FeatureGroup()
            .with_description("feature group with expectation")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("petals4"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys(["flower"])
            .with_input_feature_details(self.INPUT_FEATURE_DETAILS)
            .with_statistics_config(False)
            .with_expectation_suite(
                expectation_suite=self.define_expectation_suite_single(),
                expectation_type=ExpectationType.STRICT,
            )
        )
        return feature_group_resource1

    def define_feature_group_resource2(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_resource2 = (
            FeatureGroup()
            .with_description("feature group with expectation")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("petals5"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys(["flower"])
            .with_input_feature_details(self.INPUT_FEATURE_DETAILS)
            .with_statistics_config(False)
            .with_expectation_suite(
                expectation_suite=self.define_expectation_suite_multiple(),
                expectation_type=ExpectationType.LENIENT,
            )
        )
        return feature_group_resource2

    def test_feature_group_validation_operations_single(self):
        """Tests validation operations of feature group."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource1(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()

        assert fg.oci_feature_group.id
        fg.materialise(self.data)

        df = fg.get_validation_output().to_pandas().T
        assert df is not None
        assert "success" in df.columns
        assert True in df["success"].values

        df = fg.get_validation_output().to_summary().T
        assert df is not None
        assert "success" in df.columns
        assert True in df["success"].values

        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_without_validation(self):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()

        assert fg.oci_feature_group.id
        fg.materialise(self.data)
        df = fg.get_validation_output().to_pandas()
        assert df is None

        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_validation_operations_multi(self):
        """Tests validation operations of feature group."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource2(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()

        assert fg.oci_feature_group.id
        fg.materialise(self.data)
        df = fg.get_validation_output().to_pandas().T
        assert df is not None
        assert "success" in df.columns
        assert True in df["success"].values
        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)
