#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from ads.feature_store.common.utils.transformation_query_validator import TransformationQueryValidator
from ads.feature_store.transformation import Transformation
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.common.enums import TransformationMode
from tests.integration.feature_store.test_base import FeatureStoreTestCase
from ads.feature_store.common.spark_session_singleton import SparkSessionSingleton


class TestFeatureStoreTransformation(FeatureStoreTestCase):
    """Contains integration tests for Feature store  transformations."""

    valid_spark_queries = [
        "SELECT requisitionId, length(title) As title_word_count,"
        " CASE When length(title) > 0 Then 0 Else 1 End As empty_title,"
        " length(description) As description_word_count,"
        " length(designation) As designation_word_count FROM DATA_SOURCE_INPUT",
        "SELECT user_id, credit_score FROM DATA_SOURCE_INPUT",
        "SELECT  country, city, zipcode, state FROM DATA_SOURCE_INPUT WHERE state in ('PR', 'AZ', 'FL') order by state",
        "SELECT SalesPersonID, COUNT(SalesOrderID) AS TotalSales, SalesYear "
        "FROM DATA_SOURCE_INPUT GROUP BY SalesYear, SalesPersonID ORDER BY SalesPersonID, SalesYear"
    ]

    user_query = "SELECT sepal_length, sepal_width, petal_length, petal_width FROM DATA_SOURCE_INPUT"

    invalid_user_query = "SELECT sepal_length, sepal_width, petal_length, petal_width FROM DATA_SOURCE_INPUT1"

    def create_feature_group_resource(
            self, entity_id, feature_store_id, transformation_id
    ) -> FeatureGroup:
        feature_group_resource = FeatureGroup() \
            .with_description("feature group with statistics disabled") \
            .with_feature_store_id(feature_store_id) \
            .with_entity_id(entity_id) \
            .with_primary_keys([]) \
            .with_input_feature_details(self.INPUT_FEATURE_DETAILS) \
            .with_transformation_id(transformation_id) \
            .with_compartment_id(self.COMPARTMENT_ID) \
            .with_name(self.get_name("petals5"))\
            .create()
        return feature_group_resource

    def create_transformation_resource(
            self, feature_store_id, query, mode
    ) -> Transformation:
        transformation_resource = Transformation() \
            .with_description("feature group with statistics disabled") \
            .with_compartment_id(self.COMPARTMENT_ID) \
            .with_feature_store_id(feature_store_id) \
            .with_transformation_query_input(query) \
            .with_transformation_mode(mode) \
            .create()
        return transformation_resource

    def test_valid_queries(self):
        for spark_query in self.valid_spark_queries:
            TransformationQueryValidator.verify_sql_input(spark_query,
                                                          Transformation.CONST_DATA_SOURCE_TRANSFORMATION_INPUT)

    def test_valid_transformation(self):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id
        transformation = self.create_transformation_resource(fs.oci_fs.id, self.user_query, TransformationMode.SQL)
        assert transformation.id
        self.clean_up_transformation(transformation)
        self.clean_up_feature_store(fs)

    def test_invalid_transformation_mode_pandas_with_user_query(self):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id
        try:
            self.create_transformation_resource(fs.oci_fs.id, self.user_query, TransformationMode.PANDAS)
        except ValueError as e:
            assert e.__str__() == "Transformation query input is supported in SQL Mode only."
        self.clean_up_feature_store(fs)

    def test_invalid_transformation_query(self):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id
        try:
            self.create_transformation_resource(fs.oci_fs.id, self.invalid_user_query, TransformationMode.SQL)
        except ValueError as e:
            assert e.__str__() == f"Incorrect table template name, It should be " \
                                  f"{Transformation.CONST_DATA_SOURCE_TRANSFORMATION_INPUT}"

    def test_transformation_query_with_feature_group_job(self):
        """Tests transformation query execution."""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        transformation = self.create_transformation_resource(fs.oci_fs.id, self.user_query, TransformationMode.SQL)
        assert transformation.id

        fg = self.create_feature_group_resource(
            entity.oci_fs_entity.id, fs.oci_fs.id, transformation.id
        )
        assert fg.oci_feature_group.id

        # convert pandas to spark dataframe to run SPARK SQL transformation mode
        spark = SparkSessionSingleton().get_spark_session()
        spark_df = spark.createDataFrame(self.data)
        # get item count
        item_count = spark_df.count()
        # materialise to delta table
        fg.materialise(spark_df)
        # read dataframe
        df = fg.select().read()
        # assert dataframe
        assert df
        # assert count
        assert df.count() == item_count

        self.clean_up_feature_group(fg)
        self.clean_up_transformation(transformation)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)
