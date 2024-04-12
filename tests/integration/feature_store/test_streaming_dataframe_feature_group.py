#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import time

from delta import configure_spark_with_delta_pip
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

from ads.feature_store.common.enums import TransformationMode, ExpectationType
from ads.feature_store.statistics_config import StatisticsConfig
from tests.integration.feature_store.test_base import FeatureStoreTestCase

CHECKPOINT_DIR = "tests/integration/feature_store/test_data"


def get_streaming_df():
    spark_builder = (
        SparkSession.builder.appName("FeatureStore")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .enableHiveSupport()
    )

    spark = configure_spark_with_delta_pip(spark_builder).getOrCreate()

    # Define the schema for the streaming data frame
    credit_score_schema = (
        StructType()
        .add("user_id", "string")
        .add("date", "string")
        .add("credit_score", "string")
    )

    credit_score_streaming_df = (
        spark.readStream.option("sep", ",")
        .option("header", "true")
        .schema(credit_score_schema)
        .csv(f"{CHECKPOINT_DIR}/")
    )

    return credit_score_streaming_df


def credit_score_transformation(credit_score):
    import pyspark.sql.functions as F

    # Create a new Spark DataFrame that contains the transformed credit score.
    transformed_credit_score = credit_score.select(
        "user_id",
        "date",
        F.when(F.col("credit_score").cast("int") > 500, 1)
        .otherwise(0)
        .alias("credit_score"),
    )

    # Return the new Spark DataFrame.
    return transformed_credit_score


class TestFeatureGroupWithStreamingDataFrame(FeatureStoreTestCase):
    """Contains integration tests for Feature Group Kwargs supported transformation."""

    def create_transformation_resource_stream(self, feature_store) -> "Transformation":
        transformation = feature_store.create_transformation(
            source_code_func=credit_score_transformation,
            name="credit_score_transformation",
            transformation_mode=TransformationMode.SPARK,
        )
        return transformation

    def test_feature_group_materialization_with_streaming_data_frame(self):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        transformation = self.create_transformation_resource_stream(fs)
        streaming_df = get_streaming_df()

        stats_config = StatisticsConfig().with_is_enabled(False)
        fg = entity.create_feature_group(
            primary_keys=["User_id"],
            schema_details_dataframe=streaming_df,
            statistics_config=stats_config,
            name=self.get_name("streaming_fg_1"),
            transformation_id=transformation.id,
        )
        assert fg.oci_feature_group.id

        query = fg.materialise_stream(
            input_dataframe=streaming_df,
            checkpoint_dir=f"{CHECKPOINT_DIR}/checkpoint/{fg.name}",
        )

        assert query
        time.sleep(10)
        query.stop()

        assert fg.select().read().count() == 10

        self.clean_up_feature_group(fg)
        self.clean_up_transformation(transformation)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_materialization_with_streaming_data_frame_and_expectation(
        self,
    ):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        transformation = self.create_transformation_resource_stream(fs)
        streaming_df = get_streaming_df()

        stats_config = StatisticsConfig().with_is_enabled(False)
        # Initialize Expectation Suite
        expectation_suite_trans = ExpectationSuite(
            expectation_suite_name="feature_definition"
        )
        expectation_suite_trans.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "date"},
            )
        )
        expectation_suite_trans.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "date"},
            )
        )

        fg = entity.create_feature_group(
            primary_keys=["User_id"],
            schema_details_dataframe=streaming_df,
            statistics_config=stats_config,
            expectation_suite=expectation_suite_trans,
            expectation_type=ExpectationType.LENIENT,
            name=self.get_name("streaming_fg_2"),
            transformation_id=transformation.id,
        )
        assert fg.oci_feature_group.id

        query = fg.materialise_stream(
            input_dataframe=streaming_df,
            checkpoint_dir=f"{CHECKPOINT_DIR}/checkpoint/{fg.name}",
        )

        assert query
        time.sleep(10)
        query.stop()

        assert fg.select().read().count() == 10
        assert fg.get_validation_output().to_pandas() is None

        self.clean_up_feature_group(fg)
        self.clean_up_transformation(transformation)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_materialization_with_streaming_data_frame_and_stats(self):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        transformation = self.create_transformation_resource_stream(fs)
        streaming_df = get_streaming_df()

        fg = entity.create_feature_group(
            primary_keys=["User_id"],
            schema_details_dataframe=streaming_df,
            name=self.get_name("streaming_fg_3"),
            transformation_id=transformation.id,
        )
        assert fg.oci_feature_group.id

        query = fg.materialise_stream(
            input_dataframe=streaming_df,
            checkpoint_dir=f"{CHECKPOINT_DIR}/checkpoint/{fg.name}",
        )

        assert query
        time.sleep(10)
        query.stop()

        assert fg.select().read().count() == 10
        assert fg.get_statistics().to_pandas() is None

        self.clean_up_feature_group(fg)
        self.clean_up_transformation(transformation)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)
