from pyspark.sql.types import StructType, ShortType, IntegerType, LongType, FloatType, DoubleType, \
    BooleanType, StringType, StructField, ByteType, BinaryType, DecimalType
from tests.integration.feature_store.test_base import FeatureStoreTestCase
from ads.feature_store.input_feature_detail import FeatureDetail, FeatureType
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.feature_group_job import FeatureGroupJob
import pandas as pd
import numpy as np
import pytest
from ads.feature_store.common.spark_session_singleton import SparkSessionSingleton


class TestInputSchema(FeatureStoreTestCase):
    input_feature_details = [
        FeatureDetail("A").with_feature_type(FeatureType.STRING).with_order_number(1),
        FeatureDetail("B").with_feature_type(FeatureType.INTEGER).with_order_number(2)
    ]

    a = ["value1", "value2"]
    b = [25, 60]
    c = [30, 50]
    pandas_basic_df = pd.DataFrame(
        {
            "A": a,
            "B": b,
            "C": c
        }
    )

    schema = StructType(
        [StructField("string_col", StringType(), True),
         StructField("int_col", IntegerType(), True),
         StructField("long_col", LongType(), True)]
    )

    input_feature_details_spark = [
        FeatureDetail("string_col").with_feature_type(FeatureType.STRING).with_order_number(1),
        FeatureDetail("int_col").with_feature_type(FeatureType.INTEGER).with_order_number(2),
        FeatureDetail("C").with_feature_type(FeatureType.INTEGER).with_order_number(2),
        FeatureDetail("B").with_feature_type(FeatureType.INTEGER).with_order_number(2),
    ]

    data = [
        ("value1", 100, 1000),
        ("value2", 200, 2000)
    ]
    spark = SparkSessionSingleton(FeatureStoreTestCase.METASTORE_ID).get_spark_session()
    basic_df = spark.createDataFrame(data, schema)

    def define_feature_group_resource_with_pandas_schema(
            self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_pandas_array = (
            FeatureGroup()
            .with_description("feature group resource for pandas array types")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_pandas_array"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details)
        )
        return feature_group_pandas_array

    def define_feature_group_resource_with_spark_schema(
            self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_spark_schema = (
            FeatureGroup()
            .with_description("feature group resource for pandas array types")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_spark_schema"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details_spark)
        )
        return feature_group_spark_schema

    def test_feature_group_pandas_schema_mismatch(self):
        """Tests  pandas schema"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = self.define_feature_group_resource_with_pandas_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )
        feature_group.create()
        feature_group.materialise(self.pandas_basic_df)


        df = feature_group.select().read()
        assert len(df.columns) == 2

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_spark_schema_mismatch(self):
        """Tests  pandas date time data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = self.define_feature_group_resource_with_spark_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )
        feature_group.create()
        feature_group.materialise(self.basic_df)

        df = feature_group.select().read()
        assert len(df.columns) == 2

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)


