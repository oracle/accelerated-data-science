from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,ShortType, IntegerType, LongType, FloatType, DoubleType, \
    BooleanType, StringType, StructField,ByteType,BinaryType,DecimalType
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from decimal import Decimal

from ads.feature_store.common.enums import FeatureType
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.input_feature_detail import FeatureDetail
from tests.integration.feature_store.test_base import FeatureStoreTestCase
from ads.feature_store.common.spark_session_singleton import SparkSessionSingleton


class TestDataTypeSparkBasic(FeatureStoreTestCase):

    schema = StructType([
        StructField("short_col", ShortType(), True),
        StructField("int_col", IntegerType(), True),
        StructField("long_col", LongType(), True),
        StructField("float_col", FloatType(), True),
        StructField("double_col", DoubleType(), True),
        StructField("bool_col", BooleanType(), True),
        StructField("string_col", StringType(), True),
        StructField("byte_col", ByteType(), True),
        StructField("binary_col", BinaryType(), True),
        StructField("decimal_col", DecimalType(10, 2), True)
    ])

    input_feature_details_spark_basic = [
        FeatureDetail("short_col").with_feature_type(FeatureType.SHORT).with_order_number(1),
        FeatureDetail("int_col").with_feature_type(FeatureType.INTEGER).with_order_number(2),
        FeatureDetail("long_col").with_feature_type(FeatureType.LONG).with_order_number(3),
        FeatureDetail("float_col").with_feature_type(FeatureType.FLOAT).with_order_number(4),
        FeatureDetail("double_col").with_feature_type(FeatureType.DOUBLE).with_order_number(5),
        FeatureDetail("bool_col").with_feature_type(FeatureType.BOOLEAN).with_order_number(5),
        FeatureDetail("string_col").with_feature_type(FeatureType.STRING).with_order_number(6),
        FeatureDetail("byte_col").with_feature_type(FeatureType.BYTE).with_order_number(7),
        FeatureDetail("binary_col").with_feature_type(FeatureType.BINARY).with_order_number(8),
        FeatureDetail("decimal_col").with_feature_type(FeatureType.DECIMAL).with_order_number(9),
    ]

    # Add data to the DataFrame
    data = [
        (1, 100, 1000, 1.1, 1.11, True, "Hello", 10, bytearray(b'\x01\x02\x03'), Decimal('12.34')),
        (2, 200, 2000, 2.2, 2.22, False, "World", 20, bytearray(b'\x04\x05\x06'), Decimal(56.78))
    ]

    spark = SparkSessionSingleton(FeatureStoreTestCase.METASTORE_ID).get_spark_session()
    basic_df = spark.createDataFrame(data,schema)

    def define_feature_group_resource_with_spark_basic_infer_schema(
            self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_spark_basic = (
            FeatureGroup()
            .with_description("feature group resource spark basic types")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_spark_basic"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_schema_details_from_dataframe(self.basic_df)
        )
        return feature_group_spark_basic

    def define_feature_group_resource_with_spark_basic_with_schema(
            self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_spark_basic_schema = (
            FeatureGroup()
            .with_description("feature group resource spark basic types")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_spark_basic_schema"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details_spark_basic)
            .with_statistics_config(False)
        )
        return feature_group_spark_basic_schema

    def test_feature_group_spark_datatypes_infer_schema(self):
        """Test supported spark data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = self.define_feature_group_resource_with_spark_basic_infer_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.basic_df)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_spark_datatypes_with_schema(self):
        """Test supported spark data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = self.define_feature_group_resource_with_spark_basic_with_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.basic_df)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)