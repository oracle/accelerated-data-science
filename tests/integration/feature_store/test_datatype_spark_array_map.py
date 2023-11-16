from pyspark.sql.types import (
    StructType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    BooleanType,
    StringType,
    StructField,
    ByteType,
    BinaryType,
    DecimalType,
    DateType,
    TimestampType,
    ArrayType,
    MapType,
)
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from decimal import Decimal
from datetime import datetime

from ads.feature_store.common.enums import FeatureType
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.input_feature_detail import FeatureDetail
from tests.integration.feature_store.test_base import FeatureStoreTestCase
from ads.feature_store.common.spark_session_singleton import SparkSessionSingleton


# Define the schema for the DataFrame
class TestDataTypeSparkArrayMap(FeatureStoreTestCase):
    schema_array = StructType(
        [
            StructField("string_array_col", ArrayType(StringType()), True),
            StructField("int_array_col", ArrayType(IntegerType()), True),
            StructField("long_array_col", ArrayType(LongType()), True),
            StructField("float_array_col", ArrayType(FloatType()), True),
            StructField("double_array_col", ArrayType(DoubleType()), True),
            StructField("binary_array_col", ArrayType(BinaryType()), True),
            StructField("date_array_col", ArrayType(DateType()), True),
            StructField("timestamp_array_col", ArrayType(TimestampType()), True),
            StructField("byte_array_col", ArrayType(ByteType()), True),
            StructField("short_array_col", ArrayType(ShortType()), True),
            StructField("decimal_array_col", ArrayType(DecimalType(10, 2)), True),
            StructField("boolean_array_col", ArrayType(BooleanType()), True),
        ]
    )

    input_feature_details_spark_array = [
        FeatureDetail("string_array_col")
        .with_feature_type(FeatureType.STRING_ARRAY)
        .with_order_number(1),
        FeatureDetail("int_array_col")
        .with_feature_type(FeatureType.INTEGER_ARRAY)
        .with_order_number(2),
        FeatureDetail("long_array_col")
        .with_feature_type(FeatureType.LONG_ARRAY)
        .with_order_number(3),
        FeatureDetail("float_array_col")
        .with_feature_type(FeatureType.FLOAT_ARRAY)
        .with_order_number(4),
        FeatureDetail("double_array_col")
        .with_feature_type(FeatureType.DOUBLE_ARRAY)
        .with_order_number(5),
        FeatureDetail("binary_array_col")
        .with_feature_type(FeatureType.BINARY_ARRAY)
        .with_order_number(6),
        FeatureDetail("date_array_col")
        .with_feature_type(FeatureType.BYTE_ARRAY)
        .with_order_number(7),
        FeatureDetail("timestamp_array_col")
        .with_feature_type(FeatureType.TIMESTAMP_ARRAY)
        .with_order_number(8),
        FeatureDetail("byte_array_col")
        .with_feature_type(FeatureType.BYTE_ARRAY)
        .with_order_number(9),
        FeatureDetail("short_array_col")
        .with_feature_type(FeatureType.INTEGER_ARRAY)
        .with_order_number(10),
        FeatureDetail("decimal_array_col")
        .with_feature_type(FeatureType.COMPLEX)
        .with_order_number(11),
        FeatureDetail("boolean_array_col")
        .with_feature_type(FeatureType.COMPLEX)
        .with_order_number(12),
    ]

    schema_map = StructType(
        [
            StructField(
                "string_string_map_col", MapType(StringType(), StringType()), True
            ),
            StructField(
                "string_int_map_col", MapType(StringType(), IntegerType()), True
            ),
            StructField(
                "string_short_map_col", MapType(StringType(), ShortType()), True
            ),
            StructField("string_long_map_col", MapType(StringType(), LongType()), True),
            StructField(
                "string_float_map_col", MapType(StringType(), FloatType()), True
            ),
            StructField(
                "string_double_map_col", MapType(StringType(), DoubleType()), True
            ),
            StructField(
                "string_timestamp_map_col", MapType(StringType(), TimestampType()), True
            ),
            StructField("string_date_map_col", MapType(StringType(), DateType()), True),
            StructField(
                "string_binary_map_col", MapType(StringType(), BinaryType()), True
            ),
            StructField("string_byte_map_col", MapType(StringType(), ByteType()), True),
            StructField(
                "string_decimal_map_col",
                MapType(StringType(), DecimalType(10, 2)),
                True,
            ),
            StructField(
                "string_boolean_map_col", MapType(StringType(), BooleanType()), True
            ),
        ]
    )

    input_feature_details_spark_map = [
        FeatureDetail("string_string_map_col")
        .with_feature_type(FeatureType.STRING_STRING_MAP)
        .with_order_number(1),
        FeatureDetail("string_int_map_col")
        .with_feature_type(FeatureType.STRING_INTEGER_MAP)
        .with_order_number(2),
        FeatureDetail("string_short_map_col")
        .with_feature_type(FeatureType.STRING_SHORT_MAP)
        .with_order_number(3),
        FeatureDetail("string_long_map_col")
        .with_feature_type(FeatureType.STRING_LONG_MAP)
        .with_order_number(4),
        FeatureDetail("string_float_map_col")
        .with_feature_type(FeatureType.STRING_FLOAT_MAP)
        .with_order_number(5),
        FeatureDetail("string_double_map_col")
        .with_feature_type(FeatureType.STRING_DOUBLE_MAP)
        .with_order_number(6),
        FeatureDetail("string_timestamp_map_col")
        .with_feature_type(FeatureType.STRING_TIMESTAMP_MAP)
        .with_order_number(7),
        FeatureDetail("string_date_map_col")
        .with_feature_type(FeatureType.STRING_DATE_MAP)
        .with_order_number(9),
        FeatureDetail("string_binary_map_col")
        .with_feature_type(FeatureType.STRING_BINARY_MAP)
        .with_order_number(8),
        FeatureDetail("string_byte_map_col")
        .with_feature_type(FeatureType.STRING_BYTE_MAP)
        .with_order_number(10),
        FeatureDetail("string_decimal_map_col")
        .with_feature_type(FeatureType.COMPLEX)
        .with_order_number(11),
        FeatureDetail("string_boolean_map_col")
        .with_feature_type(FeatureType.COMPLEX)
        .with_order_number(12),
    ]

    # Add data to the DataFrame
    data_array = [
        (
            ["apple", "banana", "cherry"],
            [112345, 22345, 32345],
            [1000, 2000, 3000],
            [1.1, 2.2, 3.3],
            [1.11, 2.22, 3.33],
            [bytearray(b"\x01\x02\x03"), bytearray(b"\x04\x05\x06")],
            [
                datetime.strptime("2023-06-01", "%Y-%m-%d").date(),
                datetime.strptime("2023-06-02", "%Y-%m-%d").date(),
                datetime.strptime("2023-06-03", "%Y-%m-%d").date(),
            ],
            [
                datetime.strptime("2023-06-01 10:00:00", "%Y-%m-%d %H:%M:%S"),
                datetime.strptime("2023-06-02 10:00:00", "%Y-%m-%d %H:%M:%S"),
                datetime.strptime("2023-06-03 10:00:00", "%Y-%m-%d %H:%M:%S"),
            ],
            [10, 20, 30],
            [100, 200, 300],
            [Decimal("12.34"), Decimal("56.78")],
            [True, False, True],
        ),
        (
            ["orange", "kiwi", "melon"],
            [4, 5, 6],
            [4000, 5000, 6000],
            [4.4, 5.5, 6.6],
            [4.44, 5.55, 6.66],
            [bytearray(b"\x07\x08\x09"), bytearray(b"\x0A\x0B\x0C")],
            [
                datetime.strptime("2023-06-04", "%Y-%m-%d").date(),
                datetime.strptime("2023-06-05", "%Y-%m-%d").date(),
                datetime.strptime("2023-06-06", "%Y-%m-%d").date(),
            ],
            [
                datetime.strptime("2023-06-04 10:00:00", "%Y-%m-%d %H:%M:%S"),
                datetime.strptime("2023-06-05 10:00:00", "%Y-%m-%d %H:%M:%S"),
                datetime.strptime("2023-06-06 10:00:00", "%Y-%m-%d %H:%M:%S"),
            ],
            [40, 50, 60],
            [400, 500, 600],
            [Decimal("78.90"), Decimal("91.23")],
            [True, False, True],
        ),
    ]
    spark = SparkSessionSingleton(FeatureStoreTestCase.METASTORE_ID).get_spark_session()
    array_df = spark.createDataFrame(data_array, schema_array)

    data_map = [
        (
            {"key1": "value1", "key2": "value2"},
            {"key1": 1, "key2": 2},
            {"key1": 10, "key2": 20},
            {"key1": 100, "key2": 200},
            {"key1": 1.1, "key2": 2.2},
            {"key1": 1.11, "key2": 2.22},
            {
                "key1": datetime.strptime("2023-06-01 10:00:00", "%Y-%m-%d %H:%M:%S"),
                "key2": datetime.strptime("2023-06-02 10:00:00", "%Y-%m-%d %H:%M:%S"),
            },
            {
                "key1": datetime.strptime("2023-06-01", "%Y-%m-%d").date(),
                "key2": datetime.strptime("2023-06-02", "%Y-%m-%d").date(),
            },
            {"key1": bytearray(b"\x01\x02\x03"), "key2": bytearray(b"\x04\x05\x06")},
            {"key1": 10, "key2": 20},
            {"key3": Decimal("78.90"), "key4": Decimal("91.23")},
            {"key1": True, "key2": False},
        ),
        (
            {"key3": "value3", "key4": "value4"},
            {"key3": 3, "key4": 4},
            {"key3": 30, "key4": 40},
            {"key3": 300, "key4": 400},
            {"key3": 3.3, "key4": 4.4},
            {"key3": 3.33, "key4": 4.44},
            {
                "key3": datetime.strptime("2023-06-03 10:00:00", "%Y-%m-%d %H:%M:%S"),
                "key4": datetime.strptime("2023-06-04 10:00:00", "%Y-%m-%d %H:%M:%S"),
            },
            {
                "key3": datetime.strptime("2023-06-03", "%Y-%m-%d").date(),
                "key4": datetime.strptime("2023-06-04", "%Y-%m-%d").date(),
            },
            {"key3": bytearray(b"\x07\x08\x09"), "key4": bytearray(b"\x0A\x0B\x0C")},
            {"key3": 30, "key4": 40},
            {"key3": Decimal("78.90"), "key4": Decimal("91.23")},
            {"key1": True, "key2": False},
        ),
    ]
    map_df = spark.createDataFrame(data_map, schema_map)

    def define_feature_group_resource_with_spark_array_infer_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_spark_array = (
            FeatureGroup()
            .with_description("feature group resource spark array types")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_spark_array"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_schema_details_from_dataframe(self.array_df)
            .with_statistics_config(False)
        )
        return feature_group_spark_array

    def define_feature_group_resource_with_spark_array_with_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_spark_array_schema = (
            FeatureGroup()
            .with_description("feature group resource spark array types")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_spark_array_schema"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details_spark_array)
            .with_statistics_config(False)
        )
        return feature_group_spark_array_schema

    def define_feature_group_resource_with_spark_map_infer_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_spark_array = (
            FeatureGroup()
            .with_description("feature group resource spark map types")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_spark_map"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_schema_details_from_dataframe(self.map_df)
            .with_statistics_config(False)
        )
        return feature_group_spark_array

    def define_feature_group_resource_with_spark_map_with_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_spark_map_schema = (
            FeatureGroup()
            .with_description("feature group resource spark map types")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_spark_map_schema"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details_spark_map)
            .with_statistics_config(False)
        )
        return feature_group_spark_map_schema

    def test_feature_group_spark_datatypes_array_infer_schema(self):
        """Test supported spark array types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = (
            self.define_feature_group_resource_with_spark_array_infer_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.array_df)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_spark_datatypes_array_with_schema(self):
        """Test supported spark array types types with schema"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = self.define_feature_group_resource_with_spark_array_with_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.array_df)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_spark_datatypes_map_infer_schema(self):
        """Test supported spark map types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = self.define_feature_group_resource_with_spark_map_infer_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.array_df)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_spark_datatypes_map_with_schema(self):
        """Test supported spark map types with schema"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = self.define_feature_group_resource_with_spark_map_with_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.map_df)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)
