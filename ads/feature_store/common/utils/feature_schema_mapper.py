#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List

from ads.common.decorator.runtime_dependency import OptionalDependency

try:
    from pyspark.sql.types import *
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `pyspark` module was not found. Please run `pip install "
        f"{OptionalDependency.SPARK}`."
    )
except Exception as e:
    raise


def map_spark_type_to_feature_type(spark_type):
    """Returns the feature type corresponding to SparkType
    :param spark_type:
    :return:
    """
    spark_type_to_feature_type = {
        StringType(): "string",
        IntegerType(): "integer",
        FloatType(): "float",
        DoubleType(): "double",
        BooleanType(): "boolean",
        DateType(): "date",
        TimestampType(): "timestamp",
        DecimalType(): "decimal",
        BinaryType(): "binary",
        ArrayType(StringType()): "array",
        MapType(StringType(), StringType()): "map",
        StructType(): "struct",
        ByteType(): "byte",
        ShortType(): "short",
        LongType(): "long",
    }

    return spark_type_to_feature_type.get(spark_type).upper()


def map_pandas_type_to_feature_type(pandas_type):
    """Returns the feature type corresponding to pandas_type
    :param pandas_type:
    :return:
    """
    pandas_type_to_feature_type = {
        "object": "string",
        "int64": "integer",
        "float64": "float",
        "bool": "boolean",
    }

    return pandas_type_to_feature_type.get(pandas_type).upper()


def map_feature_type_to_spark_type(feature_type):
    """Returns the Spark Type for a particular feature type.
    :param feature_type:
    :return: Spark Type
    """
    spark_types = {
        "string": StringType(),
        "integer": IntegerType(),
        "float": FloatType(),
        "double": DoubleType(),
        "boolean": BooleanType(),
        "date": DateType(),
        "timestamp": TimestampType(),
        "decimal": DecimalType(),
        "binary": BinaryType(),
        "array": ArrayType(StringType()),
        "map": MapType(StringType(), StringType()),
        "struct": StructType(),
        "byte": ByteType(),
        "short": ShortType(),
        "long": LongType(),
    }

    return spark_types.get(feature_type.lower(), None)


def get_raw_data_source_schema(raw_feature_details: List[dict]):
    """Converts input feature details to Spark schema.

    Args:
      raw_feature_details(List[dict]): List of input feature details.

    Returns:
      StructType: Spark schema.
    """
    # Initialize the schema
    features_schema = StructType()

    for feature_details in raw_feature_details:
        # Get the name, feature_type and is_nullable
        feature_name = feature_details.get("name")
        feature_type = map_feature_type_to_spark_type(
            feature_details.get("featureType")
        )
        is_nullable = (
            True
            if feature_details.get("isNullable") is None
            else feature_details.get("isNullable")
        )

        features_schema.add(feature_name, feature_type, is_nullable)

    return features_schema
