#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import List

import numpy as np
import pandas as pd

from ads.common.decorator.runtime_dependency import OptionalDependency
from mlm_insights.constants import types

from ads.feature_store.common.enums import FeatureType

try:
    from pyspark.sql.types import *
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `pyspark` module was not found. Please run `pip install "
        f"{OptionalDependency.SPARK}`."
    )
except Exception as e:
    raise

logger = logging.getLogger(__name__)


def map_spark_type_to_feature_type(spark_type):
    """Returns the feature type corresponding to SparkType
    :param spark_type:
    :return:
    """
    spark_type_to_feature_type = {
        StringType(): FeatureType.STRING,
        IntegerType(): FeatureType.INTEGER,
        ShortType(): FeatureType.SHORT,
        LongType(): FeatureType.LONG,
        FloatType(): FeatureType.FLOAT,
        DoubleType(): FeatureType.DOUBLE,
        BooleanType(): FeatureType.BOOLEAN,
        DateType(): FeatureType.DATE,
        TimestampType(): FeatureType.TIMESTAMP,
        BinaryType(): FeatureType.BINARY,
        ByteType(): FeatureType.BYTE,
        ArrayType(StringType()): FeatureType.STRING_ARRAY,
        ArrayType(IntegerType()): FeatureType.INTEGER_ARRAY,
        ArrayType(LongType()): FeatureType.LONG_ARRAY,
        ArrayType(FloatType()): FeatureType.FLOAT_ARRAY,
        ArrayType(DoubleType()): FeatureType.DOUBLE_ARRAY,
        ArrayType(BinaryType()): FeatureType.BINARY_ARRAY,
        ArrayType(DateType()): FeatureType.DATE_ARRAY,
        ArrayType(TimestampType()): FeatureType.TIMESTAMP_ARRAY,
        ArrayType(ByteType()): FeatureType.BYTE_ARRAY,
        ArrayType(BooleanType()): FeatureType.BOOLEAN_ARRAY,
        ArrayType(ShortType()): FeatureType.SHORT_ARRAY,
        MapType(StringType(), StringType()): FeatureType.STRING_STRING_MAP,
        MapType(StringType(), IntegerType()): FeatureType.STRING_INTEGER_MAP,
        MapType(StringType(), ShortType()): FeatureType.STRING_SHORT_MAP,
        MapType(StringType(), LongType()): FeatureType.STRING_LONG_MAP,
        MapType(StringType(), FloatType()): FeatureType.STRING_FLOAT_MAP,
        MapType(StringType(), DoubleType()): FeatureType.STRING_DOUBLE_MAP,
        MapType(StringType(), TimestampType()): FeatureType.STRING_TIMESTAMP_MAP,
        MapType(StringType(), DateType()): FeatureType.STRING_DATE_MAP,
        MapType(StringType(), BinaryType()): FeatureType.STRING_BINARY_MAP,
        MapType(StringType(), ByteType()): FeatureType.STRING_BYTE_MAP,
        MapType(StringType(), BooleanType()): FeatureType.STRING_BOOLEAN_MAP,
    }
    if spark_type in spark_type_to_feature_type:
        return spark_type_to_feature_type.get(spark_type)
    else:
        return FeatureType.COMPLEX


def map_pandas_type_to_feature_type(feature_name, values):
    pandas_type = str(values.dtype)
    inferred_dtype = FeatureType.UNKNOWN
    if pandas_type is "object":
        for row in values:
            if isinstance(row, (list, np.ndarray)):
                raise TypeError(f"object of type {type(row)} not supported")
            pandas_basic_type = type(row).__name__
            current_dtype = map_pandas_basic_type_to_feature_type(pandas_basic_type)
            if inferred_dtype is FeatureType.UNKNOWN:
                inferred_dtype = current_dtype
            else:
                if (
                    current_dtype != inferred_dtype
                    and current_dtype is not FeatureType.UNKNOWN
                ):
                    raise TypeError(
                        f"Input feature '{feature_name}' has mixed types, {current_dtype} and {inferred_dtype}. "
                        f"That is not allowed. "
                    )
    else:
        inferred_dtype = map_pandas_basic_type_to_feature_type(pandas_type)
    if inferred_dtype is FeatureType.UNKNOWN:
        raise TypeError(
            f"Input feature '{feature_name}' has type {str(pandas_type)} which is not supported"
        )
    else:
        return inferred_dtype


def map_pandas_basic_type_to_feature_type(pandas_type):
    """Returns the feature type corresponding to pandas_type
    :param pandas_type:
    :return:
    """
    # TODO uint64 with bigger number cant be mapped to LongType
    pandas_type_to_feature_type = {
        "str": FeatureType.STRING,
        "string": FeatureType.STRING,
        "int": FeatureType.INTEGER,
        "int8": FeatureType.INTEGER,
        "int16": FeatureType.INTEGER,
        "int32": FeatureType.LONG,
        "int64": FeatureType.LONG,
        "uint8": FeatureType.INTEGER,
        "uint16": FeatureType.INTEGER,
        "uint32": FeatureType.LONG,
        "uint64": FeatureType.LONG,
        "float": FeatureType.FLOAT,
        "float16": FeatureType.FLOAT,
        "float32": FeatureType.DOUBLE,
        "float64": FeatureType.DOUBLE,
        "datetime64[ns]": FeatureType.TIMESTAMP,
        "datetime64[ns, UTC]": FeatureType.TIMESTAMP,
        "timedelta64[ns]": FeatureType.LONG,
        "bool": FeatureType.BOOLEAN,
        "Decimal": FeatureType.DECIMAL,
        "date": FeatureType.DATE,
    }
    if pandas_type in pandas_type_to_feature_type:
        return pandas_type_to_feature_type.get(pandas_type)
    return FeatureType.UNKNOWN


def map_feature_type_to_spark_type(feature_type):
    """Returns the Spark Type for a particular feature type.
    :param feature_type:
    :return: Spark Type
    """
    feature_type_in = FeatureType(feature_type)
    spark_types = {
        FeatureType.STRING: StringType(),
        FeatureType.SHORT: ShortType(),
        FeatureType.INTEGER: IntegerType(),
        FeatureType.LONG: LongType(),
        FeatureType.FLOAT: FloatType(),
        FeatureType.DOUBLE: DoubleType(),
        FeatureType.BOOLEAN: BooleanType(),
        FeatureType.DATE: DateType(),
        FeatureType.TIMESTAMP: TimestampType(),
        FeatureType.DECIMAL: DecimalType(),
        FeatureType.BINARY: BinaryType(),
        FeatureType.STRING_ARRAY: ArrayType(StringType()),
        FeatureType.INTEGER_ARRAY: ArrayType(IntegerType()),
        FeatureType.SHORT_ARRAY: ArrayType(ShortType()),
        FeatureType.LONG_ARRAY: ArrayType(LongType()),
        FeatureType.FLOAT_ARRAY: ArrayType(FloatType()),
        FeatureType.DOUBLE_ARRAY: ArrayType(DoubleType()),
        FeatureType.BINARY_ARRAY: ArrayType(BinaryType()),
        FeatureType.DATE_ARRAY: ArrayType(DateType()),
        FeatureType.BOOLEAN_ARRAY: ArrayType(BooleanType()),
        FeatureType.TIMESTAMP_ARRAY: ArrayType(TimestampType()),
        FeatureType.STRING_STRING_MAP: MapType(StringType(), StringType()),
        FeatureType.STRING_INTEGER_MAP: MapType(StringType(), IntegerType()),
        FeatureType.STRING_SHORT_MAP: MapType(StringType(), ShortType()),
        FeatureType.STRING_LONG_MAP: MapType(StringType(), LongType()),
        FeatureType.STRING_FLOAT_MAP: MapType(StringType(), FloatType()),
        FeatureType.STRING_DOUBLE_MAP: MapType(StringType(), DoubleType()),
        FeatureType.STRING_DATE_MAP: MapType(StringType(), DateType()),
        FeatureType.STRING_TIMESTAMP_MAP: MapType(StringType(), TimestampType()),
        FeatureType.STRING_BOOLEAN_MAP: MapType(StringType(), BooleanType()),
        FeatureType.BYTE: ByteType(),
    }
    if feature_type_in in spark_types:
        return spark_types.get(feature_type_in)
    else:
        return "COMPLEX"


def get_raw_data_source_schema(raw_feature_details: List[dict]):
    """Converts input feature details to Spark schema.

    Args:
      raw_feature_details(List[dict]): List of input feature details.

    Returns:
      StructType: Spark schema.
      :param raw_feature_details:
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


def map_feature_type_to_pandas(feature_type):
    feature_type_in = FeatureType(feature_type)
    supported_feature_type = {
        FeatureType.STRING: str,
        FeatureType.LONG: "int64",
        FeatureType.DOUBLE: "float64",
        FeatureType.TIMESTAMP: "datetime64[ns]",
        FeatureType.BOOLEAN: "bool",
        FeatureType.FLOAT: "float32",
        FeatureType.INTEGER: "int32",
        FeatureType.DECIMAL: "object",
        FeatureType.DATE: "object",
        FeatureType.STRING_ARRAY: "object",
        FeatureType.INTEGER_ARRAY: "object",
        FeatureType.LONG_ARRAY: "object",
        FeatureType.FLOAT_ARRAY: "object",
        FeatureType.DOUBLE_ARRAY: "object",
        FeatureType.TIMESTAMP_ARRAY: "object",
        FeatureType.BOOLEAN_ARRAY: "object",
        # FeatureType.DECIMAL_ARRAY: "object",
        FeatureType.DATE_ARRAY: "object",
    }
    if feature_type_in in supported_feature_type:
        return supported_feature_type.get(feature_type_in)
    else:
        raise TypeError(f"Feature Type {feature_type} is not supported for pandas")


def map_spark_type_to_stats_data_type(spark_type):
    """Maps the spark data types to MLM library data types
    args:
        param spark_type: Spark data type input from the feature dataframe on which we need stats
    :return:
        Returns the MLM data type corresponding to SparkType
    """
    spark_type_to_mlm_data_type = {
        StringType(): types.DataType.STRING,
        IntegerType(): types.DataType.INTEGER,
        FloatType(): types.DataType.FLOAT,
        DoubleType(): types.DataType.FLOAT,
        BooleanType(): types.DataType.BOOLEAN,
        DecimalType(): types.DataType.FLOAT,
        ShortType(): types.DataType.INTEGER,
        LongType(): types.DataType.INTEGER,
    }

    return spark_type_to_mlm_data_type.get(spark_type)


def map_spark_type_to_stats_variable_type(spark_type):
    """Maps the spark data types to MLM library variable types
    args:
        param spark_type: Spark data type input from the feature dataframe on which we need stats
    :return:
        Returns the MLM variable type corresponding to SparkType
    """
    spark_type_to_feature_type = {
        StringType(): types.VariableType.NOMINAL,
        IntegerType(): types.VariableType.CONTINUOUS,
        FloatType(): types.VariableType.CONTINUOUS,
        DoubleType(): types.VariableType.CONTINUOUS,
        BooleanType(): types.VariableType.BINARY,
        DecimalType(): types.VariableType.CONTINUOUS,
        ShortType(): types.VariableType.CONTINUOUS,
        LongType(): types.VariableType.CONTINUOUS,
    }

    return spark_type_to_feature_type.get(spark_type)
