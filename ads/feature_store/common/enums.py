#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from enum import Enum


class JobStatus(Enum):
    """
    An enumeration that represents the supported Job status.

    Attributes:
        SUCCEEDED (str): A string representation of the state of Succeeded job.
        FAILED (str): A string representation of the state of Failed job.
        CODE_EXECUTION (str): A string representation of the state of CodeExecution job.

    Methods:
        None
    """

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CODE_EXECUTION = "CODE_EXECUTION"


class LevelType(Enum):
    """
    An enumeration defining the different types of logging levels.

    Attributes:
        ERROR (str): A string representing the highest logging level, indicating an error in the program.
        WARNING (str): A string representing a lower logging level, indicating a potential issue or warning in the program.
    """

    ERROR = "ERROR"
    WARNING = "WARNING"


class DatasetIngestionMode(Enum):
    """
    An enumeration defining the possible modes for ingesting datasets.

    Attributes:
        SQL (str): A string representing the SQL mode, which is used to ingest datasets using SQL.
    """

    SQL = "SQL"


class IngestionType(Enum):
    """
    The type of ingestion that can be performed.

    Possible values:
        * STREAMING: The data is ingested in real time.
        * BATCH: The data is ingested in batches.
    """

    STREAMING = "STREAMING"
    BATCH = "BATCH"


class BatchIngestionMode(Enum):
    """
    An enumeration that represents the supported Ingestion Mode in feature store.

    Attributes:
        OVERWRITE (str): Ingestion mode to overwrite the data in the system.
        APPEND (str): Ingestion mode to append the data in the system.
        UPSERT (str): Ingestion mode to insert and update the data in the system.

    Methods:
        None
    """

    OVERWRITE = "OVERWRITE"
    APPEND = "APPEND"
    DEFAULT = "DEFAULT"
    UPSERT = "UPSERT"


class StreamingIngestionMode(Enum):
    """
    Enumeration for stream ingestion modes.

    - `COMPLETE`: Represents complete stream ingestion where the entire dataset is replaced.
    - `APPEND`: Represents appending new data to the existing dataset.
    - `UPDATE`: Represents updating existing data in the dataset.
    """

    COMPLETE = "COMPLETE"
    APPEND = "APPEND"
    UPDATE = "UPDATE"


class JoinType(Enum):
    """Enumeration of supported SQL join types.

    Attributes:
        INNER: Inner join.
        LEFT: Left join.
        RIGHT: Right join.
        FULL: Full outer join.
        CROSS: Cross join.
        LEFT_SEMI_JOIN: Left semi join.
    """

    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"
    LEFT_SEMI_JOIN = "LEFT_SEMI_JOIN"


class ExecutionEngine(Enum):
    """
    An enumeration that represents the supported execution engines.

    Attributes:
        SPARK (str): A string representation of the Apache Spark execution engine.
        PANDAS (str): A string representation of the Pandas execution engine.

    Methods:
        None
    """

    SPARK = "SPARK"
    PANDAS = "PANDAS"


class DataFrameType(Enum):
    """
    An enumeration that represents the supported DataFrame types.

    Attributes:
        SPARK (str): A string representation for spark Data frame type.
        PANDAS (str): A string representation for pandas Data frame type.

    Methods:
        None
    """

    SPARK = "SPARK"
    PANDAS = "PANDAS"


class ValidationEngineType(Enum):
    """
    An enumeration that represents the supported validation engines.

    Attributes:
        GREAT_EXPECTATIONS (str): A string representation of the great expectation execution engine.

    Methods:
        None
    """

    GREAT_EXPECTATIONS = "GREAT_EXPECTATIONS"


class FeatureStoreJobType(Enum):
    """
    An enumeration that represents the Job type.

    Attributes:
        FEATURE_GROUP_INGESTION (str): A string representing that job is feature group ingestion.
        DATASET_INGESTION (str): A string representing that job is dataset ingestion.
        FEATURE_GROUP_DELETION (str): A string representing that job is feature group deletion.
        DATASET_DELETION (str): A string representing that job is dataset deletion.

    Methods:
        None
    """

    FEATURE_GROUP_INGESTION = "FEATURE_GROUP_INGESTION"
    DATASET_INGESTION = "DATASET_INGESTION"
    FEATURE_GROUP_DELETION = "FEATURE_GROUP_DELETION"
    DATASET_DELETION = "DATASET_DELETION"


class LifecycleState(Enum):
    """
    An enumeration that represents the lifecycle state of feature store resources.

    Attributes:
        ACTIVE (str): A string representing Active resource.
        FAILED (str): A string representing Failed resource.
        NEEDS_ATTENTION (str): A string representing needs_attention resource.

    Methods:
        None
    """

    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    NEEDS_ATTENTION = "NEEDS_ATTENTION"


class JobConfigurationType(Enum):
    """
    An enumeration defining the different types of job configuration modes for Spark.

    Attributes:
        SPARK_BATCH_AUTOMATIC (str): A string representing automatic job configuration mode for Spark Batch jobs.
        SPARK_BATCH_MANUAL (str): A string representing manual job configuration mode for Spark Batch jobs.
    """

    SPARK_BATCH_AUTOMATIC = "SPARK_BATCH_AUTOMATIC"
    SPARK_BATCH_MANUAL = "SPARK_BATCH_MANUAL"


class ExpectationType(Enum):
    """
    An enumeration of the available expectation types for a feature store.

    Attributes:
        STRICT (str): A strict expectation type.
        LENIENT (str): A lenient expectation type.
        NO_EXPECTATION (str): A no expectation type.

    Methods:
        None
    """

    STRICT = "STRICT"
    LENIENT = "LENIENT"
    NO_EXPECTATION = "NO_EXPECTATION"


class TransformationMode(Enum):
    """
    An enumeration defining the different modes for data transformation.

    Attributes:
        SQL (str): A string representing the SQL mode, which is used to transform data using SQL queries.
        PANDAS (str): A string representing the Pandas mode, which is used to transform data using the Pandas library.
    """

    SQL = "sql"
    PANDAS = "pandas"
    SPARK = "spark"


class FilterOperators(Enum):
    """
    An enumeration defining the different comparison operators for data filtering.

    Attributes:
        GE (str): A string representing the greater than or equal to operator.
        GT (str): A string representing the greater than operator.
        NE (str): A string representing the not equals operator.
        EQ (str): A string representing the equals operator.
        LE (str): A string representing the less than or equal to operator.
        LT (str): A string representing the less than operator.
        IN (str): A string representing the in operator.
        LK (str): A string representing the like operator.
    """

    GE = "GREATER_THAN_OR_EQUAL"
    GT = "GREATER_THAN"
    NE = "NOT_EQUALS"
    EQ = "EQUALS"
    LE = "LESS_THAN_OR_EQUAL"
    LT = "LESS_THAN"
    IN = "IN"
    LK = "LIKE"


class FeatureType(Enum):
    """
    An enumeration of the available feature types for a feature store.

    Attributes:
        STRING (str): A string feature type.
        INTEGER (str): An integer feature type.
        FLOAT (str): A float feature type.
        DOUBLE (str): A double feature type.
        BOOLEAN (str): A boolean feature type.
        DATE (str): A date feature type.
        TIMESTAMP (str): A timestamp feature type.
        DECIMAL (str): A decimal feature type.
        BINARY (str): A binary feature type.
        ARRAY (str): An array feature type.
        MAP (str): A map feature type.
        STRUCT (str): A struct feature type.
    """

    STRING = "STRING"
    SHORT = "SHORT"
    INTEGER = "INTEGER"
    LONG = "LONG"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    BOOLEAN = "BOOLEAN"
    DATE = "DATE"
    TIMESTAMP = "TIMESTAMP"
    DECIMAL = "DECIMAL"
    BINARY = "BINARY"
    BYTE = "BYTE"
    STRING_ARRAY = "STRING_ARRAY"
    INTEGER_ARRAY = "INTEGER_ARRAY"
    SHORT_ARRAY = "SHORT_ARRAY"
    LONG_ARRAY = "LONG_ARRAY"
    FLOAT_ARRAY = "FLOAT_ARRAY"
    DOUBLE_ARRAY = "DOUBLE_ARRAY"
    BINARY_ARRAY = "BINARY_ARRAY"
    DATE_ARRAY = "DATE_ARRAY"
    TIMESTAMP_ARRAY = "TIMESTAMP_ARRAY"
    BYTE_ARRAY = "BYTE_ARRAY"
    BOOLEAN_ARRAY = "BOOLEAN_ARRAY"
    STRING_STRING_MAP = "STRING_STRING_MAP"
    STRING_INTEGER_MAP = "STRING_INTEGER_MAP"
    STRING_SHORT_MAP = "STRING_SHORT_MAP"
    STRING_LONG_MAP = "STRING_LONG_MAP"
    STRING_FLOAT_MAP = "STRING_FLOAT_MAP"
    STRING_DOUBLE_MAP = "STRING_DOUBLE_MAP"
    STRING_TIMESTAMP_MAP = "STRING_TIMESTAMP_MAP"
    STRING_DATE_MAP = "STRING_DATE_MAP"
    STRING_BYTE_MAP = "STRING_BYTE_MAP"
    STRING_BINARY_MAP = "STRING_BINARY_MAP"
    STRING_BOOLEAN_MAP = "STRING_BOOLEAN_MAP"
    UNKNOWN = "UNKNOWN"
    COMPLEX = "COMPLEX"


class EntityType(Enum):
    """
    An enumeration of the supported entity types.

    Attributes:
        FEATURE_GROUP (str): A string representing the feature group.
        DATASET (str): An string representing the dataset.
    """

    FEATURE_GROUP = "FEATURE_GROUP"
    DATASET = "DATASET"
