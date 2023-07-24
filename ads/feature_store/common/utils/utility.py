#!/usr/bin/env python
# -*- coding: utf-8; -*-
import copy
import os

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Union, List

from great_expectations.core import ExpectationSuite

from ads.common.decorator.runtime_dependency import OptionalDependency
from ads.feature_store.common.utils.feature_schema_mapper import (
    map_spark_type_to_feature_type,
    map_feature_type_to_pandas,
)
from ads.feature_store.feature import Feature, DatasetFeature
from ads.feature_store.feature_group_expectation import Rule, Expectation
from ads.feature_store.input_feature_detail import FeatureDetail
from ads.feature_store.common.spark_session_singleton import SparkSessionSingleton

try:
    from pyspark.pandas import DataFrame
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `pyspark` module was not found. Please run `pip install "
        f"{OptionalDependency.SPARK}`."
    )
except Exception as e:
    raise
import pandas as pd

from ads.feature_store.common.enums import (
    ExecutionEngine,
    FeatureType,
    ExpectationType,
    ValidationEngineType,
    EntityType,
)
import logging

from ads.feature_engineering.feature_type import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_execution_engine_type(
    data_frame: Union[DataFrame, pd.DataFrame]
) -> ExecutionEngine:
    """
    Determines the execution engine type for a given DataFrame.

    Args:
        data_frame (Union[DataFrame, pd.DataFrame]): The DataFrame whose execution engine type should be determined.

    Returns:
        ExecutionEngine: The execution engine type for the given DataFrame.
    """
    return (
        ExecutionEngine.PANDAS
        if isinstance(data_frame, pd.DataFrame)
        else ExecutionEngine.SPARK
    )


def get_metastore_id(feature_store_id: str):
    """
    Retrieves the metastore ID for a given feature store ID.

    Args:
        feature_store_id (str): The ID of the feature store.

    Returns:
        str: The metastore ID for the feature store, if available. Otherwise, returns None.
    """
    from ads.feature_store.feature_store import FeatureStore

    feature_store = FeatureStore.from_id(feature_store_id)

    return (
        feature_store.offline_config.get(feature_store.CONST_METASTORE_ID)
        if feature_store.offline_config
        else None
    )


def validate_delta_format_parameters(
    timestamp: datetime = None, version_number: int = None, is_restore: bool = False
):
    """
    Validate the user input provided as part of preview, restore APIs for ingested data, Ingested data is
    getting saved in versioned manner where every commit generates a commit timestamp and auto increment current version.
    This information will be used in order to provide timetravel and restore support

    Args:
        timestamp (datetime): The commit timestamp for ingestion date time
        version_number: The commit version number for ingested data
        is_restore: additional restore check  to be enabled for

    Returns:
    """

    if timestamp is not None and version_number is not None:
        logger.error(
            f"timestamp {timestamp} and version number {version_number} both are present"
        )
        raise Exception(
            f"Timestamp and version number cannot be passed at the same time"
        )
    elif is_restore and timestamp is None and version_number is None:
        logger.error(f"Either timestamp or version number must be provided for restore")
        raise Exception(
            f"Either timestamp or version number must be provided for restore"
        )
    else:
        if version_number is not None and version_number < 0:
            logger.error(f"version number {version_number} cannot be negative")
            raise Exception(f"version number cannot be negative")


def show_ingestion_summary(
    entity_id: str,
    entity_type: EntityType = EntityType.FEATURE_GROUP,
    error_details: str = None,
):
    """
    Displays a ingestion summary table with the given entity type and error details.

    Args:
        entity_id: str
        entity_type (EntityType, optional): The type of entity being ingested. Defaults to EntityType.FEATURE_GROUP.
        error_details (str, optional): Details of any errors that occurred during ingestion. Defaults to None.
    """
    from tabulate import tabulate

    table_headers = ["entity_id", "entity_type", "ingestion_status", "error_details"]
    ingestion_status = "Failed" if error_details else "Succeeded"

    table_values = [
        entity_id,
        entity_type.value,
        ingestion_status,
        error_details if error_details else "None",
    ]

    logger.info(
        "Ingestion Summary \n"
        + tabulate(
            [table_values],
            headers=table_headers,
            tablefmt="fancy_grid",
            numalign="center",
            stralign="center",
        )
    )


def show_validation_summary(ingestion_status: str, validation_output, expectation_type):
    from tabulate import tabulate

    statistics = validation_output["statistics"]

    table_headers = (
        ["expectation_type"] + list(statistics.keys()) + ["ingestion_status"]
    )

    table_values = [expectation_type] + list(statistics.values()) + [ingestion_status]

    logger.info(
        "Validation Summary \n"
        + tabulate(
            [table_values],
            headers=table_headers,
            tablefmt="fancy_grid",
            numalign="center",
            stralign="center",
        )
    )

    rule_table_headers = ["rule_type", "arguments", "status"]

    rule_table_values = [
        [
            rule_output["expectation_config"].get("expectation_type"),
            {
                key: value
                for key, value in rule_output["expectation_config"]["kwargs"].items()
                if key != "batch_id"
            },
            rule_output.get("success"),
        ]
        for rule_output in validation_output["results"]
    ]

    logger.info(
        "Validations Rules Summary \n"
        + tabulate(
            rule_table_values,
            headers=rule_table_headers,
            tablefmt="fancy_grid",
            numalign="center",
            stralign="center",
        )
    )


def get_features(
    output_columns: List[dict],
    parent_id: str,
    entity_type: EntityType = EntityType.FEATURE_GROUP,
) -> List[Feature]:
    """
    Returns a list of features, given a list of output_columns and a feature_group_id.

    Parameters:
      output_columns (List[dict]): A list of dictionaries representing the output columns, with keys "name" and "featureType".
      parent_id (str): String representing the ID of the Parent that could be FeatureGroup or Dataset.
      entity_type (EntityType): String representing the Entity Type.

    Returns:
      features (List[Feature]): A list of Feature objects representing the features.
    """
    features = []

    # Loop through each output column and create a Feature object with the name, featureType, and feature_group_id.
    for output_column in output_columns:
        features.append(
            Feature(
                output_column.get("name"),
                output_column.get("featureType"),
                parent_id,
            )
            if entity_type == EntityType.FEATURE_GROUP
            else DatasetFeature(
                output_column.get("name"),
                output_column.get("featureType"),
                parent_id,
            )
        )

    return features


def get_schema_from_pandas_df(df: pd.DataFrame, feature_store_id: str):
    spark = SparkSessionSingleton(
        get_metastore_id(feature_store_id)
    ).get_spark_session()
    converted_df = spark.createDataFrame(df)
    return get_schema_from_spark_df(converted_df)


def get_schema_from_spark_df(df: DataFrame):
    schema_details = []

    for order_number, field in enumerate(df.schema.fields, start=1):
        details = {
            "name": field.name,
            "feature_type": map_spark_type_to_feature_type(field.dataType),
            "order_number": order_number,
        }
        schema_details.append(details)

    return schema_details


def get_schema_from_df(
    data_frame: Union[DataFrame, pd.DataFrame], feature_store_id: str
) -> List[dict]:
    """
    Given a DataFrame, returns a list of dictionaries that describe its schema.
    If the DataFrame is a pandas DataFrame, it uses pandas methods to get the schema.
    If it's a PySpark DataFrame, it uses PySpark methods to get the schema.
    """
    if isinstance(data_frame, pd.DataFrame):
        return get_schema_from_pandas_df(data_frame, feature_store_id)
    else:
        return get_schema_from_spark_df(data_frame)


def get_input_features_from_df(
    data_frame: Union[DataFrame, pd.DataFrame], feature_store_id: str
) -> List[FeatureDetail]:
    """
    Given a DataFrame, returns a list of FeatureDetail objects that represent its input features.
    Each FeatureDetail object contains information about a single input feature, such as its name, data type, and
    whether it's categorical or numerical.
    """
    schema_details = get_schema_from_df(data_frame, feature_store_id)
    feature_details = []

    for schema_detail in schema_details:
        feature_details.append(FeatureDetail(**schema_detail))

    return feature_details


def convert_expectation_suite_to_expectation(
    expectation_suite: ExpectationSuite, expectation_type: ExpectationType
):
    """
    Convert an ExpectationSuite object to an Expectation object with detailed rule information.

    Args:
        expectation_suite (ExpectationSuite): The ExpectationSuite object to convert.
        expectation_type (ExpectationType): The type of expectation to assign to the resulting Expectation object.

    Returns:
        An Expectation object with the specified expectation_type and detailed rule information extracted from the
        expectation_suite.
    """
    expectation_rules = []

    index = 0
    for expectation_config in expectation_suite.expectations:
        expectation_rules.append(
            Rule(f"Rule-{index}")
            .with_rule_type(expectation_config.expectation_type)
            .with_arguments(expectation_config.kwargs)
        )
        index += 1

    return (
        Expectation(expectation_suite.expectation_suite_name)
        .with_expectation_type(expectation_type)
        .with_validation_engine_type(ValidationEngineType.GREAT_EXPECTATIONS)
        .with_rule_details(expectation_rules)
    )


def largest_matching_subset_of_primary_keys(left_feature_group, right_feature_group):
    """
    Returns the largest matching subset of primary keys between the left feature group and right feature group.

    Args:
        left_feature_group: A feature group object containing primary keys.
        right_feature_group: A feature group object containing primary keys.

    Returns:
        A set of primary key names that are common to both the left feature group and the input feature group.
    """

    # Get the primary keys for each of the feature groups.
    left_primary_keys = set(
        item["name"] for item in left_feature_group.primary_keys.get("items")
    )
    right_primary_keys = set(
        item["name"] for item in right_feature_group.primary_keys.get("items")
    )

    # Find the intersection of the two sets
    common_keys = left_primary_keys.intersection(right_primary_keys)

    return common_keys


def convert_pandas_datatype_with_schema(
    raw_feature_details: List[dict], input_df: pd.DataFrame
) -> pd.DataFrame:
    feature_detail_map = {}
    columns_to_remove = []
    for feature_details in raw_feature_details:
        feature_detail_map[feature_details.get("name")] = feature_details
    for column in input_df.columns:
        if column in feature_detail_map.keys():
            feature_details = feature_detail_map[column]
            feature_type = feature_details.get("featureType")
            pandas_type = map_feature_type_to_pandas(feature_type)
            input_df[column] = (
                input_df[column]
                .astype(pandas_type)
                .where(pd.notnull(input_df[column]), None)
            )
        else:
            logger.warning(
                "column" + column + "doesn't exist in the input feature details"
            )
            columns_to_remove.append(column)
    return input_df.drop(columns=columns_to_remove)


def convert_spark_dataframe_with_schema(
    raw_feature_details: List[dict], input_df: DataFrame
) -> DataFrame:
    feature_detail_map = {}
    columns_to_remove = []
    for feature_details in raw_feature_details:
        feature_detail_map[feature_details.get("name")] = feature_details
    for column in input_df.columns:
        if column not in feature_detail_map.keys():
            logger.warning(
                "column" + column + "doesn't exist in the input feature details"
            )
            columns_to_remove.append(column)

    return input_df.drop(*columns_to_remove)


def validate_input_feature_details(input_feature_details, data_frame):
    if isinstance(data_frame, pd.DataFrame):
        return convert_pandas_datatype_with_schema(input_feature_details, data_frame)
    return convert_spark_dataframe_with_schema(input_feature_details, data_frame)
