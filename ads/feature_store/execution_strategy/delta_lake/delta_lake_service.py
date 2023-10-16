#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging

from ads.common.decorator.runtime_dependency import OptionalDependency
from ads.feature_store.common.enums import BatchIngestionMode
from ads.feature_store.execution_strategy.engine.spark_engine import SparkEngine

try:
    from delta.tables import *
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `delta` module was not found. Please run `pip install "
        f"{OptionalDependency.SPARK}` to install delta and spark."
    )
except Exception as e:
    raise

logger = logging.getLogger(__name__)


class DeltaLakeService:
    target_delta_table_alias = "target_delta_table"
    source_delta_table_alias = "source_delta_table"
    DELTA_SCHEMA_EVOLUTION_OPTIONS = ["mergeSchema", "overwriteSchema"]

    def __init__(self, spark_session: SparkSession):
        self._spark_session = spark_session
        self.spark_engine = SparkEngine(spark_session=self._spark_session)

    def write_dataframe_to_delta_lake(
        self,
        dataflow_output,
        target_table_name,
        delta_table_primary_key,
        partition_keys,
        ingestion_mode,
        raw_schema,
        feature_options=None,
    ):
        """Writes the given data flow output to the Delta table.

        Args:
            dataflow_output (DataFrame): The data frame that needs to be written to the Delta table.
            target_table_name (str): The name of the target Delta table.
            delta_table_primary_key (List[dict]): The list of primary keys for the target Delta table.
            partition_keys(List[dict]): The List of partition Keys.
            ingestion_mode (str): The ingestion mode for the data load.
            raw_schema (StructType): The schema of the raw data being ingested.
            feature_options (Dict[str, Union[str, int, float, bool]]): Optional. The dictionary containing feature options.

        Returns:
            None.
        """
        logger.info(f"target table name {target_table_name}")

        if (
            self.spark_engine.is_delta_table_exists(target_table_name)
            and ingestion_mode.upper() == BatchIngestionMode.UPSERT.value
        ):
            logger.info(f"Upsert ops for target table {target_table_name} begin")

            if raw_schema is not None:
                # Get the source and target table columns
                source_table_columns = raw_schema.names
                target_table_columns = [
                    column_details.get("name")
                    for column_details in self.spark_engine.get_columns_from_table(
                        target_table_name
                    )
                ]

                logger.info(f"source table columns {source_table_columns}")
                logger.info(f"target table columns {target_table_columns}")

                if all(
                    feature in target_table_columns for feature in source_table_columns
                ):
                    logger.info(
                        f"execute upsert for select columns  {source_table_columns}"
                    )

                    self.__execute_delta_merge_insert_update(
                        delta_table_primary_key,
                        target_table_name,
                        dataflow_output,
                        source_table_columns,
                        feature_options,
                    )
                else:
                    logger.info(
                        f"execute upsert for all columns {source_table_columns}"
                    )
                    self.__execute_delta_merge_insert_update_all(
                        delta_table_primary_key,
                        target_table_name,
                        dataflow_output,
                        feature_options,
                    )
            else:
                self.__execute_delta_merge_insert_update_all(
                    delta_table_primary_key,
                    target_table_name,
                    dataflow_output,
                    feature_options,
                )
            logger.info(f"Upsert ops for target table {target_table_name} ended")
        else:
            self.save_delta_dataframe(
                dataflow_output,
                ingestion_mode,
                target_table_name,
                feature_options,
                partition_keys,
            )

    def __execute_delta_merge_insert_update(
        self,
        primary_keys,
        target_table_name,
        source_dataframe,
        source_table_columns,
        feature_options,
    ):
        """Executes a Delta merge, insert, or update operation based on the given parameters.

        Parameters
        ----------
        primary_keys
            A list of primary key column names used for the merge operation.
        target_table_name
            The name of the target Delta table.
        source_dataframe
            The DataFrame representing the source data for the merge operation.
        source_table_columns
            A list of column names for the source table.
        feature_options
            A dictionary containing feature options.
        """

        # Enable the schema evolution
        self.__enable_schema_evolution(feature_options)

        target_delta_table = DeltaTable.forName(self._spark_session, target_table_name)
        source_dataframe.registerTempTable(self.source_delta_table_alias)

        insert_update_set = self.__get_insert_update_query_expression(
            source_table_columns, self.source_delta_table_alias
        )

        on_condition = self.__get_delta_table_on_condition(
            self.target_delta_table_alias, self.source_delta_table_alias, primary_keys
        )

        return (
            target_delta_table.alias(self.target_delta_table_alias)
            .merge(
                source_dataframe.alias(self.source_delta_table_alias), f"{on_condition}"
            )
            .whenMatchedUpdate(set=insert_update_set)
            .whenNotMatchedInsert(values=insert_update_set)
            .execute()
        )

    def __execute_delta_merge_insert_update_all(
        self, primary_keys, target_table_name, dataframe, feature_options=None
    ):
        """Perform a merge operation on Delta tables to update and insert rows into the target table. This method
        merges all columns from the source table, irrespective of whether they exist in the target table.

        Args:
            primary_keys (List[dict]): A dictionary containing the primary keys for the target Delta table. The dictionary
                should have a single key 'items', which maps to a list of dictionaries containing the names of the
                primary key columns.
            target_table_name (str): The name of the target Delta table.
            dataframe (DataFrame): The DataFrame containing the data to merge into the target Delta table.
            feature_options (dict): A dictionary containing feature options.

        Returns:
            None
        """
        # Enable the schema
        self.__enable_schema_evolution(feature_options)
        target_delta_table = DeltaTable.forName(self._spark_session, target_table_name)
        on_condition = self.__get_delta_table_on_condition(
            self.target_delta_table_alias, self.source_delta_table_alias, primary_keys
        )

        return (
            target_delta_table.alias(self.target_delta_table_alias)
            .merge(dataframe.alias(self.source_delta_table_alias), f"{on_condition}")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    def save_delta_dataframe(
        self,
        dataframe,
        dataframe_ingestion_mode,
        table_name,
        feature_options=None,
        partition_keys=None,
    ):
        """
        Saves a DataFrame to a Delta table with the specified options.

        Args:
            dataframe (pandas.DataFrame): The DataFrame to save.
            dataframe_ingestion_mode (str): The mode to use when ingesting the DataFrame.
            table_name (str): The name of the Delta table to save the DataFrame to.
            feature_options (dict): Optional feature options to use when saving the DataFrame.

        """
        delta_partition_keys = []

        if partition_keys:
            partition_keys_items = partition_keys["items"]
            if partition_keys_items:
                delta_partition_keys = [
                    partition_key.get("name") for partition_key in partition_keys_items
                ]

        if feature_options and feature_options.get("featureOptionWriteConfigDetails"):
            feature_delta_write_option_config = feature_options.get(
                "featureOptionWriteConfigDetails"
            )

            logger.info(
                f"feature options write config details: {feature_delta_write_option_config}"
            )

            dataframe.write.format("delta").options(
                **self.get_delta_write_config(feature_delta_write_option_config)
            ).mode(dataframe_ingestion_mode).partitionBy(
                delta_partition_keys
            ).saveAsTable(
                table_name
            )
        else:
            dataframe.write.format("delta").mode(dataframe_ingestion_mode).partitionBy(
                delta_partition_keys
            ).saveAsTable(table_name)

    def get_delta_write_config(self, feature_delta_write_option_config):
        """Returns a dictionary containing delta schema configuration options based on a given dictionary of feature
        delta write options.

        Parameters
        ----------
        feature_delta_write_option_config
            A dictionary containing feature delta write options.

        Returns
        -------
        A dictionary containing delta schema configuration options.
        """
        delta_schema_config = {}

        if feature_delta_write_option_config:
            for key in self.DELTA_SCHEMA_EVOLUTION_OPTIONS:
                if (
                    key in feature_delta_write_option_config
                    and feature_delta_write_option_config[key] is not None
                ):
                    delta_schema_config[key] = str(
                        feature_delta_write_option_config[key]
                    )

        return delta_schema_config

    def __enable_schema_evolution(self, feature_options):
        """Enables schema evolution for Delta tables based on the given feature options.

        Parameters
        ----------
        feature_options
            A dictionary containing feature options.

        Returns
        -------
        A dictionary containing delta schema configuration options
        """

        if feature_options and feature_options.get("featureOptionWriteConfigDetails"):
            # enable auto merge schema for the spark session
            self._spark_session.conf.set(
                "spark.databricks.delta.schema.autoMerge.enabled", "true"
            )

    @staticmethod
    def __get_delta_table_on_condition(
        target_delta_table, source_delta_table, primary_keys
    ):
        """Returns the ON condition for the merge operation between target_delta_table and source_delta_table,
        based on the primary keys defined in primary_keys.

        Parameters
        ----------
        target_delta_table
            the name of the target delta table.
        source_delta_table
            the name of the source delta table.
        primary_keys
            primary key information of the table.

        Returns
        --------
        str
            The ON condition for the merge operation.
        """
        primary_key_items = primary_keys["items"]
        output = " AND ".join(
            f"{target_delta_table}.{pk['name']} = {source_delta_table}.{pk['name']}"
            for pk in primary_key_items
        )
        logger.info(f"Primary key on condition: {output}")

        return output

    @staticmethod
    def __get_insert_update_query_expression(feature_data_source_columns, table_name):
        """Generates an insert/update query expression to merge data from a source dataframe to a target Delta table.

        Args:
            feature_data_source_columns (list): List of columns from the source dataframe.
            table_name (str): Name of the target Delta table.

        Returns:
            dict: A dictionary containing the update set expressions for each feature column.

        """
        feature_data_update_set = {}

        for feature_column in feature_data_source_columns:
            target_column_field = str(table_name) + "." + feature_column
            feature_data_update_set[feature_column] = target_column_field

        logger.info(f"get_insert_update_query_expression {feature_data_update_set}")
        return feature_data_update_set

    def write_stream_dataframe_to_delta_lake(
        self,
        stream_dataframe,
        target_table,
        output_mode,
        query_name,
        await_termination,
        timeout,
        checkpoint_dir,
        feature_option_details,
    ):
        if query_name is None:
            query_name = "insert_stream_" + target_table.split(".")[1]

        query = (
            stream_dataframe.writeStream.outputMode(output_mode)
            .format("delta")
            .option(
                "checkpointLocation",
                checkpoint_dir,
            )
            .options(**self.get_delta_write_config(feature_option_details))
            .queryName(query_name)
            .toTable(target_table)
        )

        if await_termination:
            query.awaitTermination(timeout)

        return query
