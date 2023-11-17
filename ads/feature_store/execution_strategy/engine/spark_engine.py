#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from datetime import datetime

from ads.common.decorator.runtime_dependency import OptionalDependency

try:
    from pyspark.sql import SparkSession
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `pyspark` module was not found. Please run `pip install "
        f"{OptionalDependency.SPARK}`."
    )
except Exception as e:
    raise
from typing import List, Dict

from ads.feature_store.common.utils.feature_schema_mapper import (
    map_spark_type_to_feature_type,
)

from ads.feature_store.common.enums import DataFrameType
from ads.feature_store.common.spark_session_singleton import SparkSessionSingleton

logger = logging.getLogger(__name__)


class SparkEngine:
    def __init__(self, metastore_id: str = None, spark_session: SparkSession = None):
        if spark_session:
            self.spark = spark_session
        else:
            self.spark = SparkSessionSingleton(metastore_id).get_spark_session()

        self.managed_table_location = (
            SparkSessionSingleton().get_managed_table_location()
        )

    def get_time_version_data(
        self,
        delta_table_name: str,
        version_number: int = None,
        timestamp: datetime = None,
    ):
        split_db_name = delta_table_name.split(".")

        # Get the Delta table path
        delta_table_path = (
            f"{self.managed_table_location}/{split_db_name[0].lower()}.db/{split_db_name[1]}"
            if self.managed_table_location
            else self._get_delta_table_path(delta_table_name)
        )

        # Set read options based on version_number and timestamp
        read_options = {}
        if version_number is not None:
            read_options["versionAsOf"] = version_number
        if timestamp:
            read_options["timestampAsOf"] = timestamp

        # Load the data from the Delta table using specified read options
        df = self._read_delta_table(delta_table_path, read_options)
        return df

    def _get_delta_table_path(self, delta_table_name: str) -> str:
        """
        Get the path of the Delta table using DESCRIBE EXTENDED SQL command.

        Args:
            delta_table_name (str): The name of the Delta table.

        Returns:
            str: The path of the Delta table.
        """
        delta_table_path = (
            self.spark.sql(f"DESCRIBE EXTENDED {delta_table_name}")
            .filter("col_name = 'Location'")
            .collect()[0][1]
        )
        return delta_table_path

    def _read_delta_table(self, delta_table_path: str, read_options: Dict):
        """
        Read the Delta table using specified read options.

        Args:
            delta_table_path (str): The path of the Delta table.
            read_options (dict): Dictionary of read options for Delta table.

        Returns:
            DataFrame: The loaded DataFrame from the Delta table.
        """
        df = (
            self.spark.read.format("delta")
            .options(**read_options)
            .load(delta_table_path)
        )
        return df

    def sql(
        self,
        query: str,
        dataframe_type: DataFrameType = DataFrameType.SPARK,
        is_online: bool = False,
    ):
        """Execute SQL command on the offline or online feature store database

        Arguments
            query: The SQL query to execute.
            dataframe_type: The type of the returned dataframe. Defaults to "default".
            is_online: Set to true to execute the query against the online feature store.
                Defaults to False.

        Returns
            `DataFrame`: DataFrame depending on the chosen type.
        """
        if is_online:
            raise ValueError("Online query is not supported.")

        response_spark_df = self.spark.sql(query)
        response_df = (
            response_spark_df.toPandas()
            if dataframe_type == DataFrameType.PANDAS
            else response_spark_df
        )

        return response_df

    def is_database_exists(self, database):
        """Checks whether the database exists or not.

        Args:
          database: A string specifying the name of the database.

        Returns:
          bool: True if the database exists, False otherwise.
        """
        databases = self.spark.catalog.listDatabases()

        return any(db.name == database for db in databases)

    def is_delta_table_exists(self, table_name) -> bool:
        """Checks whether the delta table exists or not.

        Args:
          table_name: A string specifying the name of the table.

        Returns:
          bool: True if the Delta table exists, False otherwise.
        """
        table_exist = False

        try:
            # Check if spark can read the table
            self.spark.read.table(table_name)
            table_exist = True

        except:
            pass

        return table_exist

    def get_tables_from_database(self, database):
        """Get a list of tables in the specified database using Spark SQL.

        Args:
          database: A string specifying the name of the database.

        Returns:
          List: A list of strings containing the names of the tables in the database.
        """
        permanent_tables = None

        try:
            tables_list = self.spark.catalog.listTables(database)

            # tables_list contains temporary tables also, so we need to filter it.
            permanent_tables = [table for table in tables_list if not table.isTemporary]
        except Exception as e:
            logger.error("Error: The database does not exist. ", e)

        return permanent_tables

    def get_output_columns_from_table_or_dataframe(
        self, table_name: str = None, dataframe=None
    ):
        """Returns the column(features) along with type from the given table.

        Args:
          table_name(str): A string specifying the name of table name for which columns should be returned.
          dataframe: Dataframe containing the transformed dataframe.

        Returns:
         List[{"name": "<feature_name>","featureType": "<feature_type>"}]
         Returns the List of dictionary of column with name and type from the given table.

        """
        if table_name is None and dataframe is None:
            raise ValueError(
                "Either 'table_name' or 'dataframe' must be provided to retrieve output columns."
            )

        if dataframe is not None:
            feature_data_target = dataframe
        else:
            feature_data_target = self.spark.sql(f"SELECT * FROM {table_name} LIMIT 1")

        target_table_columns = []

        for field in feature_data_target.schema.fields:
            target_table_columns.append(
                {
                    "name": field.name,
                    "featureType": map_spark_type_to_feature_type(field.dataType).value,
                }
            )
        return target_table_columns

    def convert_from_pandas_to_spark_dataframe(self, dataframe):
        """Converts a pandas DataFrame to an Apache Spark DataFrame.

        Args:
          dataframe (pandas.DataFrame): The pandas DataFrame to convert.

        Returns:
         pyspark.sql.DataFrame: The converted Apache Spark DataFrame.
        """
        return self.spark.createDataFrame(dataframe)

    def delete_spark_table(self, table_name: str):
        """
        Delete the specified Spark table from the Spark session.

        Args:
            table_name (str): The full name of the table to delete.

        Returns:
            None
        """

        # Construct SQL queries to drop the table (if it exists)
        drop_table_query = f"DROP TABLE IF EXISTS {table_name}"
        logger.info("Deleting the table with query: ", drop_table_query)
        self.spark.sql(drop_table_query)

    def delete_spark_database(self, database_name: str):
        """
        Delete the specified Spark database from the Spark session.

        Args:
            database_name (str): The name of the database to delete.

        Returns:
            None
        """
        # Construct SQL queries to drop the database (if it exists)
        drop_database_query = f"DROP DATABASE IF EXISTS {database_name}"
        self.spark.sql(drop_database_query)

    def remove_table_and_database(self, database: str, table: str) -> None:
        """
        Remove the specified Spark table and its database (if it exists) from the current Spark session.

        Args:
            database (str): The name of the database containing the table to remove.
            table (str): The name of the table to remove.

        Returns:
            None.

        Raises:
            None.

        This method first checks if the specified database exists in the current Spark session by calling the
        `is_database_exists()` method. If the database exists, it deletes the specified table from the database
        by calling the `delete_spark_table()` method.

        Next, the method checks if there are any tables left in the database by calling the `get_tables_from_database()` method.
        If the method returns a list of tables, it checks if the list is empty, and if so, it deletes the database
        by calling the `delete_spark_database()` method.

        If the `get_tables_from_database()` method returns `None`, no action is taken, and the method simply returns.

        If the specified database does not exist in the current Spark session, the method simply returns without doing anything.
        """
        if not self.is_database_exists(database):
            return

        table_name = f"{database}.{table}"
        self.delete_spark_table(table_name)

        tables_list = self.get_tables_from_database(database)
        if tables_list is not None and len(tables_list) == 0:
            self.delete_spark_database(database)

    def create_database(self, database: str) -> None:
        """
        Creates a database in Spark if it does not already exist.

        Args:
            database (str): The name of the database to create.

        Returns:
            None.

        """
        # Construct a SQL query to create the database if it does not exist
        create_database_query = f"CREATE DATABASE IF NOT EXISTS {database}"

        # Use the `spark.sql()` method to execute the SQL query
        self.spark.sql(create_database_query)
