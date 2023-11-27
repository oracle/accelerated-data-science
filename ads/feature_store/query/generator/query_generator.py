#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging

from ads.feature_store.common.utils.utility import (
    largest_matching_subset_of_primary_keys,
)
from ads.feature_store.query.filter import Logic, Filter
from ads.feature_store.query.join import Join

logger = logging.getLogger(__name__)


class NotSupportedError(Exception):
    """Exception raised when an operation is not supported.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="This operation is not supported."):
        """
        Initialize a new instance of the NotSupportedError class.

        Arguments:
        message -- explanation of the error
        """
        self.message = message
        super().__init__(self.message)


class QueryGenerator:
    CONST_FEATURE_GROUP_ALIAS = "fg_{index}"
    CONST_DATASET_ALIAS = "ds_{index}"

    def __init__(self, query):
        self.query = query
        self.sql_filter = SQLFilter()

    def get_table_alias(self, index):
        """
        Generates a table alias for a feature group table using a constant feature group alias and the provided index.

        Parameters:
            index (int): The index to insert into the feature group alias string.

        Returns:
            str: A string containing the formatted table alias for the feature group table.
        """
        return self.CONST_FEATURE_GROUP_ALIAS.format(index=index)

    def generate_query(self, is_online: bool = False) -> str:
        """
        Generate a SELECT query for a database table.

        Args:
            is_online (bool, optional): Whether the query should be executed online.
                Defaults to False.

        Raises:
            NotSupportedError: If is_online is True.

        Returns:
            str: A string containing the SELECT statement.
        """

        if is_online:
            raise NotSupportedError("Online query is not supported.")

        selected_features_map = {}
        index = 0
        on_condition = []
        table = f"`{self.query.entity_id}`.{self.query.left_feature_group.name}"
        table_alias = self.get_table_alias(len(self.query.joins))
        left_table_alias = table_alias

        # store the left features in the map
        selected_features_map[table_alias] = self.query.left_features
        feature_group_id_map = {self.query.left_feature_group.id: table_alias}

        for join in self.query.joins:
            # Ge table and alias and map the features
            sub_query = join.sub_query
            right_table = f"`{sub_query.entity_id}`.{sub_query.left_feature_group.name}"
            right_table_alias = self.get_table_alias(index)
            selected_features_map[right_table_alias] = sub_query.left_features

            feature_group_id_map[sub_query.left_feature_group.id] = right_table_alias

            # Update the index
            index += 1

            on_condition.append(
                self._get_on_condition(
                    left_table_alias, right_table, right_table_alias, join
                )
            )

            left_table_alias = right_table_alias

        selected_columns = self._get_selected_columns(selected_features_map)
        filters = (
            None
            if self.query._filter is None
            else self.sql_filter.get_filter_expression(
                self.query._filter, feature_group_id_map
            )
        )

        # Define the SQL query as an f-string
        query = f"SELECT {selected_columns} FROM {table} {table_alias}"

        if on_condition:
            # If there is an ON condition, add it to the query
            query += f" {' '.join(on_condition)}"

        if filters:
            query += f" where {filters}"

        return query

    @staticmethod
    def _get_selected_columns(selected_features_map: dict):
        """
        Generates a list of selected columns with aliases.

        Args:
            selected_features_map: A dictionary mapping feature group names to a list of selected feature names.

        Returns:
            A comma-separated string of selected columns, each column having its feature group name as a prefix
            and feature name as an alias.
        """
        processed_features = {}
        columns_with_alias = []

        for feature_group_name, features in selected_features_map.items():
            for feature in features:
                if feature not in processed_features:
                    column_alias = f"{feature_group_name}.{feature} {feature}"
                    columns_with_alias.append(column_alias)
                    processed_features[feature] = True

        return ", ".join(columns_with_alias)

    def _get_on_condition(
        self,
        left_table_alias: str,
        right_table: str,
        right_table_alias: str,
        join: Join,
    ):
        """
        Helper function that returns the ON condition for a JOIN statement.

        Parameters:
           left_table_alias (str): Alias for the left table in the join.
           right_table (str): Name of the right table in the join.
           right_table_alias (str): Alias for the right table in the join.
           join (Join): Join

        Returns:
           str: ON condition for the JOIN statement.
        """
        if join.on:
            # If there is an explicit ON clause, use it to generate the join condition.
            keys = join.on
            join_condition = " AND ".join(
                [
                    f"{left_table_alias}.{key} = {right_table_alias}.{key}"
                    for key in keys
                ]
            )
        elif join.left_on and join.right_on:
            # If there is no explicit ON clause, use the left_on and right_on keys to generate the join condition.
            join_condition = " AND ".join(
                [
                    f"{left_table_alias}.{left_key} = {right_table_alias}.{right_key}"
                    for left_key, right_key in zip(join.left_on, join.right_on)
                ]
            )
        else:
            matching_primary_keys = largest_matching_subset_of_primary_keys(
                self.query.left_feature_group, join.sub_query.left_feature_group
            )
            join_condition = " AND ".join(
                [
                    f"{left_table_alias}.{key} = {right_table_alias}.{key}"
                    for key in matching_primary_keys
                ]
            )

        # Combine the join type, right table name, right table alias, and join condition to create the ON clause.
        _on_condition = f"{join.join_type} JOIN {right_table} {right_table_alias} ON {join_condition}"
        return _on_condition


class SQLFilter:
    """
    Provides utility methods for processing filters and logic for an SQL query.
    """

    @staticmethod
    def _get_expression(operator: str) -> str:
        """
        Maps filter conditions to their SQL equivalents.

        Args:
            operator (str): The filter condition to map.

        Returns:
            str: The SQL equivalent of the filter condition.
        """
        mapping = {
            "GREATER_THAN_OR_EQUAL": ">=",
            "GREATER_THAN": ">",
            "NOT_EQUALS": "!=",
            "EQUALS": "==",
            "LESS_THAN_OR_EQUAL": "<=",
            "LESS_THAN": "<",
            "IN": "in",
            "LIKE": "like",
        }
        return mapping.get(operator)

    @staticmethod
    def _get_value_using_type(value: str, feature_type: str) -> str:
        """
        Converts a filter value to the appropriate data type and formats it for use in an SQL query.

        Args:
            value (str): The filter value to convert.
            feature_type (str): The data type of the filter value.

        Returns:
            str: The converted and formatted filter value.
        """
        if feature_type == "int":
            value = int(value)
        elif feature_type == "bool":
            value = value.lower() == "true"
        elif feature_type == "float":
            value = float(value)

        if isinstance(value, str):
            value = "'" + value + "'"

        return str(value)

    def _process_filter(self, filter: Filter, feature_group_id_map: dict) -> str:
        """
        Processes a single filter and returns its SQL string representation.

        Args:
            filter (Filter): The filter to process.
            feature_group_id_map (dict): A mapping of feature group IDs to table aliases.

        Returns:
            str: The SQL string representation of the filter.
        """
        condition = self._get_expression(filter.condition)
        value = self._get_value_using_type(filter.value, filter.feature.feature_type)
        feature_name = filter.feature.feature_name
        feature_group_id = filter.feature.feature_group_id
        table_alias = feature_group_id_map[feature_group_id]

        return f"{table_alias}.{feature_name} {condition} {value}"

    def get_filter_expression(self, logic: Logic, feature_group_id_map: dict) -> str:
        """
        Processes a logic tree of filters and returns its SQL string representation.

        Args:
            logic (Logic): The root node of the logic tree to process.
            feature_group_id_map (dict): A mapping of feature group IDs to table aliases.

        Returns:
            str: The SQL string representation of the logic tree.
        """
        if logic.type == "SINGLE":
            return self._process_filter(logic.left_filter, feature_group_id_map)

        left_expression = (
            self.get_filter_expression(logic.left_logic, feature_group_id_map)
            if logic.left_logic
            else self._process_filter(logic.left_filter, feature_group_id_map)
        )
        right_expression = (
            self.get_filter_expression(logic.right_logic, feature_group_id_map)
            if logic.right_logic
            else self._process_filter(logic.right_filter, feature_group_id_map)
        )

        return f"({left_expression} {logic.type} {right_expression})"
