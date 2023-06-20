#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import logging
from typing import Optional, List, Union

from ads.feature_store.common.enums import JoinType
from ads.feature_store.common.utils.utility import get_metastore_id
from ads.feature_store.execution_strategy.engine.spark_engine import SparkEngine
from ads.feature_store.query.filter import Filter, Logic
from ads.feature_store.query.join import Join
from ads.feature_store.query.validator.query_validator import QueryValidator
from ads.jobs.builders.base import Builder

logger = logging.getLogger(__name__)


class Query(Builder):
    """
    A class representing a query that can be executed against a Spark dataframe.

    Args:
        left_feature_group (FeatureGroup): The left feature group to query.
        left_features (List[str]): A list of left feature names to include in the query.
        feature_store_id (str, optional): The ID of the feature store to query. Defaults to None.
        entity_id (str, optional): The ID of the entity to query. Defaults to None.

    Methods:
        with_left_feature_group(feature_group: FeatureGroup) -> Query:
            Sets the left feature group to query.
        with_entity_id(entity_id: str) -> Query:
            Sets the ID of the entity to query.
        with_feature_store_id(feature_store_id: str) -> Query:
            Sets the ID of the feature store to query.
        with_left_features(feature_group: FeatureGroup) -> Query:
            Sets the left features to query based on the given
            feature group.
        with_left_features(features: List[str]) -> Query:
            Sets the left features to query based on the given list of feature names.
        read(online: Optional[bool] = False) -> DataFrame:
            Executes the query and returns the result as a Spark dataframe.
        show(n: int = 10, online: Optional[bool] = False) -> None:
            Executes the query and prints the first n rows of the result.
        to_dict() -> Dict: Returns a dictionary representation of the query.
        with_filter(self, _filter) -> "Query":
            Sets the filter for the query.
        to_string(is_online) -> str:
            Returns the query as string.
        filter(self, f: Union[Filter, Logic]) -> "Query":
            Applies the filter to the query.


    Properties:
        left_feature_group (FeatureGroup): Gets or sets the left feature group to query.
        entity_id (str): Gets or sets the ID of the entity to query.
        feature_store_id (str): Gets or sets the ID of the feature store to query.
        left_features (List[str]): Gets or sets the list of left feature names to include in the query.
    """

    CONST_LEFT_FEATURE_GROUP = "leftFeatureGroup"
    CONST_LEFT_FEATURES = "leftFeatures"
    CONST_FEATURE_STORE_ID = "featureStoreId"
    CONST_ENTITY_ID = "entityId"
    CONST_JOINS = "joins"
    CONST_FILTER = "_filter"

    def __init__(
        self,
        left_feature_group,
        left_features,
        feature_store_id=None,
        entity_id=None,
        joins=None,
    ):
        super().__init__()
        self.spark_engine = SparkEngine(
            metastore_id=get_metastore_id(feature_store_id=feature_store_id)
        )
        self.with_left_feature_group(left_feature_group)
        self.with_entity_id(entity_id)
        self.with_feature_store_id(feature_store_id)
        self.with_left_features(features=left_features)
        self.with_joins(joins or [])

    @property
    def left_feature_group(self):
        return self.get_spec(self.CONST_LEFT_FEATURE_GROUP)

    @left_feature_group.setter
    def left_feature_group(self, value):
        self.with_left_feature_group(value)

    def with_left_feature_group(self, feature_group) -> "Query":
        """Sets the entity_id.

        Parameters
        ----------
        feature_group: FeatureGroup
            The feature_group.

        Returns
        -------
        Query
            The Query instance (self)
        """
        return self.set_spec(self.CONST_LEFT_FEATURE_GROUP, feature_group)

    def get_last_joined_feature_group(self):
        """
        Retrieves the last joined feature group from the list of joins,
        or returns the left feature group if no joins are present.

        Returns:
            The last joined feature group if the list of joins is non-empty,
            otherwise returns the left feature group.
        """
        if self.joins:
            return self.joins[-1].sub_query.left_feature_group
        else:
            return self.left_feature_group

    @property
    def _filter(self):
        return self.get_spec(self.CONST_FILTER)

    def with_filter(self, _filter) -> "Query":
        """Sets the filter.

        Parameters
        ----------
        _filter
            The Filter.

        Returns
        -------
        Query
            The Query instance (self)
        """
        return self.set_spec(self.CONST_FILTER, _filter)

    @property
    def joins(self) -> List[Join]:
        return self.get_spec(self.CONST_JOINS)

    @joins.setter
    def joins(self, joins: List[Join]):
        self.with_joins(joins)

    def with_joins(self, joins: List[Join]) -> "Query":
        """Sets the joins.

        Parameters
        ----------
        joins: List[Join]
            The joins for the Feature Group.
        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(self.CONST_JOINS, joins)

    @property
    def entity_id(self) -> str:
        return self.get_spec(self.CONST_ENTITY_ID)

    @entity_id.setter
    def entity_id(self, value: str):
        self.with_entity_id(value)

    def with_entity_id(self, entity_id: str) -> "Query":
        """Sets the entity_id.

        Parameters
        ----------
        entity_id: str
            The entity_id.

        Returns
        -------
        Query
            The Query instance (self)
        """
        return self.set_spec(self.CONST_ENTITY_ID, entity_id)

    @property
    def feature_store_id(self) -> str:
        return self.get_spec(self.CONST_FEATURE_STORE_ID)

    @feature_store_id.setter
    def feature_store_id(self, value: str):
        self.with_entity_id(value)

    def with_feature_store_id(self, feature_store_id: str) -> "Query":
        """Sets the feature_store_id.

        Parameters
        ----------
        feature_store_id: str
            The feature_store_id.

        Returns
        -------
        Query
            The Query instance (self)
        """
        return self.set_spec(self.CONST_FEATURE_STORE_ID, feature_store_id)

    @property
    def left_features(self) -> list:
        return self.get_spec(self.CONST_LEFT_FEATURES)

    @left_features.setter
    def left_features(self, features: List[str]):
        self.with_left_features(features)

    def with_left_features(self, features: List[str]) -> "Query":
        """Sets the left feature details for the Feature Group.

        This method sets the left feature details for the Feature Group of this Query instance.
        If the input parameter 'features' is not empty, it sets the left feature details to 'features'.
        If 'features' is empty, it populates 'features' with the names of the output features of the left feature group.

        Parameters
        ----------
        features : List[str]
            The features for the Feature Group.

        Returns
        -------
        Query
            This Query instance with the left feature details set.
        """

        if not features:
            # If 'features' is empty, populate 'features' with the names of the output features of the left feature
            # group.
            features = [
                output_feature.feature_name
                for output_feature in self.left_feature_group.features
            ]

        return self.set_spec(self.CONST_LEFT_FEATURES, features)

    def read(
        self,
        is_online: Optional[bool] = False,
    ):
        """Read the specified query into a DataFrame.

        It is possible to specify the storage (online/offline) to read from and the
        type of the output DataFrame (Spark, Pandas).

        !!! warning "Engine support"
            **Spark only**

        # Arguments
            is_online: Read from online storage. Defaults to `False`.

        # Returns
            `DataFrame`: DataFrame depending on the chosen type.
        """
        sql_query = self._generate_query(is_online)
        return self.spark_engine.sql(sql_query)

    def join(
        self,
        sub_query: "Query",
        on: Optional[List[str]] = [],
        left_on: Optional[List[str]] = [],
        right_on: Optional[List[str]] = [],
        join_type: JoinType = JoinType.INNER,
    ):
        """Join Query with another Query.

        !!! example "Join multiple feature groups"
            ```python
            fg1 = FeatureGroup.fromId("...")
            fg2 = FeatureGroup.fromId("...")
            fg3 = FeatureGroup.fromId("...")

            query = fg1.select_all()
                    .join(fg3.select_all(), left_on=["location_id"], right_on=["id"], join_type=JoinType.LEFT)
            ```

        # Arguments
            sub_query:
                Right-hand side query to join.
            on:
               List of feature names to join on if they are available in both
               feature groups. Defaults to `[]`.
            left_on:
                List of feature names to join on from the left feature group of the join. Defaults to `[]`.
            right_on:
                List of feature names to join on from the right feature group of the join. Defaults to `[]`.
            join_type:
                Type of join to perform, can be `"inner"`, `"outer"`, `"left"` or `"right"`. Defaults to "inner".

        # Returns
            `Query`: A new Query object representing the join.
        """

        join = Join(sub_query, on, left_on, right_on, join_type)
        QueryValidator.validate_query_join(self.get_last_joined_feature_group(), join)
        self.joins.append(join)
        return self

    def show(self, num_rows: int = 10, is_online: Optional[bool] = False):
        """
        Display the first N rows of the Query as a table.

        Example usage:
            fg1 = FeatureGroup.from_id("...")
            fg2 = FeatureGroup.from_id("...")

            query = fg1.select()
            query.show(10)

        Args:
            num_rows (int, optional): The number of rows to display. Defaults to 10.
            is_online (bool, optional): Whether to execute the query on an online data store.
                Defaults to False.

        Returns:
            None
        """

        # Generate the SQL query
        sql_query = self._generate_query(is_online)

        # Execute the query and retrieve a pandas DataFrame
        df = self.spark_engine.sql(sql_query)

        # Display the first N rows of the DataFrame as a table
        df.show(num_rows)

    def _generate_query(self, is_online: Optional[bool] = False):
        self.__validate_nested_joins()
        from ads.feature_store.query.generator.query_generator import QueryGenerator

        return QueryGenerator(self).generate_query(is_online)

    def filter(self, f: Union[Filter, Logic]):
        """Apply filter to the feature group.

        Selects all features and returns the resulting `Query` with the applied filter.
        ```python
        query.filter(fg.feature1 == 1).show(10)
        ```

        Composite filters require parenthesis:
        ```python
        query.filter((fg.feature1 == 1) | (fg.feature2 >= 2))
        ```

        # Arguments
            f: Filter object.

        # Returns
            `Query`. The query object with the applied filter.
        """
        if self._filter is None:
            if isinstance(f, Filter):
                self.with_filter(Logic.Single(left_f=f))
            elif isinstance(f, Logic):
                self.with_filter(f)
            else:
                raise TypeError(
                    "Expected type `Filter` or `Logic`, got `{}`".format(type(f))
                )
        elif self._filter is not None:
            self.with_filter(self._filter & f)
        return self

    def to_dict(self):
        """Returns the Query as dictionary."""
        query = copy.deepcopy(self._spec)
        return query

    def to_string(self, is_online=False):
        """Generate a string representation of the object's query.

        Args:
            is_online (bool, optional): Whether to generate an online query. Defaults to False.

        Returns:
            str: A string representation of the object's query.
        """
        fs_query = self._generate_query(is_online)

        if is_online:
            raise ValueError("Online query is not supported.")
        return fs_query

    def __validate_nested_joins(self):
        for join in self.joins:
            query = join.sub_query
            if len(query.joins) > 0:
                raise ValueError("Nested Joins are not supported.")
