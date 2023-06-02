#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
from typing import List

from ads.feature_store.common.enums import JoinType
from ads.jobs.builders.base import Builder


class Join(Builder):
    """
    The Join class is used to join two tables or datasets.

    Args
    ----
        query: The table or dataset to join.
        left_on: A list of column names from the left table to join on.
        right_on: A list of column names from the right table to join on.
        join_type (JoinType): The type of join to perform. Defaults to INNER join.

    Methods
    -------
        with_query(query: "Query") -> "Join": Sets the query attribute.
        with_left_on(left_on: List[str]) -> "Join": Sets the left_on attribute.
        with_right_on(right_on: List[str]) -> "Join": Sets the right_on attribute.
        with_join_type(join_type: JoinType) -> "Join": Sets the join_type attribute.
        to_dict() -> dict: Returns the Join as a dictionary.
    """

    CONST_QUERY = "query"
    CONST_LEFT_ON = "leftOn"
    CONST_RIGHT_ON = "rightOn"
    CONST_JOIN_TYPE = "joinType"
    CONST_ON = "on"

    def __init__(
        self, query, on, left_on, right_on, join_type: JoinType = JoinType.INNER
    ):
        super().__init__()
        self.with_on(on)
        self.with_sub_query(query)
        self.with_left_on(left_on)
        self.with_right_on(right_on)
        self.with_join_type(join_type)

    @property
    def sub_query(self) -> "Query":
        return self.get_spec(self.CONST_QUERY)

    @sub_query.setter
    def sub_query(self, value: "Query"):
        self.with_sub_query(value)

    def with_sub_query(self, sub_query: "Query") -> "Join":
        """Sets the query.

        Parameters
        ----------
        sub_query: Query
            The query .

        Returns
        -------
        Join
            The Join instance (self)
        """
        return self.set_spec(self.CONST_QUERY, sub_query)

    @property
    def on(self) -> "List[str]":
        return self.get_spec(self.CONST_ON)

    def with_on(self, on: List[str] = []):
        """Sets the query.

        Parameters
        ----------
        on: List[str]
            The on clause .

        Returns
        -------
        Join
            The Join instance (self)
        """
        return self.set_spec(self.CONST_ON, on)

    @property
    def left_on(self) -> list:
        return self.get_spec(self.CONST_LEFT_ON)

    @left_on.setter
    def left_on(self, features: List[str]):
        self.with_left_on(features)

    def with_left_on(self, left_on: List[str]) -> "Join":
        """Sets the left feature details.

        Parameters
        ----------
        left_on: List[str]
            The left feature to join on
        Returns
        -------
        Join
            The Join instance (self)
        """
        return self.set_spec(self.CONST_LEFT_ON, left_on)

    @property
    def right_on(self) -> list:
        return self.get_spec(self.CONST_RIGHT_ON)

    @right_on.setter
    def right_on(self, features: List[str]):
        self.with_left_on(features)

    def with_right_on(self, right_on: List[str]) -> "Join":
        """Sets the left feature details.

        Parameters
        ----------
        right_on: List[str]
            The right feature to join on
        Returns
        -------
        Join
            The Join instance (self)
        """
        return self.set_spec(self.CONST_RIGHT_ON, right_on)

    @property
    def join_type(self) -> list:
        return self.get_spec(self.CONST_JOIN_TYPE)

    @join_type.setter
    def join_type(self, join_type: JoinType):
        self.with_join_type(join_type)

    def with_join_type(self, join_type: JoinType) -> "Join":
        """Sets the left feature details.

        Parameters
        ----------
        join_type: JoinType
            The join type: Defaults to INNER
        Returns
        -------
        Join
            The Join instance (self)
        """
        return self.set_spec(self.CONST_JOIN_TYPE, join_type.value)

    def to_dict(self):
        """Returns the Join as dictionary."""
        query = copy.deepcopy(self._spec)
        return query
