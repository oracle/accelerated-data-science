#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import ABC, abstractmethod
from time import time
from typing import Dict, Iterator, List, Optional, Union

import impala
import impala.dbapi as impyla  # noqa
import pandas as pd
from impala.error import Error as ImpylaError  # noqa
from impala.error import HiveServer2Error as HS2Error  # noqa


class HiveConnection(ABC):
    """Base class Interface."""

    def __init__(self, **params):
        """set up the impala connection."""
        self.params = params
        self.con = None  # setup the connection

    @abstractmethod
    def get_cursor(self):
        """return the cursor from the connection.

        Returns
        -------
        HiveServer2Cursor:
            cursor using a specific client.
        """
        return None


class ImpylaHiveConnection(HiveConnection):
    """ImpalaHiveConnection class which uses impyla client."""

    def __init__(self, **params):
        """set up the impala connection."""
        self.params = params
        self.con = None  # setup the connection

    def get_cursor(self) -> "impala.hiveserver2.HiveServer2Cursor":
        """return the cursor from the connection.

        Returns
        -------
        impala.hiveserver2.HiveServer2Cursor:
            cursor using impyla client.
        """
        return None


class OracleHiveConnection(ImpylaHiveConnection):
    def __init__(
        self,
        host: str,
        port: str,
        **kwargs,
    ):
        """Initiate the connection.

        Parameters
        ----------
        host: str
            Hive host name.
        port: str
            Hive port.
        kwargs:
            Other connection parameters accepted by the client.
        """
        pass

    def insert(
        self,
        table_name: str,
        df: pd.DataFrame,
        if_exists: str,
        partition: List[str] = None,
    ):
        """insert a table from a pandas dataframe.

        Parameters
        ----------
        table_name (str):
            Table Name.
        df (pd.DataFrame):
            Data to be injected to the database.
        if_exists (str):
            Whether to replace, append or fail if the table already exists.
        partition (List[str], optional): Defaults to None.
            For partitioned tables, indicate the partition that's being
            inserted into, either with an ordered list of partition keys or a
            dict of partition field name to value. For example for the
            partition (year=2007, month=7), this can be either (2007, 7) or
            {'year': 2007, 'month': 7}.
        """
        if if_exists not in ["fail", "replace", "append"]:
            raise ValueError(
                "Unknown option `if_exists`={if_exists}. Valid options are 'fail', 'replace', 'append'"
            )
        pass

    def _fetch_by_batch(
        self, cursor: "impala.hiveserver2.HiveServer2Cursor", chunksize: int
    ):
        """fetch the data by batch of chunksize."""
        while True:
            rows = cursor.fetchmany(chunksize)
            if rows:
                yield rows
            else:
                break

    def query(
        self,
        sql: str,
        bind_variables: Optional[Dict] = None,
        chunksize: Optional[int] = None,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """Query data which support select statement.

        Parameters
        ----------
        sql (str):
            sql query.
        bind_variables (Optional[Dict]):
            Parameters to be bound to variables in the SQL query, if any.
            Impyla supports all DB API `paramstyle`s, including `qmark`,
            `numeric`, `named`, `format`, `pyformat`.
        chunksize (Optional[int]): . Defaults to None.
            chunksize of each of the dataframe in the iterator.

        Returns
        -------
        Union[pd.DataFrame, Iterator[pd.DataFrame]]:
            A pandas DataFrame or a pandas DataFrame iterator.
        """
        return None
