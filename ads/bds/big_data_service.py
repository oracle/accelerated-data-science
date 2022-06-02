#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from email.policy import default
import logging
from unittest.mock import DEFAULT
import numpy as np
import pandas as pd
import re

from abc import ABC, abstractmethod
from time import time
from typing import Dict, Iterator, List, Optional, Union
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)

logger = logging.getLogger("ads.hive_connector")
CURSOR_SIZE = 50000
SERVICE_NAME = "hive"
DEFAULT_BATCH_SIZE = 1000
BDS_HIVE_DEFAULT_PORT = "10000"


class HiveConnection(ABC):
    """Base class Interface."""

    def __init__(self, **params):
        """set up hive connection."""
        self.params = params
        self.con = None  # setup the connection

    @abstractmethod
    def get_cursor(self):
        """Returns the cursor from the connection.

        Returns
        -------
        HiveServer2Cursor:
            cursor using a specific client.
        """
        pass

    @abstractmethod
    def get_engine(self):
        """Returns engine from the connection.

        Returns
        -------
        Engine object for the connection.
        """
        pass


class ImpylaHiveConnection(HiveConnection):
    """ImpalaHiveConnection class which uses impyla client."""

    @runtime_dependency(module="impala", install_from=OptionalDependency.BDS)
    def __init__(self, **params):
        """set up the impala connection."""
        from impala.dbapi import connect

        self.params = params
        self.con = connect(**self.params)

    def get_cursor(self) -> "impala.hiveserver2.HiveServer2Cursor":
        """Returns the cursor from the connection.

        Returns
        -------
        impala.hiveserver2.HiveServer2Cursor:
            cursor using impyla client.
        """
        return self.con.cursor()

    @runtime_dependency(module="sqlalchemy", install_from=OptionalDependency.BDS)
    def get_engine(self, schema="default"):
        """return the sqlalchemy engine from the connection.

        Parameters
        ----------
        schema: str
            Default to "default". The default schema used for query.

        Returns
        -------
        sqlalchemy.engine:
            engine using a specific client.
        """
        from sqlalchemy.engine import create_engine

        logger.info(
            f'Creating sqlalchemy engine with: hive://{self.params["host"]}:{self.params["port"]}/{schema}'
        )
        return create_engine(
            f'hive://{self.params["host"]}:{self.params["port"]}/{schema}'
        )


class HiveConnectionFactory:
    clientprovider = {
        "impyla": ImpylaHiveConnection,
    }

    @classmethod
    def get(cls, driver="impyla"):
        return cls.clientprovider.get(driver)


class ADSHiveConnection:
    def __init__(
        self,
        host: str,
        port: str = BDS_HIVE_DEFAULT_PORT,
        auth_mechanism: str = "GSSAPI",
        driver: str = "impyla",
        **kwargs,
    ):
        """Initiate the connection.

        Parameters
        ----------
        host: str
            Hive host name.
        port: str
            Hive port. Default to 10000.
        auth_mechanism: str
            Default to "GSSAPI". Using "PLAIN" for unsecure cluster.
        driver: str
            Default to "impyla". Client used to communicate with Hive. Only support impyla by far.
        kwargs:
            Other connection parameters accepted by the client.
        """
        kwargs["host"] = host
        kwargs["port"] = port
        kwargs["kerberos_service_name"] = SERVICE_NAME
        kwargs["auth_mechanism"] = auth_mechanism

        Connection = HiveConnectionFactory.get(driver)
        if not Connection:
            raise Exception(
                f"Driver {driver} does not have either required dependency or is not supported."
            )
        self.connection = Connection(**kwargs)

    def insert(
        self,
        table_name: str,
        df: pd.DataFrame,
        if_exists: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs,
    ):
        """insert a table from a pandas dataframe.

        Parameters
        ----------
        table_name (str):
            Table Name. Table name contains database name as well. By default it will use 'default' database.
            You can specify the database name by `table_name=<db_name>.<tb_name>`.
        df (pd.DataFrame):
            Data to be injected to the database.
        if_exists (str):
            Whether to replace, append or fail if the table already exists.
        batch_size: int, default 1000
            Inserting in batches improves insertion performance. Choose this value based on available memory and network bandwidth.
        kwargs (dict):
            Other parameters used by pandas.DataFrame.to_sql.
        """
        if if_exists.lower() not in ("fail", "replace", "append"):
            raise ValueError(
                f"Unknown option `if_exists`={if_exists}. Valid options are 'fail', 'replace', 'append'."
            )

        schema = kwargs.pop("schema") if "schema" in kwargs else "default"
        engine = self.connection.get_engine(schema)
        try:
            df.to_sql(
                name=table_name.lower(),
                con=engine,
                if_exists=if_exists,
                index=False,
                method="multi",
                chunksize=batch_size,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Runtime Error: {e.args[0]}")

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
    ) -> Union["pd.DataFrame", Iterator["pd.DataFrame"]]:
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
        start_time = time()

        cursor = self.connection.get_cursor()

        cursor.set_arraysize(arraysize=CURSOR_SIZE)
        cursor.execute(sql, bind_variables)
        columns = [row[0] for row in cursor.description]

        if chunksize:
            logger.info(f"Chunksize is {chunksize}")
            df = iter(
                (
                    pd.DataFrame(data=rows, columns=columns)
                    for rows in self._fetch_by_batch(cursor, chunksize)
                )
            )

        else:
            df = pd.DataFrame(
                cursor,
                columns=columns,
            )
            duration = time() - start_time
            logger.info(
                f"fetched {df.shape[0]} rows at {df.shape[0]/duration:.2f} rows/seconds"
            )

        return df
