#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.bds.big_data_service import ADSHiveConnection
from ads.common.decorator.runtime_dependency import OptionalDependency
from typing import Dict, Optional, Union, Iterator
from pandas import DataFrame


class ConnectionFactory:
    connectionprovider = {"hive": ADSHiveConnection}

    try:
        from ads.oracledb.oracle_db import OracleRDBMSConnection

        connectionprovider["oracle"] = OracleRDBMSConnection
    except:
        pass

    try:
        from ads.mysqldb.mysql_db import MySQLRDBMSConnection

        connectionprovider["mysql"] = MySQLRDBMSConnection
    except:
        pass

    @classmethod
    def get(cls, engine="oracle"):
        Connection = cls.connectionprovider.get(engine, None)

        if not Connection:
            if engine == "mysql":
                print("Requires mysql-connector-python package to use mysql engine")
            elif engine == "oracle":
                print(
                    f"The `oracledb` or `cx_Oracle` module was not found. Please run "
                    f"`pip install {OptionalDependency.DATA}`."
                )
            raise Exception(
                f"Engine {engine} does not have either required dependency or is not supported."
            )
        return Connection


class DBAccessMixin:
    @staticmethod
    def read_sql(
        sql: str,
        connection_parameters: dict,
        bind_variables: Dict = {},
        chunksize: Optional[int] = None,
        engine="oracle",
    ) -> Union["DataFrame", Iterator["DataFrame"]]:
        """Read SQL query from oracle database into a DataFrame.

        Parameters
        ----------
        sql: str
            SQL query to be executed.
        connection_parameters: dict
            A dictionary of connection_parameters - {"user_name":"", "password":"", "service_name":"", "wallet_location":""}
        bind_variables: Optional[Dict]
            Key value of pair of bind variables and corresponding values
        chunksize: Optional[int], default None
            If specified, return an iterator where `chunksize` is the number of rows to include in each chunk.
        engine: {'oracle', 'mysql', 'hive'}, default 'oracle'
            Select the database type - MySQL/Oracle/Hive to store the data

        Returns
        -------
            DataFrame or Iterator[DataFrame]
                DataFrame or Iterator[DataFrame].

        Examples
        --------
        >>> connection_parameters = {
                "user_name": "<username>",
                "password": "<password>",
                "service_name": "{service_name}_{high|med|low}",
                "wallet_location": "/full/path/to/my_wallet.zip",
            }
        >>> import pandas as pd
        >>> import ads
        >>> df = pd.DataFrame.ads.read_sql("SELECT * from Employee", connection_parameters=connection_parameters)
        >>> df_with_bind = pd.DataFrame.ads.read_sql("SELECT * from EMPLOYEE WHERE EMPLOYEE_ID = :ID", bind_variables={"ID":"121212", connection_parameters=connection_parameters)


        """
        Connection = ConnectionFactory.get(engine)

        return Connection(**connection_parameters).query(
            sql, bind_variables=bind_variables, chunksize=chunksize
        )

    def to_sql(
        self,
        table_name: str,
        connection_parameters: dict,
        if_exists: str = "fail",
        batch_size=100000,
        engine="oracle",
        encoding="utf-8",
    ):
        """To save the dataframe df to database.

        Parameters
        ----------
        table_name: str
            Name of SQL table.
        connection_parameters: dict
            A dictionary of connection_parameters - {"user_name":"", "password":"", "service_name":"", "wallet_location":""}
        if_exists: : {'fail', 'replace', 'append'}, default 'fail'
            How to behave if the table already exists.
            * fail: Raise a ValueError. If table exists, do nothing
            * replace: Drop the table before inserting new values. If table exists, drop it, recreate it, and insert data.
            * append: Insert new values to the existing table. If table exists, insert data. Create if does not exist.
        batch_size: int, default 100000
            Inserting in batches improves insertion performance. Choose this value based on available memore and network bandwidth.
        engine: {'oracle', 'mysql'}, default 'oracle'
            Select the database type - MySQL or Oracle to store the data
        encoding: str, default is "utf-8"
            Encoding provided will be used for ecoding all columns, when inserting into table


        Returns
        -------
            None
                Nothing.
        Examples
        --------
        >>> connection_parameters = {
                "user_name": "<username>",
                "password": "<password>",
                "service_name": "{service_name}_{high|med|low}",
                "wallet_location": "/full/path/to/my_wallet.zip",
            }
        >>> import pandas as pd
        >>> import ads
        >>> df2 = pd.read_csv("my/data/csv")
        >>> df2.ads.to_sql("MY_DATA_CSV", connection_parameters=connection_parameters)
        """
        if if_exists not in ["fail", "replace", "append"]:
            raise ValueError(
                f"Unknown option `if_exists`={if_exists}. Valid options are 'fail', 'replace', 'append'"
            )

        Connection = ConnectionFactory.get(engine)
        return Connection(**connection_parameters).insert(
            table_name, self._obj, if_exists, batch_size, encoding
        )
