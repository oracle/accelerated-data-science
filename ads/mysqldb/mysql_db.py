#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ads
import os
import tempfile
import zipfile
from time import time
from typing import Dict, Optional, List, Union, Iterator

import pandas as pd
import numpy as np

import logging

logger = logging.getLogger("ads.msql_connector")


from mysql.connector import connection

CURSOR_SIZE = 50000


class MySQLRDBMSConnection(connection.MySQLConnection):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        if kwargs["user_name"]:
            kwargs["user"] = kwargs.pop("user_name")
        super().__init__(*args, **kwargs)

    def insert(
        self,
        table_name: str,
        df: pd.DataFrame,
        if_exists: str,
        batch_size=100000,
        encoding="utf-8",
    ):

        if if_exists not in ["fail", "replace", "append"]:
            raise ValueError(
                f"Unknown option `if_exists`={if_exists}. Valid options are 'fail', 'replace', 'append'"
            )
        start_time = time()

        df_orcl = df.copy()
        # "object" type can be actual objects, or just plain strings, when inserting into the
        # database we need to stringify these so they can be represented in a VARCHAR2 column

        # object_columns = df_orcl.select_dtypes(include=["object"]).columns
        # df_orcl = df_orcl.where(pd.notnull(df_orcl), None)
        # df_orcl[object_columns] = df_orcl[object_columns].astype(str)

        # prep column names for valid Oracle column names (alpha + # $ _)
        df_orcl.columns = df_orcl.columns.str.replace(r"\W+", "_", regex=True)
        table_exist = True
        with self.cursor() as cursor:

            if if_exists != "replace":
                try:
                    cursor.execute(f"SELECT 1 from {table_name} LIMIT 1")
                    cursor.fetchall()
                except Exception as e:
                    table_exist = False
                if if_exists == "fail" and table_exist:
                    raise ValueError(
                        f"Table {table_name} already exists. Set `if_exists`='replace' or 'append' to replace or append to the existing table"
                    )
            # Oracle doesn't have boolean so convert to 1/0
            df_orcl = df_orcl.where(
                df.applymap(type) != bool, df_orcl.replace({True: 1, False: 0})
            )

            type_mappings = {
                "bool": "NUMBER(1)",
                "int16": "INTEGER",
                "int32": "INTEGER",
                "int64": "INTEGER",
                "float16": "FLOAT",
                "float32": "FLOAT",
                "float64": "FLOAT",
                "datetime64": "DATETIME",
            }
            # add in any string types as Oracle's VARCHAR type setting length to accommodate longest
            def get_max_str_len(df, column, encoding):
                return df[column].dropna().str.encode(encoding).map(len).max()

            longest_string_column = max(
                (
                    get_max_str_len(df_orcl, c, encoding)
                    for c in df_orcl.select_dtypes(
                        include=["object", "category"]
                    ).columns
                ),
                default=0,
            )

            logger.debug(f"Max string column value: {longest_string_column}")

            datatypes = {
                c: f"VARCHAR({get_max_str_len(df_orcl, c, encoding)})"
                for c in df_orcl.select_dtypes(include=["object", "category"]).columns
            }

            for df_type, orcl_type in type_mappings.items():

                datatypes.update(
                    {
                        column: orcl_type
                        for column in df_orcl.select_dtypes(include=df_type).columns
                    }
                )

            if set(datatypes.keys()) != set(df_orcl.columns):
                raise Exception(
                    f"Unable to determine MySQL data type to use for column(s): {', '.join(set(df_orcl.columns)-set(datatypes.keys()))}"
                )

            # create table
            if if_exists == "replace":
                try:
                    cursor.execute(f"drop table {table_name}")
                except:
                    logger.info(f"Table {table_name} does not exist")
            if if_exists == "replace" or not table_exist:
                sql = (
                    f"create table {table_name} ("
                    + ", ".join([f"{col} {datatypes[col]}" for col in df_orcl.columns])
                    + ")"
                )
                logger.info(sql)
                try:
                    cursor.execute(sql)
                except Exception as e:
                    raise Exception(
                        f"'{e if e.args else 'Unexpected error encountered'}' with query: '{sql}'"
                    ) from e

            # insert
            bind_columns = ", ".join([f"{col}" for col in df_orcl.columns])
            bind_variables = ",".join(["%s" for _ in df_orcl.columns])
            sql = f"insert into {table_name}({bind_columns}) values({bind_variables})"

            logger.info(sql)

            # prevent buffer reallocation by locking in the longest string value
            # cursor.setinputsizes(None, longest_string_column)

            # replace NaN with None before turning into database records, important - don't
            # do this earlier in the logic because it can change the pandas
            # data types

            record_data = list(
                (
                    [
                        xr.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
                        if hasattr(xr, "to_pydatetime")
                        else xr
                        for xr in x
                    ]
                    for x in df_orcl.replace({np.nan: None}).itertuples(
                        index=False, name=None
                    )
                )
            )

            def chunks(lst: List, batch_size: int):
                """Yield successive batch_size chunks from lst."""
                for i in range(0, len(lst), batch_size):
                    yield lst[i : i + batch_size]

            for batch in chunks(record_data, batch_size=batch_size):
                cursor.executemany(sql, batch)

            self.commit()

            duration = time() - start_time
            logger.info(
                f"inserted {df_orcl.shape[0]} rows at {df_orcl.shape[0]/duration:.2f} rows/seconds"
            )

    def _fetch_by_batch(self, cursor, chunksize):
        while True:
            rows = cursor.fetchmany(chunksize)
            if rows:
                yield rows
            else:
                break

    def query(
        self, sql: str, bind_variables: Optional[Dict], chunksize=None
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:

        start_time = time()

        cursor = self.cursor(prepared=True)
        cursor.arraysize = CURSOR_SIZE

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
