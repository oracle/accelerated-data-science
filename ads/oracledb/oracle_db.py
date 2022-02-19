#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ads
from ads.common.utils import ORACLE_DEFAULT_PORT
import cx_Oracle
import logging
import numpy as np
import os
import pandas as pd
import tempfile
from time import time
from typing import Dict, Optional, List, Union, Iterator
import zipfile

logger = logging.getLogger("ads.oracle_connector")


class OracleRDBMSConnection(cx_Oracle.Connection):
    def __init__(
        self,
        user_name,
        password,
        service_name=None,
        wallet_file=None,
        sid=None,
        dsn=None,
        host=None,
        port=ORACLE_DEFAULT_PORT,
        **kwargs,
    ):
        self.temp_dir = None
        self.tns_entries = {}
        kwargs["user"] = user_name
        kwargs["password"] = password
        if "wallet_location" in kwargs:
            wallet_file = kwargs.pop("wallet_location")
        if wallet_file:
            self._setup_wallet_file(wallet_file)
            if service_name:
                dsn = service_name
            if dsn in self.tns_entries:
                dsn = self.tns_entries[dsn]
                kwargs = kwargs.copy()
                kwargs["dsn"] = dsn
        elif dsn:
            kwargs["dsn"] = dsn
        elif service_name or sid:
            logger.info(f"Connecting to {host}:{port}/{service_name or sid}")
            if not host:
                raise ValueError(
                    "Missing `host` information or wallet file. To connect to the database without using wallet file, `host` infomration is required. Please set `host` or `wallet_file` with a valid value."
                )
            kwargs["dsn"] = self._construct_dsn_(host, port, service_name, sid)

        super().__init__(**kwargs)

        logger.info(
            f"cx_Oracle version: {'.'.join([str(x) for x in cx_Oracle.clientversion()[:2]])}"
        )
        logger.info(
            f"RDBMS version: {'.'.join([str(x) for x in self.version.split('.')[:2]])}"
        )

    def __del__(self):
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

    def _construct_dsn_(self, host, port, service_name=None, sid=None):
        return cx_Oracle.makedsn(host, port, sid=sid, service_name=service_name)

    def _setup_wallet_file(self, wallet_file: str):
        # extract files in wallet zip to a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        zipfile.ZipFile(wallet_file).extractall(self.temp_dir.name)

        # parse tnsnames.ora to get list of entries and modify them to include the wallet location
        fname = os.path.join(self.temp_dir.name, "tnsnames.ora")
        for line in open(fname):
            pos = line.find(" = ")
            if pos < 0:
                continue
            name = line[:pos]
            entry = line[pos + 3 :].strip()
            key_phrase = "(security="
            pos = entry.find(key_phrase) + len(key_phrase)
            wallet_entry = f"(MY_WALLET_DIRECTORY={self.temp_dir.name})"
            entry = entry[:pos] + wallet_entry + entry[pos:]
            self.tns_entries[name] = entry

    def insert(
        self, table_name: str, df: pd.DataFrame, if_exists: str, batch_size=100000
    ):

        if if_exists not in ["fail", "replace", "append"]:
            raise ValueError(
                "Unknown option `if_exists`={if_exists}. Valid options are 'fail', 'replace', 'append'"
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
                    cursor.execute(f"SELECT 1 from {table_name} FETCH NEXT 1 ROWS ONLY")
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
                "datetime64": "TIMESTAMP",
            }
            # add in any string types as Oracle's VARCHAR type setting length to accomodate longest
            def get_max_str_len(df, column):
                max_str_len = df[column].dropna().astype(str).str.len().max()
                if max_str_len >= 4000:
                    raise Exception(
                        f'Error: Column "{column}" is too long to store in a VARCHAR2 column'
                    )
                return max_str_len

            longest_string_column = max(
                (
                    get_max_str_len(df_orcl, c)
                    for c in df_orcl.select_dtypes(
                        include=["object", "category"]
                    ).columns
                ),
                default=0,
            )

            logger.debug(f"Max string column value: {longest_string_column}")

            datatypes = {
                c: f"VARCHAR2({get_max_str_len(df_orcl, c)})"
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
                    f"unable to determine oracle data type to use for column(s): {', '.join(set(df_orcl.columns)-set(datatypes.keys()))}"
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
                cursor.execute(sql)

            # insert
            bind_columns = ", ".join([f"{col}" for col in df_orcl.columns])
            bind_variables = ", ".join([f":{col}" for col in df_orcl.columns])
            sql = f"insert into {table_name}({bind_columns}) values({bind_variables})"

            logger.info(sql)

            # prevent buffer reallocations by locking in the longest string value
            # cursor.setinputsizes(None, longest_string_column)

            # replace NaN with None before turning into database records, important - don't
            # do this earlier in the logic because it can change the pandas
            # data types

            record_data = list(df_orcl.replace({np.nan: None}).itertuples(index=False))

            def chunks(lst: List, batch_size: int):
                """Yield successive batch_size chunks from lst."""
                for i in range(0, len(lst), batch_size):
                    yield lst[i : i + batch_size]

            for batch in chunks(record_data, batch_size=batch_size):

                cursor.executemany(sql, batch, batcherrors=True)

                for error in cursor.getbatcherrors():
                    logger.error(
                        f"Error: {error.message}, at row offset: {error.offset}"
                    )
                    raise RuntimeError(
                        f"Error: {error.message}, at row offset: {error.offset}"
                    )

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

        cursor = self.cursor()
        cursor.arraysize = 50000

        if chunksize:
            logger.info(f"Chunksize is {chunksize}")
            cursor.execute(sql, **bind_variables)
            columns = [row[0] for row in cursor.description]
            df = iter(
                (
                    pd.DataFrame(data=rows, columns=columns)
                    for rows in self._fetch_by_batch(cursor, chunksize)
                )
            )

        else:
            df = pd.DataFrame(
                cursor.execute(sql, **bind_variables),
                columns=[row[0] for row in cursor.description],
            )
            duration = time() - start_time
            logger.info(
                f"fetched {df.shape[0]} rows at {df.shape[0]/duration:.2f} rows/seconds"
            )

        return df
