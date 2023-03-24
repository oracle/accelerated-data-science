#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from mock import patch
from unittest import mock
from unittest.mock import call

import pandas as pd
import pytest

from ads.oracledb.oracle_db import OracleRDBMSConnection


class TestOracle_DB:

    table_name = "TEST_TABLE_V1"
    data = {
        "col1": [1, 2],
        "col2": ["èèààòò£±", "text"],
    }
    df = pd.DataFrame(data=data)
    expected_varchar2_length = 16

    @mock.patch("oracledb.Connection.commit")
    @mock.patch("oracledb.Connection.cursor")
    def test_insert_with_if_exists_equals_replace(self, cursor, commit):
        """Test insert method with `replace` set in 'if_exists' argument.
        This test also checks that unicode text length is calculated correct in bytes."""
        with patch.object(
            OracleRDBMSConnection, "__init__", lambda a, b, c, d, e, f, g, h, i, j: None
        ):
            connection = OracleRDBMSConnection(
                None, None, None, None, None, None, None, None, None
            )
            connection.temp_dir = "temp_dir"

            connection.insert(
                table_name=self.table_name, df=self.df, if_exists="replace"
            )

            calls = [
                call().__enter__(),
                call().__enter__().execute(f"drop table {self.table_name}"),
                call()
                .__enter__()
                .execute(
                    f"create table TEST_TABLE_V1 (col1 INTEGER, col2 VARCHAR2({self.expected_varchar2_length}))"
                ),
                call()
                .__enter__()
                .executemany(
                    "insert into TEST_TABLE_V1(col1, col2) values(:col1, :col2)",
                    [(1, "èèààòò£±"), (2, "text")],
                    batcherrors=True,
                ),
            ]
            cursor.assert_has_calls(calls, any_order=False)
            connection.temp_dir = None

    @mock.patch("oracledb.Connection.cursor")
    def test_insert_with_if_exists_equals_fail(self, cursor):
        """Test insert method with `fail` set in 'if_exists' argument."""
        with patch.object(
            OracleRDBMSConnection,
            "__init__",
            lambda a, b, c, d, e, f, g, h, i, j: None,
        ):
            connection = OracleRDBMSConnection(
                None, None, None, None, None, None, None, None, None
            )
            connection.temp_dir = "temp_dir"
            with pytest.raises(ValueError):
                connection.insert(
                    table_name=self.table_name, df=self.df, if_exists="fail"
                )

                calls = [
                    call().__enter__(),
                    call()
                    .__enter__()
                    .execute("SELECT 1 from {self.table_name} FETCH NEXT 1 ROWS ONLY"),
                ]
                cursor.assert_has_calls(calls, any_order=False)
            connection.temp_dir = None

    def test_insert_with_if_exists_incorrect(self):
        """Test insert method with incorrect value for 'if_exists' argument."""
        with patch.object(
            OracleRDBMSConnection,
            "__init__",
            lambda a, b, c, d, e, f, g, h, i, j: None,
        ):
            connection = OracleRDBMSConnection(
                None, None, None, None, None, None, None, None, None
            )
            with pytest.raises(ValueError):
                connection.insert(
                    table_name=self.table_name, df=self.df, if_exists="incorrect_value"
                )
            connection.temp_dir = None
