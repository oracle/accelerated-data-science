#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import mock
import unittest
import pytest
import pandas as pd
from ads.bds.big_data_service import ADSHiveConnection


class TestADSHiveConnection(unittest.TestCase):
    connection_parameters = {
        "host": "<host>",
        "port": "<port>",
    }

    df = pd.DataFrame(
        data={
            "str": ["a", "b", "c", "d", "e"],
            "num": [1.0, 1.1, 1.2, 1.3, 1.4],
            "int": [0, 1, 2, 2, 3],
            "bool": [True, True, False, False, False],
        }
    )

    table_name = "adsunittest"

    @mock.patch("impala.dbapi.connect")
    def test_init(self, mock_connect):
        """Test initializing a hive connection."""
        ADSHiveConnection(**self.connection_parameters)
        mock_connect.assert_called_with(
            host=self.connection_parameters["host"],
            port=self.connection_parameters["port"],
            auth_mechanism="GSSAPI",
            kerberos_service_name="hive",
        )

    @mock.patch("impala.dbapi.connect")
    def test_init_failed(self, mock_connect):
        """Test fail to initializing a hive connection because of missing host and port."""
        invalid_connection_parameters = {
            "user_name": "<username>",
            "password": "<password>",
            "service_name": "<service_name>",
        }
        with pytest.raises(TypeError):
            ADSHiveConnection(**invalid_connection_parameters)

    @mock.patch("impala.dbapi.connect")
    def test_insert_if_exists_invalid(self, mock_connect):
        """test insert if if_exists is invalid."""
        if_exists = "invalid"
        hive_connection = ADSHiveConnection(**self.connection_parameters)
        with pytest.raises(
            ValueError,
            match=f"Unknown option `if_exists`={if_exists}. Valid options are 'fail', 'replace', 'append'",
        ):
            hive_connection.insert(
                table_name=self.table_name, df=self.df, if_exists=if_exists
            )

    @mock.patch("impala.dbapi.connect")
    @mock.patch("pandas.DataFrame.to_sql", return_value=mock.MagicMock())
    @mock.patch("sqlalchemy.engine.create_engine", return_value=mock.MagicMock())
    def test_insert(self, mock_engine, mock_to_sql, mock_connect):
        if_exists = "replace"
        batch_size = 1000
        hive_connection = ADSHiveConnection(**self.connection_parameters)
        hive_connection.insert(
            table_name=self.table_name,
            df=self.df,
            if_exists=if_exists,
            batch_size=batch_size,
        )

        mock_to_sql.assert_called_with(
            name=self.table_name,
            con=mock_engine.return_value,
            if_exists=if_exists,
            index=False,
            method="multi",
            chunksize=batch_size,
        )

    @mock.patch("impala.dbapi.connect")
    def test_query(self, mock_connect):
        """test query function"""
        sql = "SELECT * FROM tableA"
        Connection = ADSHiveConnection(**self.connection_parameters)

        mock_con = mock_connect.return_value
        mock_cur = mock_con.cursor.return_value

        Connection.query(sql=sql)
        mock_cur.execute.assert_called_with(sql, None)

    @mock.patch("impala.dbapi.connect")
    def test_query_bind(self, mock_connect):
        """test query function with bind options"""
        sql = "SELECT * FROM tableA"
        Connection = ADSHiveConnection(**self.connection_parameters)
        mock_con = mock_connect.return_value
        mock_cur = mock_con.cursor.return_value

        Connection.query(sql=sql, bind_variables={})
        mock_cur.execute.assert_called_with(sql, {})

    @mock.patch("impala.dbapi.connect")
    @mock.patch("ads.bds.big_data_service.ADSHiveConnection._fetch_by_batch")
    def test_query_chunksize(self, mock_fetch, mock_connect):
        """test query function with chunksize."""
        sql = "SELECT * FROM tableA"
        Connection = ADSHiveConnection(**self.connection_parameters)
        mock_con = mock_connect.return_value
        mock_cur = mock_con.cursor.return_value
        Connection.query(sql=sql, chunksize=100)
        mock_fetch.assert_called_with(mock_cur, 100)
