#!/usr/bin/env python
from typing import Union

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import pytest
from ads.opctl.operator.lowcode.common.utils import (
    load_data,
)
from ads.opctl.operator.common.operator_config import InputData
from unittest.mock import patch, Mock, MagicMock
import unittest
import pandas as pd

mock_secret = {
    'user_name': 'mock_user',
    'password': 'mock_password',
    'service_name': 'mock_service_name'
}

mock_connect_args = {
    'user': 'mock_user',
    'password': 'mock_password',
    'service_name': 'mock_service_name',
    'dsn': 'mock_dsn'
}

# Mock data for testing
mock_data = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

mock_db_connection = MagicMock()

load_secret_err_msg = "Vault exception message"
db_connect_err_msg = "Mocked DB connection error"


def mock_oracledb_connect_failure(*args, **kwargs):
    raise Exception(db_connect_err_msg)


def mock_oracledb_connect(**kwargs):
    assert kwargs == mock_connect_args, f"Expected connect_args {mock_connect_args}, but got {kwargs}"
    return mock_db_connection


class MockADBSecretKeeper:
    @staticmethod
    def __enter__(*args, **kwargs):
        return mock_secret

    @staticmethod
    def __exit__(*args, **kwargs):
        pass

    @staticmethod
    def load_secret(vault_secret_id, wallet_dir):
        return MockADBSecretKeeper()

    @staticmethod
    def load_secret_fail(*args, **kwargs):
        raise Exception(load_secret_err_msg)


class TestDataLoad(unittest.TestCase):
    def setUp(self):
        self.data_spec = Mock(spec=InputData)
        self.data_spec.connect_args = {
            'dsn': 'mock_dsn'
        }
        self.data_spec.vault_secret_id = 'mock_secret_id'
        self.data_spec.table_name = 'mock_table_name'
        self.data_spec.url = None
        self.data_spec.format = None
        self.data_spec.columns = None
        self.data_spec.limit = None

    def testLoadSecretAndDBConnection(self):
        with patch('ads.secrets.ADBSecretKeeper.load_secret', side_effect=MockADBSecretKeeper.load_secret):
            with patch('oracledb.connect', side_effect=mock_oracledb_connect):
                with patch('pandas.read_sql', return_value=mock_data) as mock_read_sql:
                    data = load_data(self.data_spec)
                    mock_read_sql.assert_called_once_with(f"SELECT * FROM {self.data_spec.table_name}",
                                                          mock_db_connection)
                    pd.testing.assert_frame_equal(data, mock_data)

    def testLoadVaultFailure(self):
        with patch('ads.secrets.ADBSecretKeeper.load_secret', side_effect=MockADBSecretKeeper.load_secret_fail):
            with pytest.raises(Exception) as e:
                load_data(self.data_spec)

        expected_msg = f"Could not retrieve database credentials from vault {self.data_spec.vault_secret_id}: {load_secret_err_msg}"
        assert str(e.value) == expected_msg, f"Expected exception message '{expected_msg}', but got '{str(e)}'"

    def testDBConnectionFailure(self):
        with patch('ads.secrets.ADBSecretKeeper.load_secret', side_effect=MockADBSecretKeeper.load_secret):
            with patch('oracledb.connect', side_effect=mock_oracledb_connect_failure):
                with pytest.raises(Exception) as e:
                    load_data(self.data_spec)

        assert str(e.value) == db_connect_err_msg , f"Expected exception message '{db_connect_err_msg }', but got '{str(e)}'"
