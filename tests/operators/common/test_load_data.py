#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import pytest
from ads.opctl.operator.lowcode.common.utils import (
    load_data,
)
from ads.opctl.operator.common.operator_config import InputData
from unittest.mock import patch, Mock
import unittest


class TestDataLoad(unittest.TestCase):
    def setUp(self):
        self.data_spec = Mock(spec=InputData)
        self.data_spec.connect_args = {
            'dsn': '(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.us-ashburn-1.oraclecloud.com))(connect_data=(service_name=q9tjyjeyzhxqwla_h8posa0j7hooatry_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))',
            'wallet_password': '@Varsha1'
        }
        self.data_spec.vault_secret_id = 'ocid1.vaultsecret.oc1.iad.amaaaaaav66vvnialgpfay4ys5shd6y5nu4f2tn2e3qius2s23adzipuyhqq'
        self.data_spec.table_name = 'DF_SALARY'
        self.data_spec.url = None
        self.data_spec.format = None
        self.data_spec.columns = None
        self.data_spec.limit = None

    def testLoadSecretAndDBConnection(self):
        data = load_data(self.data_spec)
        assert len(data) == 135, f"Expected length 135, but got {len(data)}"
        expected_columns = ['CODE', 'PAY_MONTH', 'FIXED_SAL']
        assert list(
            data.columns) == expected_columns, f"Expected columns {expected_columns}, but got {list(data.columns)}"

    def testLoadVaultFailure(self):
        msg = "Vault exception message"

        def mock_load_secret(*args, **kwargs):
            raise Exception(msg)

        with patch('ads.secrets.ADBSecretKeeper.load_secret', side_effect=mock_load_secret):
            with pytest.raises(Exception) as e:
                load_data(self.data_spec)

        expected_msg = f"Could not retrieve database credentials from vault {self.data_spec.vault_secret_id}: {msg}"
        assert str(e.value) == expected_msg, f"Expected exception message '{expected_msg}', but got '{str(e)}'"

    def testDBConnectionFailure(self):
        msg = "Mocked DB connection error"

        def mock_oracledb_connect(*args, **kwargs):
            raise Exception(msg)

        with patch('oracledb.connect', side_effect=mock_oracledb_connect):
            with pytest.raises(Exception) as e:
                load_data(self.data_spec)

        assert str(e.value) == msg, f"Expected exception message '{msg}', but got '{str(e)}'"
