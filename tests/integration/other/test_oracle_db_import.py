#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import os
import pytest

from tests.integration.config import secrets


@pytest.fixture
def connection_parameters():
    return {
        "user_name": secrets.other.test_oracle_username,
        "password": secrets.other.test_oracle_password
        or os.environ["test_oracle_password"],
        "dsn": secrets.other.test_oracle_dsn,
    }


@pytest.mark.cx_Oracle
def test_check_tls_cx_Oracle_warning(connection_parameters, caplog):
    with caplog.at_level(logging.DEBUG, logger="ads.oracle_connector"):
        from ads.oracledb.oracle_db import OracleRDBMSConnection

        try:
            OracleRDBMSConnection(**connection_parameters)
            print("LOgs:", caplog.record_tuples)
        except:
            pass

    # print("LOgs:", caplog.record_tuples, len(caplog.record_tuples))
    assert (
        "ads.oracle_connector",
        logging.WARNING,
        "If you are connecting to Autonomous Database in TLS, install `oracledb` python package while using the connection string from OCI Autonomous Database Console.",
    ) in caplog.record_tuples
