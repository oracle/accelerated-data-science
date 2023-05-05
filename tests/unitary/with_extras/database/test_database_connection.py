#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from zipfile import ZipFile

from ads.database import connection
from ads.database.connection import Connector
from ads.database.connection import OracleConnector
from ads.vault.vault import Vault
from collections import namedtuple
from mock import MagicMock, patch, Mock
from oci.exceptions import ServiceError
import oci
import os
import pytest
import sys


class TestDatabaseConnection:
    """Contains test cases for ads.database.connection.py"""

    local_credential = {
        "database_name": "datamart",
        "username": "Testing",
        "password": "This-is-not-real-psw",
        "database_type": "oracle",
    }
    update_credential = {"k": "v"}
    key = "Testing"
    default_repository_path = os.path.join(os.path.expanduser("~"), ".database")
    fake_repo_path = "/tmp/nonexistent_path"
    secret_id = "ocid1.vaultsecret.oc1.iad.<unique_ocid>"
    wallet_path = "test_wallet_path.zip"
    invalid_keys = ["/testing", ".testing", "../"]

    def test_update_repository_with_replace_set_to_true_1(self):
        """Test saving value into local database store."""
        res = connection.update_repository(key=self.key, value=self.local_credential)
        assert res == self.local_credential

    def test_update_repository_with_replace_set_to_true_2(self):
        """Test saving value into local database store."""
        res = connection.update_repository(key=self.key, value=self.update_credential)
        assert res == self.update_credential

    def test_update_repository_with_replace_set_to_false(self):
        """Test saving value into local database store."""
        res = connection.update_repository(
            key=self.key, value=self.update_credential, replace=False
        )
        assert res == self.update_credential

    def test_update_repository_with_invalid_key(self):
        """Test saving value into local database store provided with invalid key."""
        for key in self.invalid_keys:
            with pytest.raises(ValueError) as execinfo:
                connection.update_repository(key=key, value=self.update_credential)
            assert str(execinfo.value) == f"{key} is not a valid directory name."

    def test_get_repository(self):
        """Test getting all values from local database store."""
        res = connection.get_repository(key=self.key)
        assert res == self.update_credential

    def test_get_repository_with_nonexistent_repository_path(self):
        """Test getting all values from local database store when repository_path does not exist."""
        with pytest.raises(ValueError) as execinfo:
            connection.get_repository(key=self.key, repository_path=self.fake_repo_path)
        assert str(execinfo.value) == f"{self.fake_repo_path} does not exist."

    def test_get_repository_with_nonexistent_db_path(self):
        """Test getting all values from local database store when db_path does not exist."""
        fake_db_path = os.path.join(self.default_repository_path, "fake_testing")
        with pytest.raises(ValueError) as execinfo:
            connection.get_repository(key="fake_testing")
        assert str(execinfo.value) == f"{fake_db_path} does not exist."

    def test_get_repository_with_invalid_key(self):
        """Test getting all values from local database store with invalid key."""
        for key in self.invalid_keys:
            with pytest.raises(ValueError) as execinfo:
                connection.get_repository(key=key)
            assert str(execinfo.value) == f"{key} is not a valid directory name."

    def test_import_wallet(self):
        """Test saving wallet to local store."""
        db_path = os.path.join(self.default_repository_path, self.key)
        with ZipFile(self.wallet_path, "w") as zip_object:
            # Adding files that need to be zipped
            zip_object.writestr("sqlnet.ora", data="test_data")
            zip_object.writestr("config.json", data='{"database_type": "oracle"}')
        connection.import_wallet(wallet_path=self.wallet_path, key=self.key)
        assert os.environ.get("TNS_ADMIN") == db_path
        assert os.path.exists(os.path.join(db_path, "config.json"))
        if os.path.exists(self.wallet_path):
            os.remove(self.wallet_path)


    def test_import_wallet_with_invalid_key(self):
        """Test saving wallet to local store with invalid key."""
        for key in self.invalid_keys:
            with pytest.raises(ValueError) as execinfo:
                connection.import_wallet(wallet_path=self.wallet_path, key=key)
            assert str(execinfo.value) == f"{key} is not a valid directory name."

    def test_import_wallet_with_nonexistent_wallet_path(self):
        """Test saving wallet to local store when wallet path does not exist."""
        fake_wallet_path = "/tmp/fake_wallet_db.zip"
        with pytest.raises(ValueError) as execinfo:
            connection.import_wallet(wallet_path=fake_wallet_path, key=self.key)
        assert str(execinfo.value) == f"{fake_wallet_path} does not exist."

    def test_import_wallet_with_nonexistent_db_path(self):
        """Test saving wallet to local store when db_path does not exist."""
        fake_db_path = os.path.join(self.default_repository_path, "fake_testing")
        with pytest.raises(ValueError) as execinfo:
            connection.import_wallet(wallet_path=self.wallet_path, key="fake_testing")
        assert str(execinfo.value) == f"{fake_db_path} does not exist."

    def test_connector_with_sqlalchemy_uninstalled(self):
        """Test making connection with given local dir.

        Mock sqlalchemy.create_engine() to avoid connect to real database.
        """
        with patch.dict(sys.modules, {"sqlalchemy": None}):
            with pytest.raises(ModuleNotFoundError):
                connection.update_repository(key=self.key, value=self.local_credential)
                connector = Connector(key=self.key)

    def test_connector_with_cx_oracle_uninstalled(self):
        """Test making connection with given local dir.

        Mock sqlalchemy.create_engine() to avoid connect to real database.
        """
        with patch.dict(sys.modules, {"cx_Oracle": None}):
            with pytest.raises(ModuleNotFoundError):
                connection.update_repository(key=self.key, value=self.local_credential)
                connector = Connector(key=self.key)

    @patch("sqlalchemy.create_engine")
    def test_connector_with_local_dir(self, mock_create_engine):
        """Test making connection with given local dir.

        Mock sqlalchemy.create_engine() to avoid connect to real database.
        """
        connection.update_repository(key=self.key, value=self.local_credential)
        connector = Connector(key=self.key)
        assert connector.config == self.local_credential
        assert connector.uri.split("@")[1] == self.local_credential["database_name"]
        assert (
            connector.uri.split("//")[1].split(":")[0]
            == self.local_credential["username"]
        )
        assert (
            connector.uri.split(":")[2].split("@")[0]
            == self.local_credential["password"]
        )
        assert connector.uri.split("+")[0] == self.local_credential["database_type"]

    def test_connector_with_nonexistent_local_dir(self):
        """Test making connection with given nonexistent local dir."""
        connection.update_repository(key=self.key, value=self.local_credential)
        with pytest.raises(ValueError) as execinfo:
            connector = Connector(key=self.key, repository_path=self.fake_repo_path)
        assert str(execinfo.value) == f"{self.fake_repo_path} does not exist."

    @patch.object(connection, "OracleConnector")
    def test_connector_with_command_line(self, mock_sqlalchemy):
        """Test making connection with command line.

        Mock sqlalchemy.create_engine() to avoid connect to real database.
        """
        connector = Connector(
            username="test_admin",
            password="test_pwd",
            database_name="TestDB",
            database_type="oracle",
        )
        mock_sqlalchemy.assert_called_once()
        assert connector.config == {
            "username": "test_admin",
            "password": "test_pwd",
            "database_name": "TestDB",
            "database_type": "oracle",
        }

    def test_OracleConnector_init_missing_valid_keys(self):
        """Test connecting oracle database."""
        oracle_connection_config = {
            "database_name": "datamart",
            "database_type": "oracle",
            "username": "Testing",
        }
        with pytest.raises(ValueError) as execinfo:
            oracle_connector = OracleConnector(oracle_connection_config)
        assert str(execinfo.value) == "password is a required parameter to connect."

    def test_connector_with_vault_secret_id_is_invalid(self):
        """Test making connection with vault, but secret_id_is_invalid."""
        invalid_secret_id = "aaaaaaaaaaaaaaaaaa"
        with pytest.raises(ValueError) as execinfo:
            connector = Connector(secret_id=invalid_secret_id)
        assert str(execinfo.value) == f"{invalid_secret_id} is not a valid secret id."

    @patch("oci.secrets.SecretsClient.get_secret_bundle")
    def test_connector_with_vault_raise_exception(self, mock_get_secret_bundle):
        """Fail to retrieve from Oracle Cloud Infrastructure Vault."""
        mock_get_secret_bundle.side_effect = ServiceError(
            status=404, code=None, headers={}, message="error test msg"
        )
        with pytest.raises(ServiceError):
            connector = Connector(secret_id=self.secret_id)

    @patch("oci.secrets.SecretsClient.get_secret_bundle")
    @patch.object(connection, "OracleConnector")
    def test_connector_with_vault(self, mock_sqlalchemy, mock_get_secret_bundle):
        """Test making connection with vault."""

        def _generate_get_secret_bundle_data(secret_id, secret_bundle_content):
            entity_item = {
                "secret_bundle_content": secret_bundle_content,
                "secret_id": secret_id,
                "version_number": 1,
            }
            response = oci.secrets.models.SecretBundle(**entity_item)
            return response

        def _get_secret_bundle_content(content):
            entity_item = {
                "content": content,
                "content_type": "BASE64",
            }
            response = oci.secrets.models.Base64SecretBundleContentDetails(
                **entity_item
            )
            return response

        test_config = {
            "database_name": "db201910031555_high",
            "database_type": "oracle",
            "password": "random_password",
            "username": "admin",
        }

        # build secret_bundle_content
        encode = Vault._dict_to_secret(test_config)
        secret_content = _get_secret_bundle_content(content=encode)

        # build get_secret_bundle response
        wrapper = namedtuple("wrapper", ["data"])
        secret_bundle = _generate_get_secret_bundle_data(
            secret_id=self.secret_id, secret_bundle_content=secret_content
        )
        client_get_secret_bundle_response = wrapper(data=secret_bundle)
        mock_get_secret_bundle.return_value = client_get_secret_bundle_response

        connector = Connector(secret_id=self.secret_id)
        mock_sqlalchemy.assert_called_once()
        assert connector.config == test_config

    def test_connector_raise_exception(self):
        """Test making connection with exception raising."""
        # invalid key
        for key in self.invalid_keys:
            with pytest.raises(ValueError) as execinfo:
                connector = Connector(key=key)
            assert str(execinfo.value) == f"{key} is not a valid directory name."

        # nonexistent db path
        fake_db_path = os.path.join(self.default_repository_path, "fake_testing")
        with pytest.raises(ValueError) as execinfo:
            connector = Connector(key="fake_testing")
        assert str(execinfo.value) == f"{fake_db_path} does not exist."

        # missing database_type in config
        valid_database_types = ["oracle"]
        credential_without_database_types = {
            "database_name": "db201910031555_high",
            "password": "random_password",
            "username": "admin",
        }
        connection.update_repository(
            key=self.key, value=credential_without_database_types
        )
        with pytest.raises(ValueError) as execinfo:
            connector = Connector(key=self.key)
        assert (
            str(execinfo.value) == f"The database_type needs to be specified. "
            f"Valid database types are {valid_database_types}"
        )

        # invalid database_type
        credential_with_invalid_database_type = {
            "database_name": "db201910031555_high",
            "database_type": "mysql",
            "password": "random_password",
            "username": "admin",
        }
        connection.update_repository(
            key=self.key, value=credential_with_invalid_database_type
        )
        with pytest.raises(ValueError) as execinfo:
            connector = Connector(key=self.key)
        assert (
            str(execinfo.value)
            == f"{credential_with_invalid_database_type['database_type']} is not a valid database type. "
            f"Valid database types are {valid_database_types}"
        )
