#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.secrets import MySQLDBSecretKeeper
from base64 import b64encode
from collections import namedtuple
import json
import pytest
from unittest import mock
from unittest.mock import patch
import os
import yaml


credentials = namedtuple(
    "Credentials",
    ["user_name", "password", "host", "port", "database"],
)


@pytest.fixture
def key_encoding():
    user_name = "myuser"
    password = "this-is-not-the-secret"
    database = "service_high"
    host = "myhost"
    secret_dict = {
        "user_name": user_name,
        "password": password,
        "host": host,
        "port": "3306",
    }
    encoded = b64encode(json.dumps(secret_dict).encode("utf-8")).decode("utf-8")
    return (
        credentials(
            user_name=user_name,
            password=password,
            host=host,
            port="3306",
            database=None,
        ),
        secret_dict,
        encoded,
    )


@pytest.fixture
def key_encoding_with_database():
    user_name = "myuser"
    password = "this-is-not-the-secret"
    database = "mydb"
    host = "myhost"
    secret_dict = {
        "user_name": user_name,
        "password": password,
        "host": host,
        "port": "3306",
        "database": database,
    }
    encoded = b64encode(json.dumps(secret_dict).encode("utf-8")).decode("utf-8")
    return (
        credentials(
            user_name=user_name,
            password=password,
            database=database,
            host=host,
            port="3306",
        ),
        secret_dict,
        encoded,
    )


@pytest.fixture
def key_encoding_with_port():
    user_name = "myuser"
    password = "this-is-not-the-secret"
    host = "myhost"
    port = "12121212"
    database = "service_high"
    secret_dict = {
        "user_name": user_name,
        "password": password,
        "database": database,
        "host": host,
        "port": port,
    }
    encoded = b64encode(json.dumps(secret_dict).encode("utf-8")).decode("utf-8")
    return (
        credentials(
            user_name=user_name,
            password=password,
            database=database,
            host=host,
            port=port,
        ),
        secret_dict,
        encoded,
    )


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_encode(mock_client, mock_signer, key_encoding):
    mysqlsecretkeeper = MySQLDBSecretKeeper(
        *key_encoding[0],
        vault_id="ocid1.vault.oc1.<unique_ocid>",
        key_id="ocid1.key.oc1.<unique_ocid>",
        compartment_id="ocid1.compartment.oc1..<unique_ocid>",
    )

    assert mysqlsecretkeeper.encode().encoded == key_encoding[2]


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_decode(mock_client, mock_signer, key_encoding):
    mysqlsecretkeeper = MySQLDBSecretKeeper(
        None,
        vault_id="ocid1.vault.oc1.<unique_ocid>",
        key_id="ocid1.key.oc1.<unique_ocid>",
        compartment_id="ocid1.compartment.oc1..<unique_ocid>",
    )
    mysqlsecretkeeper.encoded = key_encoding[2]
    assert mysqlsecretkeeper.decode().data.user_name == key_encoding[0].user_name
    assert mysqlsecretkeeper.decode().data.password == key_encoding[0].password
    assert mysqlsecretkeeper.decode().data.database == key_encoding[0].database
    assert mysqlsecretkeeper.decode().data.host == key_encoding[0].host
    assert mysqlsecretkeeper.decode().data.port == key_encoding[0].port


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_decode_with_port(mock_client, mock_signer, key_encoding_with_port):
    mysqlsecretkeeper = MySQLDBSecretKeeper(
        None,
        vault_id="ocid1.vault.oc1.<unique_ocid>",
        key_id="ocid1.key.oc1.<unique_ocid>",
        compartment_id="ocid1.compartment.oc1..<unique_ocid>",
    )
    mysqlsecretkeeper.encoded = key_encoding_with_port[2]
    assert (
        mysqlsecretkeeper.decode().data.user_name == key_encoding_with_port[0].user_name
    )
    assert (
        mysqlsecretkeeper.decode().data.password == key_encoding_with_port[0].password
    )
    assert (
        mysqlsecretkeeper.decode().data.database == key_encoding_with_port[0].database
    )
    assert mysqlsecretkeeper.decode().data.host == key_encoding_with_port[0].host
    assert mysqlsecretkeeper.decode().data.port == key_encoding_with_port[0].port


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_decode_with_database(mock_client, mock_signer, key_encoding_with_database):
    mysqlsecretkeeper = MySQLDBSecretKeeper(
        None,
        vault_id="ocid1.vault.oc1.<unique_ocid>",
        key_id="ocid1.key.oc1.<unique_ocid>",
        compartment_id="ocid1.compartment.oc1..<unique_ocid>",
    )
    mysqlsecretkeeper.encoded = key_encoding_with_database[2]
    assert (
        mysqlsecretkeeper.decode().data.user_name
        == key_encoding_with_database[0].user_name
    )
    assert (
        mysqlsecretkeeper.decode().data.password
        == key_encoding_with_database[0].password
    )
    assert (
        mysqlsecretkeeper.decode().data.database
        == key_encoding_with_database[0].database
    )
    assert mysqlsecretkeeper.decode().data.host == key_encoding_with_database[0].host
    assert mysqlsecretkeeper.decode().data.port == key_encoding_with_database[0].port


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_mysqldb_context(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with MySQLDBSecretKeeper.load_secret(
            source="ocid1.secret.oc1.<unique_ocid>", export_env=True
        ) as mysqlsecretkeeper:
            assert mysqlsecretkeeper == {**key_encoding[1], "database": None}
            assert all(
                os.environ.get(f) == getattr(key_encoding[0], f)
                for f in key_encoding[0]._fields
            )
        assert mysqlsecretkeeper == {
            **{k: None for k in key_encoding[1]},
            "database": None,
        }
    assert all(k not in os.environ for k in key_encoding[1])


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_mysqldb_context_namespace(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with MySQLDBSecretKeeper.load_secret(
            source="ocid.secret.id", export_prefix="mydatabase", export_env=True
        ) as mysqlsecretkeeper:
            assert mysqlsecretkeeper == {**key_encoding[1], "database": None}
            assert os.environ.get("mydatabase.password") == key_encoding[0].password
        assert mysqlsecretkeeper == {
            **{k: None for k in key_encoding[1]},
            "database": None,
        }
    assert "mydatabase.password" not in os.environ


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_mysqldb_context_noexport(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with MySQLDBSecretKeeper.load_secret(
            source="ocid.secret.id",
        ) as mysqlsecretkeeper:
            assert mysqlsecretkeeper == {**key_encoding[1], "database": None}
        assert mysqlsecretkeeper == {
            **{k: None for k in key_encoding[1]},
            "database": None,
        }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_export_vault_details(mock_client, mock_signer, key_encoding, tmpdir):
    mysqlsecretkeeper = MySQLDBSecretKeeper(
        key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            mysqlsecretkeeper.encode().save("testname", "testdescription").secret_id
            == "ocid.secret.id"
        )
        mysqlsecretkeeper.export_vault_details(os.path.join(tmpdir, "test.json"))
        with open(os.path.join(tmpdir, "test.json")) as tf:
            assert json.load(tf) == {
                "key_id": "ocid.key",
                "secret_id": "ocid.secret.id",
                "vault_id": "ocid.vault",
            }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_export_vault_details(mock_client, mock_signer, key_encoding, tmpdir):
    mysqlsecretkeeper = MySQLDBSecretKeeper(
        key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            mysqlsecretkeeper.encode().save("testname", "testdescription").secret_id
            == "ocid.secret.id"
        )
        mysqlsecretkeeper.export_vault_details(
            os.path.join(tmpdir, "test.yaml"), format="yaml"
        )
        with open(os.path.join(tmpdir, "test.yaml")) as tf:
            assert yaml.load(tf, Loader=yaml.FullLoader) == {
                "key_id": "ocid.key",
                "secret_id": "ocid.secret.id",
                "vault_id": "ocid.vault",
            }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_export_vault_details_error(mock_client, mock_signer, key_encoding, tmpdir):
    mysqlsecretkeeper = MySQLDBSecretKeeper(
        key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            mysqlsecretkeeper.encode().save("testname", "testdescription").secret_id
            == "ocid.secret.id"
        )
        with pytest.raises(
            ValueError,
            match=f"Unrecognized format: paaml. Value values are - json, yaml, yml",
        ):
            mysqlsecretkeeper.export_vault_details(
                os.path.join(tmpdir, "test.yaml"), format="paaml"
            )


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_load_from_invalid_file(mock_client, mock_signer, key_encoding, tmpdir):
    with open(os.path.join(tmpdir, "test.yaml"), "w") as tf:
        yaml.dump(
            {
                "key_id": "ocid.key",
                "vault_id": "ocid.vault",
            },
            tf,
        )
    with pytest.raises(
        ValueError,
        match=f'The file: {os.path.join(tmpdir, "test.yaml")} does not contain all the required attributes - secret_id.',
    ):
        with MySQLDBSecretKeeper.load_secret(
            source=os.path.join(tmpdir, "test.yaml"), format="yaml"
        ) as mysqlsecretkeeper:
            assert mysqlsecretkeeper == key_encoding[1]
