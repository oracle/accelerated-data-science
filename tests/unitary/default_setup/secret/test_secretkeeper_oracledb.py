#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.secrets import OracleDBSecretKeeper
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
    ["user_name", "password", "service_name", "sid", "host", "port"],
)


@pytest.fixture
def key_encoding():
    user_name = "myuser"
    password = "this-is-not-the-secret"
    service_name = "service_high"
    host = "myhost"
    secret_dict = {
        "user_name": user_name,
        "password": password,
        "host": host,
        "port": "1521",
        "service_name": service_name,
    }
    encoded = b64encode(json.dumps(secret_dict).encode("utf-8")).decode("utf-8")
    return (
        credentials(
            user_name=user_name,
            password=password,
            service_name=service_name,
            host=host,
            port="1521",
            sid=None,
        ),
        secret_dict,
        encoded,
    )


@pytest.fixture
def key_encoding_with_sid():
    user_name = "myuser"
    password = "this-is-not-the-secret"
    sid = "siddfdf"
    host = "myhost"
    secret_dict = {
        "user_name": user_name,
        "password": password,
        "host": host,
        "port": "1521",
        "sid": sid,
    }
    encoded = b64encode(json.dumps(secret_dict).encode("utf-8")).decode("utf-8")
    return (
        credentials(
            user_name=user_name,
            password=password,
            service_name=None,
            host=host,
            port="1521",
            sid=sid,
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
    service_name = "service_high"
    secret_dict = {
        "user_name": user_name,
        "password": password,
        "service_name": service_name,
        "host": host,
        "port": port,
    }
    encoded = b64encode(json.dumps(secret_dict).encode("utf-8")).decode("utf-8")
    return (
        credentials(
            user_name=user_name,
            password=password,
            service_name=service_name,
            host=host,
            port=port,
            sid=None,
        ),
        secret_dict,
        encoded,
    )


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_encode(mock_client, mock_signer, key_encoding):
    oraclesecretkeeper = OracleDBSecretKeeper(
        *key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )

    assert oraclesecretkeeper.encode().encoded == key_encoding[2]


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_decode(mock_client, mock_signer, key_encoding):
    oraclesecretkeeper = OracleDBSecretKeeper(
        None,
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    oraclesecretkeeper.encoded = key_encoding[2]
    assert oraclesecretkeeper.decode().data.user_name == key_encoding[0].user_name
    assert oraclesecretkeeper.decode().data.password == key_encoding[0].password
    assert oraclesecretkeeper.decode().data.service_name == key_encoding[0].service_name
    assert oraclesecretkeeper.decode().data.host == key_encoding[0].host
    assert oraclesecretkeeper.decode().data.port == key_encoding[0].port
    assert oraclesecretkeeper.decode().data.sid == key_encoding[0].sid


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_decode_with_port(mock_client, mock_signer, key_encoding_with_port):
    oraclesecretkeeper = OracleDBSecretKeeper(
        None,
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    oraclesecretkeeper.encoded = key_encoding_with_port[2]
    assert (
        oraclesecretkeeper.decode().data.user_name
        == key_encoding_with_port[0].user_name
    )
    assert (
        oraclesecretkeeper.decode().data.password == key_encoding_with_port[0].password
    )
    assert (
        oraclesecretkeeper.decode().data.service_name
        == key_encoding_with_port[0].service_name
    )
    assert oraclesecretkeeper.decode().data.host == key_encoding_with_port[0].host
    assert oraclesecretkeeper.decode().data.port == key_encoding_with_port[0].port
    assert oraclesecretkeeper.decode().data.sid == key_encoding_with_port[0].sid


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_decode_with_sid(mock_client, mock_signer, key_encoding_with_sid):
    oraclesecretkeeper = OracleDBSecretKeeper(
        None,
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    oraclesecretkeeper.encoded = key_encoding_with_sid[2]
    assert (
        oraclesecretkeeper.decode().data.user_name == key_encoding_with_sid[0].user_name
    )
    assert (
        oraclesecretkeeper.decode().data.password == key_encoding_with_sid[0].password
    )
    assert (
        oraclesecretkeeper.decode().data.service_name
        == key_encoding_with_sid[0].service_name
    )
    assert oraclesecretkeeper.decode().data.host == key_encoding_with_sid[0].host
    assert oraclesecretkeeper.decode().data.port == key_encoding_with_sid[0].port
    assert oraclesecretkeeper.decode().data.sid == key_encoding_with_sid[0].sid


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_oracledb_context(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with OracleDBSecretKeeper.load_secret(
            source="ocid.secret.id", export_env=True
        ) as oraclesecretkeeper:
            assert oraclesecretkeeper == {**key_encoding[1], "dsn": None, "sid": None}
            assert all(
                os.environ.get(f) == getattr(key_encoding[0], f)
                for f in key_encoding[0]._fields
            )
        assert oraclesecretkeeper == {
            **{k: None for k in key_encoding[1]},
            "dsn": None,
            "sid": None,
        }
    assert all(k not in os.environ for k in key_encoding[1])


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_oracledb_context_namespace(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with OracleDBSecretKeeper.load_secret(
            source="ocid.secret.id", export_prefix="mydatabase", export_env=True
        ) as oraclesecretkeeper:
            assert oraclesecretkeeper == {**key_encoding[1], "dsn": None, "sid": None}
            assert os.environ.get("mydatabase.password") == key_encoding[0].password
        assert oraclesecretkeeper == {
            **{k: None for k in key_encoding[1]},
            "dsn": None,
            "sid": None,
        }
    assert "mydatabase.password" not in os.environ


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_oracledb_context_noexport(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with OracleDBSecretKeeper.load_secret(
            source="ocid.secret.id",
        ) as oraclesecretkeeper:
            assert oraclesecretkeeper == {**key_encoding[1], "dsn": None, "sid": None}
        assert oraclesecretkeeper == {
            **{k: None for k in key_encoding[1]},
            "dsn": None,
            "sid": None,
        }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_export_vault_details(mock_client, mock_signer, key_encoding, tmpdir):
    oraclesecretkeeper = OracleDBSecretKeeper(
        key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            oraclesecretkeeper.encode().save("testname", "testdescription").secret_id
            == "ocid.secret.id"
        )
        oraclesecretkeeper.export_vault_details(os.path.join(tmpdir, "test.json"))
        with open(os.path.join(tmpdir, "test.json")) as tf:
            assert json.load(tf) == {
                "key_id": "ocid.key",
                "secret_id": "ocid.secret.id",
                "vault_id": "ocid.vault",
            }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_export_vault_details(mock_client, mock_signer, key_encoding, tmpdir):
    oraclesecretkeeper = OracleDBSecretKeeper(
        key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            oraclesecretkeeper.encode().save("testname", "testdescription").secret_id
            == "ocid.secret.id"
        )
        oraclesecretkeeper.export_vault_details(
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
    oraclesecretkeeper = OracleDBSecretKeeper(
        key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            oraclesecretkeeper.encode().save("testname", "testdescription").secret_id
            == "ocid.secret.id"
        )
        with pytest.raises(
            ValueError,
            match=f"Unrecognized format: paaml. Value values are - json, yaml, yml",
        ):
            oraclesecretkeeper.export_vault_details(
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
        with OracleDBSecretKeeper.load_secret(
            source=os.path.join(tmpdir, "test.yaml"), format="yaml"
        ) as oraclesecretkeeper:
            assert oraclesecretkeeper == key_encoding[1]
