#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.secrets import AuthTokenSecretKeeper
from base64 import b64encode, b64decode
import json
import pytest
from unittest import mock
from unittest.mock import patch
import os
import yaml


@pytest.fixture
def key_encoding():
    auth_token = "a1b2c33d4e5f6g7h8i9jakblc"
    secret_dict = {"auth_token": auth_token}
    encoded = b64encode(json.dumps(secret_dict).encode("utf-8")).decode("utf-8")
    return (auth_token, secret_dict, encoded)


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_encode(mock_client, mock_signer, key_encoding):
    apisecretkeeper = AuthTokenSecretKeeper(
        key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )

    assert apisecretkeeper.encode().encoded == key_encoding[2]


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_decode(mock_client, mock_signer, key_encoding):
    apisecretkeeper = AuthTokenSecretKeeper(
        None,
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    apisecretkeeper.encoded = key_encoding[2]
    assert apisecretkeeper.decode().data.auth_token == key_encoding[0]


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_api_context(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with AuthTokenSecretKeeper.load_secret(
            source="ocid.secret.id", export_env=True
        ) as apisecretkeeper:
            assert apisecretkeeper == key_encoding[1]
            assert os.environ.get("auth_token") == key_encoding[0]
    assert apisecretkeeper == {"auth_token": None}
    assert "auth_token" not in os.environ


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_api_context_namespace(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with AuthTokenSecretKeeper.load_secret(
            source="ocid.secret.id", export_prefix="kafka", export_env=True
        ) as apisecretkeeper:
            assert apisecretkeeper == key_encoding[1]
            assert os.environ.get("kafka.auth_token") == key_encoding[0]
    assert apisecretkeeper == {"auth_token": None}
    assert "kafka.auth_token" not in os.environ


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_api_context_noexport(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with AuthTokenSecretKeeper.load_secret(
            source="ocid.secret.id",
        ) as apisecretkeeper:
            assert apisecretkeeper == key_encoding[1]
            assert "auth_token" not in os.environ
    assert apisecretkeeper == {"auth_token": None}
    assert "auth_token" not in os.environ


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_export_vault_details(mock_client, mock_signer, key_encoding, tmpdir):
    apisecretkeeper = AuthTokenSecretKeeper(
        key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            apisecretkeeper.encode().save("testname", "testdescription").secret_id
            == "ocid.secret.id"
        )
        apisecretkeeper.export_vault_details(os.path.join(tmpdir, "test.json"))
        with open(os.path.join(tmpdir, "test.json")) as tf:
            assert json.load(tf) == {
                "key_id": "ocid.key",
                "secret_id": "ocid.secret.id",
                "vault_id": "ocid.vault",
            }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_export_vault_details(mock_client, mock_signer, key_encoding, tmpdir):
    apisecretkeeper = AuthTokenSecretKeeper(
        key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            apisecretkeeper.encode().save("testname", "testdescription").secret_id
            == "ocid.secret.id"
        )
        apisecretkeeper.export_vault_details(
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
    apisecretkeeper = AuthTokenSecretKeeper(
        key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            apisecretkeeper.encode().save("testname", "testdescription").secret_id
            == "ocid.secret.id"
        )
        with pytest.raises(
            ValueError,
            match=f"Unrecognized format: paaml. Value values are - json, yaml, yml",
        ):
            apisecretkeeper.export_vault_details(
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
        with AuthTokenSecretKeeper.load_secret(
            source=os.path.join(tmpdir, "test.yaml"), format="yaml"
        ) as apisecretkeeper:
            assert apisecretkeeper == key_encoding[1]
