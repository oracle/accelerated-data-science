#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.secrets import ADBSecretKeeper
from base64 import b64encode
import json
import pytest
import os
import zipfile
from unittest import mock
from unittest.mock import patch
import hashlib
from collections import namedtuple
import yaml


@pytest.fixture
def key_encoding():
    user_name = "myuser"
    password = "this-is-not-the-secret"
    service_name = "service_high"
    wallet_location = "/this/is/wallet.zip"
    secret_dict = {
        "user_name": user_name,
        "password": password,
        "service_name": service_name,
    }
    encoded = b64encode(json.dumps(secret_dict).encode("utf-8")).decode("utf-8")
    return (
        (user_name, password, service_name, wallet_location),
        secret_dict,
        encoded,
        wallet_location,
    )


def generate_wallet_data(wallet_zip_path, wallet_dir_path):
    files = 4
    file_content = {}
    file_encoded = {}
    file_secret_ids = []
    content_map = {}
    filenames = []
    for i in range(files):
        file_name = os.path.join(wallet_dir_path, f"{i}.sample")
        filenames.append(file_name)
        with open(file_name, "wb") as outfile:
            content = f"file content {i}".encode("utf-8")
            outfile.write(content)
            file_content[os.path.basename(file_name)] = content
            file_encoded[os.path.basename(file_name)] = b64encode(content).decode(
                "utf-8"
            )
            file_secret_ids.append(f"ocid1.secret.oc1.<unique_ocid>{i}")
            content_map[f"ocid1.secret.oc1.<unique_ocid>{i}"] = b64encode(
                json.dumps(
                    {
                        "filename": os.path.basename(file_name),
                        "content": file_encoded[os.path.basename(file_name)],
                    }
                ).encode("utf-8")
            ).decode("utf-8")

    with zipfile.ZipFile(wallet_zip_path, "w") as wzipf:
        for file in filenames:
            wzipf.write(file, os.path.basename(file))
    return file_content, file_encoded, filenames, file_secret_ids, content_map


@pytest.fixture
def key_encoding_with_wallet(tmpdir):
    user_name = "myuser"
    password = "this-is-not-the-secret"
    service_name = "service_high"
    wallet_file_name = "wallet.zip"
    wallet_dir = os.path.join(tmpdir, "wallet")
    os.makedirs(wallet_dir)
    wallet_location = os.path.join(wallet_dir, wallet_file_name)

    wallet_details = generate_wallet_data(wallet_location, wallet_dir)

    secret_dict = {
        "user_name": user_name,
        "password": password,
        "service_name": service_name,
        "wallet_file_name": wallet_file_name,
    }

    encoded_secret_data = {
        **secret_dict,
        "wallet_content": wallet_details[1],
    }
    saved_secret_data = {**secret_dict, "wallet_secret_ids": wallet_details[3]}

    encoded = b64encode(json.dumps(saved_secret_data).encode("utf-8")).decode("utf-8")

    credentials = namedtuple(
        "Credentials", ["user_name", "password", "service_name", "wallet_location"]
    )
    testdata = namedtuple(
        "TestData",
        [
            "credentials",
            "encoded_secret_data",
            "encoded",
            "wallet_location",
            "filecontent_secret_map",
        ],
    )
    return testdata(
        credentials(user_name, password, service_name, wallet_location),
        encoded_secret_data,
        encoded,
        wallet_location,
        wallet_details[4],
    )


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_encode(mock_client, mock_signer, key_encoding):
    adwsecretkeeper = ADBSecretKeeper(
        *key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )

    assert adwsecretkeeper.encode(serialize_wallet=False).encoded == key_encoding[2]


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_save(mock_client, mock_signer, key_encoding, tmpdir):
    adwsecretkeeper = ADBSecretKeeper(
        *key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            adwsecretkeeper.encode().save("testname", "testdescription").secret_id
            == "ocid.secret.id"
        )
        adwsecretkeeper.export_vault_details(os.path.join(tmpdir, "test.json"))
        with open(os.path.join(tmpdir, "test.json")) as tf:
            assert json.load(tf) == {
                "key_id": "ocid.key",
                "secret_id": "ocid.secret.id",
                "vault_id": "ocid.vault",
            }
        adwsecretkeeper.export_vault_details(os.path.join(tmpdir, "test.yaml"))
        with open(os.path.join(tmpdir, "test.yaml")) as tf:
            assert yaml.load(tf, Loader=yaml.FullLoader) == {
                "key_id": "ocid.key",
                "secret_id": "ocid.secret.id",
                "vault_id": "ocid.vault",
            }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_save_without_explicit_encoding(
    mock_client, mock_signer, key_encoding, tmpdir
):
    adwsecretkeeper = ADBSecretKeeper(
        *key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            adwsecretkeeper.save("testname", "testdescription").secret_id
            == "ocid.secret.id"
        )
        adwsecretkeeper.export_vault_details(os.path.join(tmpdir, "test.json"))
        with open(os.path.join(tmpdir, "test.json")) as tf:
            assert json.load(tf) == {
                "key_id": "ocid.key",
                "secret_id": "ocid.secret.id",
                "vault_id": "ocid.vault",
            }
        adwsecretkeeper.export_vault_details(os.path.join(tmpdir, "test.yaml"))
        with open(os.path.join(tmpdir, "test.yaml")) as tf:
            assert yaml.load(tf, Loader=yaml.FullLoader) == {
                "key_id": "ocid.key",
                "secret_id": "ocid.secret.id",
                "vault_id": "ocid.vault",
            }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_context(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with ADBSecretKeeper.load_secret(
            source="ocid.secret.id",
            wallet_location="/this/is/mywallet.zip",
            export_env=True,
        ) as adwsecretkeeper:
            assert adwsecretkeeper == {
                **key_encoding[1],
                "wallet_location": "/this/is/mywallet.zip",
            }
            assert os.environ.get("user_name") == key_encoding[0][0]
            assert os.environ.get("password") == key_encoding[0][1]
            assert os.environ.get("service_name") == key_encoding[0][2]
            assert os.environ.get("wallet_location") == "/this/is/mywallet.zip"
        assert adwsecretkeeper == {
            "user_name": None,
            "password": None,
            "service_name": None,
            "wallet_location": None,
        }
        assert os.environ.get("user_name") is None
        assert os.environ.get("password") is None
        assert os.environ.get("service_name") is None
        assert os.environ.get("wallet_location") is None


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_keeper_no_wallet(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with ADBSecretKeeper.load_secret(
            source="ocid.secret.id",
        ) as adwsecretkeeper:
            assert adwsecretkeeper == {
                **key_encoding[1],
                "wallet_location": None,
            }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_keeper_with_repository(mock_client, mock_signer, key_encoding, tmpdir):
    expected = {**key_encoding[1], "wallet_location": key_encoding[3]}
    os.makedirs(os.path.join(tmpdir, "testdb"))
    with open(os.path.join(tmpdir, "testdb", "config.json"), "w") as conffile:
        json.dump(expected, conffile)

    adwsecretkeeper = ADBSecretKeeper(repository_path=tmpdir, repository_key="testdb")
    assert adwsecretkeeper.to_dict() == expected


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_context_namespace(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with ADBSecretKeeper.load_secret(
            source="ocid.secret.id",
            wallet_location="/this/is/mywallet.zip",
            export_prefix="myapp",
            export_env=True,
        ) as adwsecretkeeper:
            assert adwsecretkeeper == {
                **key_encoding[1],
                "wallet_location": "/this/is/mywallet.zip",
            }
            assert os.environ.get("myapp.user_name") == key_encoding[0][0]
            assert os.environ.get("myapp.password") == key_encoding[0][1]
            assert os.environ.get("myapp.service_name") == key_encoding[0][2]
            assert os.environ.get("myapp.wallet_location") == "/this/is/mywallet.zip"
        assert adwsecretkeeper == {
            "user_name": None,
            "password": None,
            "service_name": None,
            "wallet_location": None,
        }
        assert os.environ.get("myapp.user_name") is None
        assert os.environ.get("myapp.password") is None
        assert os.environ.get("myapp.service_name") is None
        assert os.environ.get("myapp.wallet_location") is None


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_context_noexport(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with ADBSecretKeeper.load_secret(
            source="ocid.secret.id",
            wallet_location="/this/is/mywallet.zip",
        ) as adwsecretkeeper:
            assert adwsecretkeeper == {
                **key_encoding[1],
                "wallet_location": "/this/is/mywallet.zip",
            }

            assert os.environ.get("user_name") is None
            assert os.environ.get("password") is None
            assert os.environ.get("service_name") is None
            assert os.environ.get("wallet_location") is None

        assert adwsecretkeeper == {
            "user_name": None,
            "password": None,
            "service_name": None,
            "wallet_location": None,
        }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_decode(mock_client, mock_signer, key_encoding):
    adwsecretkeeper = ADBSecretKeeper(
        *key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )

    adwsecretkeeper.encoded = key_encoding[2]
    decoded = adwsecretkeeper.decode()

    assert decoded.data.user_name == key_encoding[0][0]
    assert decoded.data.password == key_encoding[0][1]
    assert decoded.data.service_name == key_encoding[0][2]

    assert decoded.data.to_dict() == {
        **key_encoding[1],
        "wallet_location": key_encoding[0][3],
    }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_with_wallet_storage_encode(
    mock_client, mock_signer, key_encoding_with_wallet
):
    adwsecretkeeper = ADBSecretKeeper(
        *key_encoding_with_wallet.credentials,
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )

    assert (
        adwsecretkeeper.encode(serialize_wallet=True).encoded
        == key_encoding_with_wallet.encoded_secret_data
    )


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_with_wallet_storage_encode_value_error(
    mock_client, mock_signer, key_encoding_with_wallet
):
    with pytest.raises(
        ValueError,
        match=f"Missing path to wallet zip file. Required wallet zip file path to be set in `wallet_location` ",
    ):
        adwsecretkeeper = ADBSecretKeeper(
            "user_name",
            "dummy",
            "dummy",
            vault_id="ocid.vault",
            key_id="ocid.key",
            compartment_id="dummy",
        )
        adwsecretkeeper.encode(serialize_wallet=True)


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_with_wallet_storage_decode(
    mock_client, mock_signer, key_encoding_with_wallet, tmpdir
):
    wallet_dir = os.path.join(tmpdir, "test_wallet_dir")
    os.makedirs(wallet_dir)

    def mock_get_secret_id(
        id, decoded, content_map=key_encoding_with_wallet.filecontent_secret_map
    ):
        return content_map[id]

    with mock.patch(
        "ads.vault.Vault.get_secret", side_effect=mock_get_secret_id
    ) as mocked_getsecret:
        adwsecretkeeper = ADBSecretKeeper(
            vault_id="ocid.vault",
            key_id="ocid.key",
            compartment_id="dummy",
            wallet_dir=wallet_dir,
        )
        adwsecretkeeper.encoded = key_encoding_with_wallet.encoded

        decoded = adwsecretkeeper.decode()
        assert decoded.data.user_name == key_encoding_with_wallet.credentials.user_name
        assert decoded.data.password == key_encoding_with_wallet.credentials.password
        assert (
            decoded.data.service_name
            == key_encoding_with_wallet.credentials.service_name
        )

        assert adwsecretkeeper.to_dict() == {
            "user_name": key_encoding_with_wallet.credentials.user_name,
            "password": key_encoding_with_wallet.credentials.password,
            "service_name": key_encoding_with_wallet.credentials.service_name,
            "wallet_location": f"{os.path.join(wallet_dir,'wallet.zip')}",
        }

        # with open(key_encoding_with_wallet[3], "rb") as orgfile:
        #     with open(f"{os.path.join(wallet_dir,'wallet.zip')}", "rb") as newfile:
        #         assert (
        #             hashlib.md5(orgfile.read()).hexdigest()
        #             == hashlib.md5(newfile.read()).hexdigest()
        #         )


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_with_wallet_storage_context_manager(
    mock_client, mock_signer, key_encoding_with_wallet, tmpdir
):
    wallet_dir = os.path.join(tmpdir, "test_wallet_dir")
    os.makedirs(wallet_dir)
    content_map = key_encoding_with_wallet[4]
    content_map["meta.secret.id"] = key_encoding_with_wallet.encoded

    def mock_get_secret_id(
        id, decoded, content_map=key_encoding_with_wallet.filecontent_secret_map
    ):
        return content_map[id]

    with mock.patch(
        "ads.vault.Vault.get_secret", side_effect=mock_get_secret_id
    ) as mocked_getsecret:
        with ADBSecretKeeper.load_secret(
            wallet_dir=wallet_dir, source="meta.secret.id", export_env=True
        ) as adwsecretkeeper:
            assert adwsecretkeeper == {
                "user_name": key_encoding_with_wallet.credentials.user_name,
                "password": key_encoding_with_wallet.credentials.password,
                "service_name": key_encoding_with_wallet.credentials.service_name,
                "wallet_location": f"{os.path.join(wallet_dir,'wallet.zip')}",
            }
            assert (
                os.environ.get("user_name")
                == key_encoding_with_wallet.credentials.user_name
            )
            assert (
                os.environ.get("password")
                == key_encoding_with_wallet.credentials.password
            )
            assert (
                os.environ.get("service_name")
                == key_encoding_with_wallet.credentials.service_name
            )
            assert (
                os.environ.get("wallet_location")
                == f"{os.path.join(wallet_dir,'wallet.zip')}"
            )

        assert adwsecretkeeper == {
            "user_name": None,
            "password": None,
            "service_name": None,
            "wallet_location": None,
        }
        assert os.environ.get("user_name") is None
        assert os.environ.get("password") is None
        assert os.environ.get("service_name") is None
        assert os.environ.get("wallet_location") is None


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_with_wallet_storage_context_manager_namespace(
    mock_client, mock_signer, key_encoding_with_wallet, tmpdir
):
    wallet_dir = os.path.join(tmpdir, "test_wallet_dir")
    os.makedirs(wallet_dir)
    content_map = key_encoding_with_wallet[4]
    content_map["meta.secret.id"] = key_encoding_with_wallet.encoded

    def mock_get_secret_id(
        id, decoded, content_map=key_encoding_with_wallet.filecontent_secret_map
    ):
        return content_map[id]

    with mock.patch(
        "ads.vault.Vault.get_secret", side_effect=mock_get_secret_id
    ) as mocked_getsecret:
        with ADBSecretKeeper.load_secret(
            wallet_dir=wallet_dir,
            source="meta.secret.id",
            export_prefix="myapp",
            export_env=True,
        ) as adwsecretkeeper:
            assert adwsecretkeeper == {
                "user_name": key_encoding_with_wallet.credentials.user_name,
                "password": key_encoding_with_wallet.credentials.password,
                "service_name": key_encoding_with_wallet.credentials.service_name,
                "wallet_location": f"{os.path.join(wallet_dir,'wallet.zip')}",
            }
            assert (
                os.environ.get("myapp.user_name")
                == key_encoding_with_wallet.credentials.user_name
            )
            assert (
                os.environ.get("myapp.password")
                == key_encoding_with_wallet.credentials.password
            )
            assert (
                os.environ.get("myapp.service_name")
                == key_encoding_with_wallet.credentials.service_name
            )
            assert (
                os.environ.get("myapp.wallet_location")
                == f"{os.path.join(wallet_dir,'wallet.zip')}"
            )

        assert adwsecretkeeper == {
            "user_name": None,
            "password": None,
            "service_name": None,
            "wallet_location": None,
        }
        assert os.environ.get("myapp.user_name") is None
        assert os.environ.get("myapp.password") is None
        assert os.environ.get("myapp.service_name") is None
        assert os.environ.get("myapp.wallet_location") is None


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_with_wallet_storage_context_manager_noexport(
    mock_client, mock_signer, key_encoding_with_wallet, tmpdir
):
    wallet_dir = os.path.join(tmpdir, "test_wallet_dir")
    os.makedirs(wallet_dir)
    content_map = key_encoding_with_wallet[4]
    content_map["meta.secret.id"] = key_encoding_with_wallet.encoded

    def mock_get_secret_id(
        id, decoded, content_map=key_encoding_with_wallet.filecontent_secret_map
    ):
        return content_map[id]

    with mock.patch(
        "ads.vault.Vault.get_secret", side_effect=mock_get_secret_id
    ) as mocked_getsecret:
        with ADBSecretKeeper.load_secret(
            wallet_dir=wallet_dir,
            source="meta.secret.id",
        ) as adwsecretkeeper:
            assert adwsecretkeeper == {
                "user_name": key_encoding_with_wallet.credentials.user_name,
                "password": key_encoding_with_wallet.credentials.password,
                "service_name": key_encoding_with_wallet.credentials.service_name,
                "wallet_location": f"{os.path.join(wallet_dir,'wallet.zip')}",
            }
            assert os.environ.get("user_name") is None
            assert os.environ.get("password") is None
            assert os.environ.get("service_name") is None
            assert os.environ.get("wallet_location") is None

        assert adwsecretkeeper == {
            "user_name": None,
            "password": None,
            "service_name": None,
            "wallet_location": None,
        }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_with_wallet_storage_save(
    mock_client, mock_signer, key_encoding_with_wallet, tmpdir
):
    wallet_dir = os.path.join(tmpdir, "test_wallet_dir")
    os.makedirs(wallet_dir)

    file_content_secret_id = key_encoding_with_wallet.filecontent_secret_map
    id_content_map = {file_content_secret_id[id]: id for id in file_content_secret_id}

    def mock_create_secret(
        content,
        secret_name,
        description,
        encode,
        freeform_tags,
        defined_tags,
        content_map=id_content_map,
    ):
        return content_map.get(
            content, "meta.secret.id"
        )  # Default is `meta.secret.id`. It is hard to predict the base64 encoding of the meta secret as we cannot guess the secret id order that will be generated in the meta secret.

    with mock.patch(
        "ads.vault.Vault.create_secret", side_effect=mock_create_secret
    ) as mocked_getsecret:
        adwsecretkeeper = ADBSecretKeeper(
            vault_id="ocid.vault",
            key_id="ocid.key",
            compartment_id="dummy",
            wallet_location=key_encoding_with_wallet.wallet_location,
        )

        saved_secret = adwsecretkeeper.save(
            "testname", "testdescription", save_wallet=True
        )
        assert saved_secret.secret_id == "meta.secret.id"

        saved_secret.export_vault_details(os.path.join(tmpdir, "info.json"))
        with open(os.path.join(tmpdir, "info.json")) as tf:
            assert json.load(tf) == {
                "vault_id": "ocid.vault",
                "key_id": "ocid.key",
                "secret_id": "meta.secret.id",
            }
        saved_secret.export_vault_details(
            os.path.join(tmpdir, "info.yaml"), format="yml"
        )
        with open(os.path.join(tmpdir, "info.yaml")) as tf:
            assert yaml.load(tf, Loader=yaml.FullLoader) == {
                "vault_id": "ocid.vault",
                "key_id": "ocid.key",
                "secret_id": "meta.secret.id",
            }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_with_wallet_storage_save_without_explicit_encode(
    mock_client, mock_signer, key_encoding_with_wallet, tmpdir
):
    wallet_dir = os.path.join(tmpdir, "test_wallet_dir")
    os.makedirs(wallet_dir)

    file_content_secret_id = key_encoding_with_wallet.filecontent_secret_map
    id_content_map = {file_content_secret_id[id]: id for id in file_content_secret_id}

    def mock_create_secret(
        content,
        secret_name,
        description,
        encode,
        freeform_tags,
        defined_tags,
        content_map=id_content_map,
    ):
        return content_map.get(
            content, "meta.secret.id"
        )  # Default is `meta.secret.id`. It is hard to predict the base64 encoding of the meta secret as we cannot guess the secret id order that will be generated in the meta secret.

    with mock.patch(
        "ads.vault.Vault.create_secret", side_effect=mock_create_secret
    ) as mocked_getsecret:
        adwsecretkeeper = ADBSecretKeeper(
            vault_id="ocid.vault",
            key_id="ocid.key",
            compartment_id="dummy",
            wallet_location=key_encoding_with_wallet.wallet_location,
        )

        saved_secret = adwsecretkeeper.save(
            "testname", "testdescription", save_wallet=True
        )
        assert saved_secret.secret_id == "meta.secret.id"

        saved_secret.export_vault_details(os.path.join(tmpdir, "info.json"))
        with open(os.path.join(tmpdir, "info.json")) as tf:
            assert json.load(tf) == {
                "vault_id": "ocid.vault",
                "key_id": "ocid.key",
                "secret_id": "meta.secret.id",
            }
        saved_secret.export_vault_details(
            os.path.join(tmpdir, "info.yaml"), format="yaml"
        )
        with open(os.path.join(tmpdir, "info.yaml")) as tf:
            assert yaml.load(tf, Loader=yaml.FullLoader) == {
                "vault_id": "ocid.vault",
                "key_id": "ocid.key",
                "secret_id": "meta.secret.id",
            }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_adw_with_wallet_storage_load_from_file(
    mock_client, mock_signer, key_encoding_with_wallet, tmpdir
):
    wallet_dir = os.path.join(tmpdir, "test_wallet_dir")
    os.makedirs(wallet_dir)
    content_map = key_encoding_with_wallet[4]
    content_map["meta.secret.id"] = key_encoding_with_wallet.encoded

    with open(os.path.join(tmpdir, "info.json"), "w") as vf:
        json.dump(
            {
                "vault_id": "ocid.vault",
                "key_id": "ocid.key",
                "secret_id": "meta.secret.id",
            },
            vf,
        )

    def mock_get_secret_id(
        id, decoded, content_map=key_encoding_with_wallet.filecontent_secret_map
    ):
        return content_map[id]

    with mock.patch("ads.vault.Vault.get_secret", side_effect=mock_get_secret_id) as _:
        with ADBSecretKeeper.load_secret(
            source=os.path.join(os.path.join(tmpdir, "info.json")),
            format="json",
            wallet_dir=wallet_dir,
            export_env=True,
        ) as adwsecretkeeper:
            assert adwsecretkeeper == {
                "user_name": key_encoding_with_wallet.credentials.user_name,
                "password": key_encoding_with_wallet.credentials.password,
                "service_name": key_encoding_with_wallet.credentials.service_name,
                "wallet_location": f"{os.path.join(wallet_dir,'wallet.zip')}",
            }
            assert (
                os.environ.get("user_name")
                == key_encoding_with_wallet.credentials.user_name
            )
            assert (
                os.environ.get("password")
                == key_encoding_with_wallet.credentials.password
            )
            assert (
                os.environ.get("service_name")
                == key_encoding_with_wallet.credentials.service_name
            )
            assert (
                os.environ.get("wallet_location")
                == f"{os.path.join(wallet_dir,'wallet.zip')}"
            )

        assert adwsecretkeeper == {
            "user_name": None,
            "password": None,
            "service_name": None,
            "wallet_location": None,
        }
        assert os.environ.get("user_name") is None
        assert os.environ.get("password") is None
        assert os.environ.get("service_name") is None
        assert os.environ.get("wallet_location") is None

    with open(os.path.join(tmpdir, "info.yaml"), "w") as vf:
        yaml.dump(
            {
                "vault_id": "ocid.vault",
                "key_id": "ocid.key",
                "secret_id": "meta.secret.id",
            },
            vf,
        )

    def mock_get_secret_id(
        id, decoded, content_map=key_encoding_with_wallet.filecontent_secret_map
    ):
        return content_map[id]

    with mock.patch("ads.vault.Vault.get_secret", side_effect=mock_get_secret_id) as _:
        with ADBSecretKeeper.load_secret(
            source=os.path.join(os.path.join(tmpdir, "info.yaml")),
            format="yaml",
            wallet_dir=wallet_dir,
            export_env=True,
        ) as adwsecretkeeper:
            assert adwsecretkeeper == {
                "user_name": key_encoding_with_wallet.credentials.user_name,
                "password": key_encoding_with_wallet.credentials.password,
                "service_name": key_encoding_with_wallet.credentials.service_name,
                "wallet_location": f"{os.path.join(wallet_dir,'wallet.zip')}",
            }
            assert (
                os.environ.get("user_name")
                == key_encoding_with_wallet.credentials.user_name
            )
            assert (
                os.environ.get("password")
                == key_encoding_with_wallet.credentials.password
            )
            assert (
                os.environ.get("service_name")
                == key_encoding_with_wallet.credentials.service_name
            )
            assert (
                os.environ.get("wallet_location")
                == f"{os.path.join(wallet_dir,'wallet.zip')}"
            )

        assert adwsecretkeeper == {
            "user_name": None,
            "password": None,
            "service_name": None,
            "wallet_location": None,
        }
        assert os.environ.get("user_name") is None
        assert os.environ.get("password") is None
        assert os.environ.get("service_name") is None
        assert os.environ.get("wallet_location") is None
