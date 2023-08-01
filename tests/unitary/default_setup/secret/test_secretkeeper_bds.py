#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import hashlib
import json
import os
from base64 import b64encode
from collections import namedtuple
from unittest import mock
from unittest.mock import patch

import pytest
import yaml
from ads.secrets.big_data_service import BDSSecretKeeper


@pytest.fixture
def key_encoding():
    principal = "fake_principal"
    hdfs_host = "fake_hdfs_host"
    hive_host = "fake_hive_host"
    hdfs_port = "fake_hdfs_port"
    hive_port = "fake_hive_port"
    kerb5_path = "kerb5_fake_location"
    keytab_path = "keytab_fake_location"

    secret_dict = {
        "principal": principal,
        "hdfs_host": hdfs_host,
        "hive_host": hive_host,
        "hdfs_port": hdfs_port,
        "hive_port": hive_port,
        "kerb5_path": kerb5_path,
        "keytab_path": keytab_path,
    }
    encoded = b64encode(json.dumps(secret_dict).encode("utf-8")).decode("utf-8")
    return (
        (
            principal,
            hdfs_host,
            hive_host,
            hdfs_port,
            hive_port,
            kerb5_path,
            None,
            keytab_path,
            None,
        ),
        secret_dict,
        encoded,
    )


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_encode(mock_client, mock_signer, key_encoding):
    bdssecretkeeper = BDSSecretKeeper(
        *key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )

    assert bdssecretkeeper.encode(serialize=False).encoded == key_encoding[2]


def generate_keytab_kerb5_config_data(
    keytab_path="fake_keytab_path.keytab", kerb5_path="fake_kerb5_path.cfg"
):
    file_content = {}
    file_encoded = {}
    file_dict = {}
    file_secret_id = "ocid.secret."
    content_map = {}
    filenames = [(keytab_path, "keytab_content"), (kerb5_path, "kerb5_content")]
    for i, (file_name, key) in enumerate(filenames):
        with open(file_name, "wb") as outfile:
            content = f"file content {i}".encode("utf-8")
            outfile.write(content)
            file_content[os.path.basename(file_name)] = content
            file_encoded[os.path.basename(file_name)] = b64encode(content).decode(
                "utf-8"
            )
        file_dict[key] = file_encoded[os.path.basename(file_name)]
    content_map[f"ocid.secret."] = b64encode(
        json.dumps(file_dict).encode("utf-8")
    ).decode("utf-8")

    return file_content, file_encoded, filenames, file_secret_id, content_map


@pytest.fixture
def key_encoding_with_keytab_kerb5(tmpdir):
    principal = "fake_principal"
    hdfs_host = "fake_hdfs_host"
    hive_host = "fake_hive_host"
    hdfs_port = "fake_hdfs_port"
    hive_port = "fake_hive_port"

    kerb5_path = "fake_kerb5_path.cfg"
    keytab_path = "fake_keytab_path.keytab"

    keytab_path = "fake_keytab_path.keytab"
    kerb5_path = "fake_kerb5_path.cfg"
    file_dir = os.path.join(tmpdir, "keytab_krb5")
    os.makedirs(file_dir)
    keytab_path = os.path.join(file_dir, keytab_path)
    kerb5_path = os.path.join(file_dir, kerb5_path)

    keytab_krb5_details = generate_keytab_kerb5_config_data(keytab_path, kerb5_path)

    secret_dict = {
        "principal": principal,
        "hdfs_host": hdfs_host,
        "hive_host": hive_host,
        "hdfs_port": hdfs_port,
        "hive_port": hive_port,
        "kerb5_path": kerb5_path,
        "keytab_path": keytab_path,
    }
    encoded_secret_data = {
        **secret_dict,
        "keytab_content": keytab_krb5_details[1][os.path.basename(keytab_path)],
        "kerb5_content": keytab_krb5_details[1][os.path.basename(kerb5_path)],
    }
    saved_secret_data = {**secret_dict, "secret_id": keytab_krb5_details[3]}

    encoded = b64encode(json.dumps(saved_secret_data).encode("utf-8")).decode("utf-8")

    credentials = namedtuple(
        "Credentials",
        [
            "principal",
            "hdfs_host",
            "hive_host",
            "hdfs_port",
            "hive_port",
            "kerb5_path",
            "kerb5_content",
            "keytab_path",
            "keytab_content",
        ],
    )
    testdata = namedtuple(
        "TestData",
        [
            "credentials",
            "encoded_secret_data",
            "encoded",
            "filecontent_secret_map",
        ],
    )
    return testdata(
        credentials(
            principal,
            hdfs_host,
            hive_host,
            hdfs_port,
            hive_port,
            kerb5_path,
            None,
            keytab_path,
            None,
        ),
        encoded_secret_data,
        encoded,
        keytab_krb5_details[4],
    )


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_encode_with_keytab_kerb5(
    mock_client, mock_signer, key_encoding_with_keytab_kerb5
):
    bdssecretkeeper = BDSSecretKeeper(
        *key_encoding_with_keytab_kerb5.credentials,
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    assert (
        bdssecretkeeper.encode(serialize=True).encoded
        == key_encoding_with_keytab_kerb5.encoded_secret_data
    )


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_save_without_explicit_encoding(
    mock_client, mock_signer, key_encoding, tmpdir
):
    bdssecretkeeper = BDSSecretKeeper(
        *key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )
    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            bdssecretkeeper.encode(serialize=False)
            .save("testname", "testdescription", save_files=False)
            .secret_id
            == "ocid.secret.id"
        )
        bdssecretkeeper.export_vault_details(os.path.join(tmpdir, "test.json"))
        with open(os.path.join(tmpdir, "test.json")) as tf:
            assert json.load(tf) == {
                "key_id": "ocid.key",
                "secret_id": "ocid.secret.id",
                "vault_id": "ocid.vault",
            }
        bdssecretkeeper.export_vault_details(os.path.join(tmpdir, "test.yaml"))
        with open(os.path.join(tmpdir, "test.yaml")) as tf:
            assert yaml.load(tf, Loader=yaml.FullLoader) == {
                "key_id": "ocid.key",
                "secret_id": "ocid.secret.id",
                "vault_id": "ocid.vault",
            }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_save(mock_client, mock_signer, key_encoding_with_keytab_kerb5, tmpdir):
    bdssecretkeeper = BDSSecretKeeper(
        *key_encoding_with_keytab_kerb5.credentials,
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )

    with mock.patch(
        "ads.vault.Vault.create_secret", return_value="ocid.secret.id"
    ) as _:
        assert (
            bdssecretkeeper.save("testname", "testdescription").secret_id
            == "ocid.secret.id"
        )
        bdssecretkeeper.export_vault_details(os.path.join(tmpdir, "test.json"))
        with open(os.path.join(tmpdir, "test.json")) as tf:
            assert json.load(tf) == {
                "key_id": "ocid.key",
                "secret_id": "ocid.secret.id",
                "vault_id": "ocid.vault",
            }
        bdssecretkeeper.export_vault_details(os.path.join(tmpdir, "test.yaml"))
        with open(os.path.join(tmpdir, "test.yaml")) as tf:
            assert yaml.load(tf, Loader=yaml.FullLoader) == {
                "key_id": "ocid.key",
                "secret_id": "ocid.secret.id",
                "vault_id": "ocid.vault",
            }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_context(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with BDSSecretKeeper.load_secret(
            source="ocid.secret.id",
            export_env=True,
        ) as bdssecretkeeper:
            assert bdssecretkeeper == key_encoding[1]
            assert os.environ.get("principal") == key_encoding[0][0]
            assert os.environ.get("hdfs_host") == key_encoding[0][1]
            assert os.environ.get("hive_host") == key_encoding[0][2]
            assert os.environ.get("hdfs_port") == key_encoding[0][3]
            assert os.environ.get("hive_port") == key_encoding[0][4]
            assert os.environ.get("kerb5_path") == key_encoding[0][5]
            assert os.environ.get("keytab_path") == key_encoding[0][7]
        assert bdssecretkeeper == {
            "principal": None,
            "hdfs_host": None,
            "hive_host": None,
            "hdfs_port": None,
            "hive_port": None,
            "kerb5_path": None,
            "keytab_path": None,
        }
        assert os.environ.get("principal") is None
        assert os.environ.get("hdfs_host") is None
        assert os.environ.get("hive_host") is None
        assert os.environ.get("hdfs_port") is None
        assert os.environ.get("hive_port") is None
        assert os.environ.get("kerb5_path") is None
        assert os.environ.get("keytab_path") is None


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_keeper_no_file(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with BDSSecretKeeper.load_secret(
            source="ocid.secret.id",
        ) as bdssecretkeeper:
            assert bdssecretkeeper == {
                **key_encoding[1],
            }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_context_namespace(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with BDSSecretKeeper.load_secret(
            source="ocid.secret.id",
            export_prefix="myapp",
            export_env=True,
        ) as bdssecretkeeper:
            assert bdssecretkeeper == {
                **key_encoding[1],
            }
            assert bdssecretkeeper == key_encoding[1]
            assert os.environ.get("myapp.principal") == key_encoding[0][0]
            assert os.environ.get("myapp.hdfs_host") == key_encoding[0][1]
            assert os.environ.get("myapp.hive_host") == key_encoding[0][2]
            assert os.environ.get("myapp.hdfs_port") == key_encoding[0][3]
            assert os.environ.get("myapp.hive_port") == key_encoding[0][4]
            assert os.environ.get("myapp.kerb5_path") == key_encoding[0][5]
            assert os.environ.get("myapp.keytab_path") == key_encoding[0][7]
        assert bdssecretkeeper == {
            "principal": None,
            "hdfs_host": None,
            "hive_host": None,
            "hdfs_port": None,
            "hive_port": None,
            "kerb5_path": None,
            "keytab_path": None,
        }
        assert os.environ.get("myapp.principal") is None
        assert os.environ.get("myapp.hdfs_host") is None
        assert os.environ.get("myapp.hive_host") is None
        assert os.environ.get("myapp.hdfs_port") is None
        assert os.environ.get("myapp.hive_port") is None
        assert os.environ.get("myapp.kerb5_path") is None
        assert os.environ.get("myapp.keytab_path") is None


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_context_noexport(mock_client, mock_signer, key_encoding):
    with mock.patch(
        "ads.vault.Vault.get_secret", return_value=key_encoding[2]
    ) as mocked_getsecret:
        with BDSSecretKeeper.load_secret(
            source="ocid.secret.id",
        ) as bdssecretkeeper:
            assert bdssecretkeeper == {
                **key_encoding[1],
            }

        assert os.environ.get("myapp.principal") is None
        assert os.environ.get("myapp.hdfs_host") is None
        assert os.environ.get("myapp.hive_host") is None
        assert os.environ.get("myapp.hdfs_port") is None
        assert os.environ.get("myapp.hive_port") is None
        assert os.environ.get("myapp.kerb5_path") is None
        assert os.environ.get("myapp.keytab_path") is None
        assert bdssecretkeeper == {
            "principal": None,
            "hdfs_host": None,
            "hive_host": None,
            "hdfs_port": None,
            "hive_port": None,
            "kerb5_path": None,
            "keytab_path": None,
        }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_decode(mock_client, mock_signer, key_encoding):
    bdssecretkeeper = BDSSecretKeeper(
        *key_encoding[0],
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )

    bdssecretkeeper.encoded = key_encoding[2]
    decoded = bdssecretkeeper.decode(save_files=False)

    assert decoded.data.principal == key_encoding[0][0]
    assert decoded.data.hdfs_host == key_encoding[0][1]
    assert decoded.data.hive_host == key_encoding[0][2]
    assert decoded.data.hdfs_port == key_encoding[0][3]
    assert decoded.data.hive_port == key_encoding[0][4]
    assert decoded.data.kerb5_path == key_encoding[0][5]
    assert decoded.data.keytab_path == key_encoding[0][7]

    assert decoded.data.to_dict() == {
        **key_encoding[1],
    }


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_with_keytab_kerb5_encode(
    mock_client, mock_signer, key_encoding_with_keytab_kerb5
):
    bdssecretkeeper = BDSSecretKeeper(
        *key_encoding_with_keytab_kerb5.credentials,
        vault_id="ocid.vault",
        key_id="ocid.key",
        compartment_id="dummy",
    )

    assert (
        bdssecretkeeper.encode(serialize=True).encoded
        == key_encoding_with_keytab_kerb5.encoded_secret_data
    )


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_with_keytab_kerb5_encode_value_error(mock_client, mock_signer):
    with pytest.raises(ValueError):
        bdssecretkeeper = BDSSecretKeeper(
            "principal",
            "hdfs_host",
            "hive_host",
            "hdfs_port",
            "hive_port",
            None,
            None,
            None,
            None,
            vault_id="ocid.vault",
            key_id="ocid.key",
            compartment_id="dummy",
        )
        bdssecretkeeper.encode(serialize=True)


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_with_keytab_kerb5_decode(
    mock_client, mock_signer, key_encoding_with_keytab_kerb5, tmpdir
):
    def mock_get_secret_id(
        id, decoded, content_map=key_encoding_with_keytab_kerb5.filecontent_secret_map
    ):
        return content_map[id]

    with mock.patch(
        "ads.vault.Vault.get_secret", side_effect=mock_get_secret_id
    ) as mocked_getsecret:
        bdssecretkeeper = BDSSecretKeeper(
            vault_id="ocid.vault",
            key_id="ocid.key",
            compartment_id="dummy",
        )
        bdssecretkeeper.encoded = key_encoding_with_keytab_kerb5.encoded
        decoded = bdssecretkeeper.decode()

        assert (
            decoded.data.principal
            == key_encoding_with_keytab_kerb5.credentials.principal
        )
        assert (
            decoded.data.hdfs_host
            == key_encoding_with_keytab_kerb5.credentials.hdfs_host
        )
        assert (
            decoded.data.hive_host
            == key_encoding_with_keytab_kerb5.credentials.hive_host
        )
        assert (
            decoded.data.hdfs_port
            == key_encoding_with_keytab_kerb5.credentials.hdfs_port
        )
        assert (
            decoded.data.hive_port
            == key_encoding_with_keytab_kerb5.credentials.hive_port
        )
        assert (
            decoded.data.kerb5_path
            == key_encoding_with_keytab_kerb5.credentials.kerb5_path
        )
        assert (
            decoded.data.keytab_path
            == key_encoding_with_keytab_kerb5.credentials.keytab_path
        )

        assert bdssecretkeeper.to_dict() == {
            "principal": key_encoding_with_keytab_kerb5.credentials.principal,
            "hdfs_host": key_encoding_with_keytab_kerb5.credentials.hdfs_host,
            "hive_host": key_encoding_with_keytab_kerb5.credentials.hive_host,
            "hdfs_port": key_encoding_with_keytab_kerb5.credentials.hdfs_port,
            "hive_port": key_encoding_with_keytab_kerb5.credentials.hive_port,
            "kerb5_path": key_encoding_with_keytab_kerb5.credentials.kerb5_path,
            "keytab_path": key_encoding_with_keytab_kerb5.credentials.keytab_path,
        }
        with open(decoded.data.kerb5_path, "rb") as orgfile:
            with open(
                f"{os.path.expanduser('~/.bds_config/krb5.conf')}", "rb"
            ) as newfile:
                assert (
                    hashlib.md5(orgfile.read()).hexdigest()
                    == hashlib.md5(newfile.read()).hexdigest()
                )
        with open(decoded.data.keytab_path, "rb") as orgfile:
            with open(
                f"{key_encoding_with_keytab_kerb5.credentials.keytab_path}", "rb"
            ) as newfile:
                assert (
                    hashlib.md5(orgfile.read()).hexdigest()
                    == hashlib.md5(newfile.read()).hexdigest()
                )


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_with_keytab_krb5_context_manager(
    mock_client, mock_signer, key_encoding_with_keytab_kerb5, tmpdir
):
    files_dir = os.path.join(tmpdir, "test_keytab_krb5_folder")
    os.makedirs(files_dir)
    content_map = key_encoding_with_keytab_kerb5[3]
    content_map["meta.secret.id"] = key_encoding_with_keytab_kerb5.encoded

    def mock_get_secret_id(
        id, decoded, content_map=key_encoding_with_keytab_kerb5.filecontent_secret_map
    ):
        return content_map[id]

    with mock.patch(
        "ads.vault.Vault.get_secret", side_effect=mock_get_secret_id
    ) as mocked_getsecret:
        with BDSSecretKeeper.load_secret(
            source="meta.secret.id", export_env=True
        ) as bdssecretkeeper:
            assert bdssecretkeeper == {
                "principal": key_encoding_with_keytab_kerb5.credentials.principal,
                "hdfs_host": key_encoding_with_keytab_kerb5.credentials.hdfs_host,
                "hive_host": key_encoding_with_keytab_kerb5.credentials.hive_host,
                "hdfs_port": key_encoding_with_keytab_kerb5.credentials.hdfs_port,
                "hive_port": key_encoding_with_keytab_kerb5.credentials.hive_port,
                "kerb5_path": key_encoding_with_keytab_kerb5.credentials.kerb5_path,
                "keytab_path": key_encoding_with_keytab_kerb5.credentials.keytab_path,
            }
            assert os.environ.get("principal") == key_encoding_with_keytab_kerb5[0][0]
            assert os.environ.get("hdfs_host") == key_encoding_with_keytab_kerb5[0][1]
            assert os.environ.get("hive_host") == key_encoding_with_keytab_kerb5[0][2]
            assert os.environ.get("hdfs_port") == key_encoding_with_keytab_kerb5[0][3]
            assert os.environ.get("hive_port") == key_encoding_with_keytab_kerb5[0][4]
            assert os.environ.get("kerb5_path") == key_encoding_with_keytab_kerb5[0][5]
            assert os.environ.get("keytab_path") == key_encoding_with_keytab_kerb5[0][7]

        assert bdssecretkeeper == {
            "principal": None,
            "hdfs_host": None,
            "hive_host": None,
            "hdfs_port": None,
            "hive_port": None,
            "kerb5_path": None,
            "keytab_path": None,
        }

        assert os.environ.get("principal") is None
        assert os.environ.get("hdfs_host") is None
        assert os.environ.get("hive_host") is None
        assert os.environ.get("hdfs_port") is None
        assert os.environ.get("hive_port") is None
        assert os.environ.get("kerb5_path") is None
        assert os.environ.get("keytab_path") is None


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_with_keytab_kerb5_context_manager_namespace(
    mock_client, mock_signer, key_encoding_with_keytab_kerb5, tmpdir
):
    files_dir = os.path.join(tmpdir, "test_keytab_kerb5_dir")
    os.makedirs(files_dir)
    content_map = key_encoding_with_keytab_kerb5[3]
    content_map["meta.secret.id"] = key_encoding_with_keytab_kerb5.encoded

    def mock_get_secret_id(
        id, decoded, content_map=key_encoding_with_keytab_kerb5.filecontent_secret_map
    ):
        return content_map[id]

    with mock.patch(
        "ads.vault.Vault.get_secret", side_effect=mock_get_secret_id
    ) as mocked_getsecret:
        with BDSSecretKeeper.load_secret(
            source="meta.secret.id",
            export_prefix="myapp",
            export_env=True,
        ) as bdssecretkeeper:
            assert bdssecretkeeper == {
                "principal": key_encoding_with_keytab_kerb5.credentials.principal,
                "hdfs_host": key_encoding_with_keytab_kerb5.credentials.hdfs_host,
                "hive_host": key_encoding_with_keytab_kerb5.credentials.hive_host,
                "hdfs_port": key_encoding_with_keytab_kerb5.credentials.hdfs_port,
                "hive_port": key_encoding_with_keytab_kerb5.credentials.hive_port,
                "kerb5_path": key_encoding_with_keytab_kerb5.credentials.kerb5_path,
                "keytab_path": key_encoding_with_keytab_kerb5.credentials.keytab_path,
            }
            assert (
                os.environ.get("myapp.principal")
                == key_encoding_with_keytab_kerb5[0][0]
            )
            assert (
                os.environ.get("myapp.hdfs_host")
                == key_encoding_with_keytab_kerb5[0][1]
            )
            assert (
                os.environ.get("myapp.hive_host")
                == key_encoding_with_keytab_kerb5[0][2]
            )
            assert (
                os.environ.get("myapp.hdfs_port")
                == key_encoding_with_keytab_kerb5[0][3]
            )
            assert (
                os.environ.get("myapp.hive_port")
                == key_encoding_with_keytab_kerb5[0][4]
            )
            assert (
                os.environ.get("myapp.kerb5_path")
                == key_encoding_with_keytab_kerb5[0][5]
            )
            assert (
                os.environ.get("myapp.keytab_path")
                == key_encoding_with_keytab_kerb5[0][7]
            )

        assert bdssecretkeeper == {
            "principal": None,
            "hdfs_host": None,
            "hive_host": None,
            "hdfs_port": None,
            "hive_port": None,
            "kerb5_path": None,
            "keytab_path": None,
        }
        assert os.environ.get("myapp.principal") is None
        assert os.environ.get("myapp.hdfs_host") is None
        assert os.environ.get("myapp.hive_host") is None
        assert os.environ.get("myapp.hdfs_port") is None
        assert os.environ.get("myapp.hive_port") is None
        assert os.environ.get("myapp.kerb5_path") is None
        assert os.environ.get("myapp.keytab_path") is None


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_with_keytab_kerb5_context_manager_noexport(
    mock_client, mock_signer, key_encoding_with_keytab_kerb5, tmpdir
):
    content_map = key_encoding_with_keytab_kerb5[3]
    content_map["meta.secret.id"] = key_encoding_with_keytab_kerb5.encoded

    def mock_get_secret_id(
        id, decoded, content_map=key_encoding_with_keytab_kerb5.filecontent_secret_map
    ):
        return content_map[id]

    with mock.patch(
        "ads.vault.Vault.get_secret", side_effect=mock_get_secret_id
    ) as mocked_getsecret:
        with BDSSecretKeeper.load_secret(
            source="meta.secret.id",
        ) as bdssecretkeeper:
            assert bdssecretkeeper == {
                "principal": key_encoding_with_keytab_kerb5.credentials.principal,
                "hdfs_host": key_encoding_with_keytab_kerb5.credentials.hdfs_host,
                "hive_host": key_encoding_with_keytab_kerb5.credentials.hive_host,
                "hdfs_port": key_encoding_with_keytab_kerb5.credentials.hdfs_port,
                "hive_port": key_encoding_with_keytab_kerb5.credentials.hive_port,
                "kerb5_path": key_encoding_with_keytab_kerb5.credentials.kerb5_path,
                "keytab_path": key_encoding_with_keytab_kerb5.credentials.keytab_path,
            }
            assert os.environ.get("principal") is None
            assert os.environ.get("hdfs_host") is None
            assert os.environ.get("hive_host") is None
            assert os.environ.get("hdfs_port") is None
            assert os.environ.get("hive_port") is None
            assert os.environ.get("kerb5_path") is None
            assert os.environ.get("keytab_path") is None

        assert bdssecretkeeper == {
            "principal": None,
            "hdfs_host": None,
            "hive_host": None,
            "hdfs_port": None,
            "hive_port": None,
            "kerb5_path": None,
            "keytab_path": None,
        }
        assert os.environ.get("principal") is None
        assert os.environ.get("hdfs_host") is None
        assert os.environ.get("hive_host") is None
        assert os.environ.get("hdfs_port") is None
        assert os.environ.get("hive_port") is None
        assert os.environ.get("kerb5_path") is None
        assert os.environ.get("keytab_path") is None


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_bds_with_keytab_kerb5_load_from_file(
    mock_client, mock_signer, key_encoding_with_keytab_kerb5, tmpdir
):
    content_map = key_encoding_with_keytab_kerb5[3]
    content_map["meta.secret.id"] = key_encoding_with_keytab_kerb5.encoded

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
        id, decoded, content_map=key_encoding_with_keytab_kerb5.filecontent_secret_map
    ):
        return content_map[id]

    with mock.patch("ads.vault.Vault.get_secret", side_effect=mock_get_secret_id) as _:
        with BDSSecretKeeper.load_secret(
            source=os.path.join(os.path.join(tmpdir, "info.json")),
            format="json",
            export_env=True,
        ) as bdssecretkeeper:
            assert bdssecretkeeper == {
                "principal": key_encoding_with_keytab_kerb5.credentials.principal,
                "hdfs_host": key_encoding_with_keytab_kerb5.credentials.hdfs_host,
                "hive_host": key_encoding_with_keytab_kerb5.credentials.hive_host,
                "hdfs_port": key_encoding_with_keytab_kerb5.credentials.hdfs_port,
                "hive_port": key_encoding_with_keytab_kerb5.credentials.hive_port,
                "kerb5_path": key_encoding_with_keytab_kerb5.credentials.kerb5_path,
                "keytab_path": key_encoding_with_keytab_kerb5.credentials.keytab_path,
            }
            assert os.environ.get("principal") == key_encoding_with_keytab_kerb5[0][0]
            assert os.environ.get("hdfs_host") == key_encoding_with_keytab_kerb5[0][1]
            assert os.environ.get("hive_host") == key_encoding_with_keytab_kerb5[0][2]
            assert os.environ.get("hdfs_port") == key_encoding_with_keytab_kerb5[0][3]
            assert os.environ.get("hive_port") == key_encoding_with_keytab_kerb5[0][4]
            assert os.environ.get("kerb5_path") == key_encoding_with_keytab_kerb5[0][5]
            assert os.environ.get("keytab_path") == key_encoding_with_keytab_kerb5[0][7]

        assert bdssecretkeeper == {
            "principal": None,
            "hdfs_host": None,
            "hive_host": None,
            "hdfs_port": None,
            "hive_port": None,
            "kerb5_path": None,
            "keytab_path": None,
        }
        assert os.environ.get("principal") is None
        assert os.environ.get("hdfs_host") is None
        assert os.environ.get("hive_host") is None
        assert os.environ.get("hdfs_port") is None
        assert os.environ.get("hive_port") is None
        assert os.environ.get("kerb5_path") is None
        assert os.environ.get("keytab_path") is None

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
        id, decoded, content_map=key_encoding_with_keytab_kerb5.filecontent_secret_map
    ):
        return content_map[id]

    with mock.patch("ads.vault.Vault.get_secret", side_effect=mock_get_secret_id) as _:
        with BDSSecretKeeper.load_secret(
            source=os.path.join(os.path.join(tmpdir, "info.yaml")),
            format="yaml",
            export_env=True,
        ) as bdssecretkeeper:
            assert bdssecretkeeper == {
                "principal": key_encoding_with_keytab_kerb5.credentials.principal,
                "hdfs_host": key_encoding_with_keytab_kerb5.credentials.hdfs_host,
                "hive_host": key_encoding_with_keytab_kerb5.credentials.hive_host,
                "hdfs_port": key_encoding_with_keytab_kerb5.credentials.hdfs_port,
                "hive_port": key_encoding_with_keytab_kerb5.credentials.hive_port,
                "kerb5_path": key_encoding_with_keytab_kerb5.credentials.kerb5_path,
                "keytab_path": key_encoding_with_keytab_kerb5.credentials.keytab_path,
            }

            assert os.environ.get("principal") == key_encoding_with_keytab_kerb5[0][0]
            assert os.environ.get("hdfs_host") == key_encoding_with_keytab_kerb5[0][1]
            assert os.environ.get("hive_host") == key_encoding_with_keytab_kerb5[0][2]
            assert os.environ.get("hdfs_port") == key_encoding_with_keytab_kerb5[0][3]
            assert os.environ.get("hive_port") == key_encoding_with_keytab_kerb5[0][4]
            assert os.environ.get("kerb5_path") == key_encoding_with_keytab_kerb5[0][5]
            assert os.environ.get("keytab_path") == key_encoding_with_keytab_kerb5[0][7]

        assert bdssecretkeeper == {
            "principal": None,
            "hdfs_host": None,
            "hive_host": None,
            "hdfs_port": None,
            "hive_port": None,
            "kerb5_path": None,
            "keytab_path": None,
        }
        assert os.environ.get("principal") is None
        assert os.environ.get("hdfs_host") is None
        assert os.environ.get("hive_host") is None
        assert os.environ.get("hdfs_port") is None
        assert os.environ.get("hive_port") is None
        assert os.environ.get("kerb5_path") is None
        assert os.environ.get("keytab_path") is None
