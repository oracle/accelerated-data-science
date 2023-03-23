#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.secrets import SecretKeeper
from base64 import b64encode
import pytest
from unittest.mock import patch


@pytest.fixture
def key_encoding():
    content = "a1b2c33d4e5f6g7h8i9jakblc".encode("utf-8")
    encoded = b64encode(content).decode("utf-8")
    return (content, encoded)


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_encode(mock_client, mock_signer, key_encoding):

    with pytest.raises(ValueError, match="No payload to encode"):
        secretkeeper = SecretKeeper(
            key_encoding[0],
            vault_id="ocid1.vault.oc1.<unique_ocid>",
            key_id="ocid1.key.oc1.<unique_ocid>",
            compartment_id="ocid1.compartment.oc1..<unique_ocid>",
        )
        secretkeeper.encode()


@patch("ads.common.auth.default_signer")
@patch("ads.common.oci_client.OCIClientFactory")
def test_decode(mock_client, mock_signer, key_encoding):
    secretkeeper = SecretKeeper(
        None,
        vault_id="ocid1.vault.oc1.<unique_ocid>",
        key_id="ocid1.key.oc1.<unique_ocid>",
        compartment_id="ocid1.compartment.oc1..<unique_ocid>",
    )
    secretkeeper.encoded = key_encoding[1]
    assert secretkeeper.decode().secret == key_encoding[0].decode("utf-8")
