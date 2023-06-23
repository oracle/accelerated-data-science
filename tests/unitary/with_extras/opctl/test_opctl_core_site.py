#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
import pytest
import oci

from mock import patch

from ads.opctl.spark.cmds import generate_core_site_properties


class TestCoreSite:
    @patch("oci.config.validate_config")
    @patch("oci.signer.load_private_key_from_file")
    @patch("oci.config.invalid_key_file_path_checker")
    def test_generate_core_site_with_api_key(self, mock_checker, mock_load_key, mock_validate_config):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "config.ini"), "w") as f:
                f.write(
                    """
[PROFILE_NAME]
user = ocid1.user.oc1.xxxxx
fingerprint = 79:42:80:31:52:12:34
tenancy = ocid1.tenancy.oc1.xxxx
region = us-ashburn-1
key_file = ~/.oci/oci_api_key.pem
                """
                )
            properties = generate_core_site_properties(
                "api_key", os.path.join(td, "config.ini"), "PROFILE_NAME"
            )
            assert properties == [
                (
                    "fs.oci.client.hostname",
                    f"https://objectstorage.us-ashburn-1.oraclecloud.com",
                ),
                ("fs.oci.client.auth.tenantId", "ocid1.tenancy.oc1.xxxx"),
                ("fs.oci.client.auth.userId", "ocid1.user.oc1.xxxxx"),
                ("fs.oci.client.auth.fingerprint", "79:42:80:31:52:12:34"),
                (
                    "fs.oci.client.auth.pemfilepath",
                    os.path.expanduser("~/.oci/oci_api_key.pem"),
                ),
            ]

    def test_generate_core_site_with_rp(self, monkeypatch):
        monkeypatch.setenv("NB_REGION", "us-ashburn-1")
        properties = generate_core_site_properties("resource_principal")
        assert properties == [
            (
                "fs.oci.client.hostname",
                f"https://objectstorage.us-ashburn-1.oraclecloud.com",
            ),
            (
                "fs.oci.client.custom.authenticator",
                "com.oracle.bmc.hdfs.auth.ResourcePrincipalsCustomAuthenticator",
            ),
        ]
