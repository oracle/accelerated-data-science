#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import patch
import pytest
from pytest import MonkeyPatch
import importlib

import ads


class TestUserAgent:
    def setup_method(self):
        # "tenancy", "user" and "fingerprint" can be any random value
        self.test_config = {
            "tenancy": "ocid1.tenancy.oc1..<unique_ocid>",
            "user": "ocid1.user.oc1..<unique_ocid>",
            "fingerprint": "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
            "key_file": "<path>/<to>/<key_file>",
            "region": "test_region",
        }

    def teardown_method(self):
        self.test_config = {}

    @patch("oci.signer.Signer")
    def test_user_agent_api_keys_using_test_profile(self, mock_signer):
        with patch("oci.config.from_file", return_value=self.test_config):
            auth_info = ads.auth.api_keys("test_path", "TEST_PROFILE")
            assert (
                auth_info["config"].get("additional_user_agent")
                == f"Oracle-ads/version={ads.__version__}/surface=WORKSTATION"
            )

    @patch("oci.auth.signers.get_resource_principals_signer")
    def test_user_agent_rp(self, mock_signer):
        importlib.reload(ads.config)
        importlib.reload(ads.telemetry)
        auth_info = ads.auth.resource_principal()
        assert (
            auth_info["config"].get("additional_user_agent")
            == f"Oracle-ads/version={ads.__version__}/surface=WORKSTATION"
        )

    @patch("oci.signer.load_private_key_from_file")
    def test_user_agent_default_signer(self, mock_load_key_file):
        monkeypatch = MonkeyPatch()
        monkeypatch.delenv("OCI_RESOURCE_PRINCIPAL_VERSION", raising=False)
        importlib.reload(ads.config)
        with patch("oci.config.from_file", return_value=self.test_config):
            auth_info = ads.auth.default_signer()
            assert (
                auth_info["config"].get("additional_user_agent")
                == f"Oracle-ads/version={ads.__version__}/surface=WORKSTATION"
            )

    @pytest.mark.parametrize(
        "RESOURCE_KEY, USER_AGENT_VALUE",
        [
            ("MD_OCID", "DATASCIENCE_MODEL_DEPLOYMENT"),
            ("JOB_RUN_OCID", "DATASCIENCE_JOB"),
            ("NB_SESSION_OCID", "DATASCIENCE_NOTEBOOK"),
            ("DATAFLOW_RUN_ID", "DATAFLOW"),
        ],
    )
    @patch("oci.signer.load_private_key_from_file")
    def test_user_agent_default_signer_known_resources(
        self, mock_load_key_file, RESOURCE_KEY, USER_AGENT_VALUE
    ):
        monkeypatch = MonkeyPatch()
        monkeypatch.setenv("OCI_RESOURCE_PRINCIPAL_VERSION", "1.1")
        monkeypatch.setenv(RESOURCE_KEY, "1234")

        importlib.reload(ads.config)
        importlib.reload(ads)
        importlib.reload(ads.auth)
        importlib.reload(ads.telemetry)

        with patch("oci.config.from_file", return_value=self.test_config):
            auth_info = ads.auth.default_signer()
            assert (
                auth_info["config"].get("additional_user_agent")
                == f"Oracle-ads/version={ads.__version__}/surface={USER_AGENT_VALUE}"
            )
        monkeypatch.delenv("OCI_RESOURCE_PRINCIPAL_VERSION", raising=False)
        monkeypatch.delenv(RESOURCE_KEY, raising=False)

    @patch("oci.signer.Signer")
    def test_user_agent_default_singer_ociservice(
        self,
        mock_signer,
    ):
        monkeypatch = MonkeyPatch()
        monkeypatch.setenv("OCI_RESOURCE_PRINCIPAL_VERSION", "1.1")

        importlib.reload(ads.config)
        importlib.reload(ads.telemetry)

        with patch("oci.config.from_file", return_value=self.test_config):
            auth_info = ads.auth.default_signer()
            assert (
                auth_info["config"].get("additional_user_agent")
                == f"Oracle-ads/version={ads.__version__}/surface=OCI_SERVICE"
            )
        monkeypatch.delenv("OCI_RESOURCE_PRINCIPAL_VERSION", raising=False)
