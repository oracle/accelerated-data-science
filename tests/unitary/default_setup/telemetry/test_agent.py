#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib
import oci
from unittest.mock import patch

import pytest
# from pytest import MonkeyPatch

import ads
from ads.telemetry.telemetry import (
    EXTRA_USER_AGENT_INFO,
    LIBRARY,
    UNKNOWN,
    USER_AGENT_KEY,
)


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

    @patch("oci.config.validate_config")
    @patch("oci.signer.Signer")
    def test_user_agent_api_keys_using_test_profile(self, mock_signer, mock_validate_config):
        with patch("oci.config.from_file", return_value=self.test_config):
            auth_info = ads.auth.api_keys("test_path", "TEST_PROFILE")
            assert (
                auth_info["config"].get("additional_user_agent")
                == f"{LIBRARY}/version={ads.__version__}#surface=WORKSTATION#api={UNKNOWN}"
            )

    @patch("oci.auth.signers.get_resource_principals_signer")
    def test_user_agent_rp(self, mock_signer, monkeypatch):
        monkeypatch.delenv("OCI_RESOURCE_PRINCIPAL_VERSION", raising=False)
        importlib.reload(ads.config)
        importlib.reload(ads.telemetry)
        auth_info = ads.auth.resource_principal()
        assert (
            auth_info["config"].get("additional_user_agent")
            == f"{LIBRARY}/version={ads.__version__}#surface=WORKSTATION#api={UNKNOWN}"
        )

    @patch("oci.config.validate_config")
    @patch("oci.signer.load_private_key_from_file")
    def test_user_agent_default_signer(self, mock_load_key_file, mock_validate_config, monkeypatch):
        # monkeypatch = MonkeyPatch()
        monkeypatch.delenv("OCI_RESOURCE_PRINCIPAL_VERSION", raising=False)
        importlib.reload(ads.config)
        with patch("oci.config.from_file", return_value=self.test_config):
            auth_info = ads.auth.default_signer()
            assert (
                auth_info["config"].get("additional_user_agent")
                == f"{LIBRARY}/version={ads.__version__}#surface=WORKSTATION#api={UNKNOWN}"
            )

    @pytest.mark.parametrize(
        "INPUT_DATA, EXPECTED_RESULT",
        [
            (
                {"RESOURCE_KEY": "MD_OCID", EXTRA_USER_AGENT_INFO: ""},
                {
                    "USER_AGENT_VALUE": "DATASCIENCE_MODEL_DEPLOYMENT",
                    EXTRA_USER_AGENT_INFO: UNKNOWN,
                },
            ),
            (
                {"RESOURCE_KEY": "JOB_RUN_OCID", EXTRA_USER_AGENT_INFO: "test_api"},
                {
                    "USER_AGENT_VALUE": "DATASCIENCE_JOB",
                    EXTRA_USER_AGENT_INFO: "test_api",
                },
            ),
            (
                {"RESOURCE_KEY": "NB_SESSION_OCID", EXTRA_USER_AGENT_INFO: None},
                {
                    "USER_AGENT_VALUE": "DATASCIENCE_NOTEBOOK",
                    EXTRA_USER_AGENT_INFO: UNKNOWN,
                },
            ),
            (
                {
                    "RESOURCE_KEY": "DATAFLOW_RUN_ID",
                    EXTRA_USER_AGENT_INFO: "some_class&some_method",
                },
                {
                    "USER_AGENT_VALUE": "DATAFLOW",
                    EXTRA_USER_AGENT_INFO: "some_class&some_method",
                },
            ),
        ],
    )
    @patch("oci.config.validate_config")
    @patch("oci.signer.load_private_key_from_file")
    def test_user_agent_default_signer_known_resources(
        self,mock_load_key_file, mock_validate_config, monkeypatch, INPUT_DATA, EXPECTED_RESULT
    ):
        # monkeypatch = MonkeyPatch()
        monkeypatch.setenv("OCI_RESOURCE_PRINCIPAL_VERSION", "1.1")
        monkeypatch.setenv(INPUT_DATA["RESOURCE_KEY"], "1234")
        if INPUT_DATA[EXTRA_USER_AGENT_INFO] is not None:
            monkeypatch.setenv(EXTRA_USER_AGENT_INFO, INPUT_DATA[EXTRA_USER_AGENT_INFO])

        importlib.reload(ads.config)
        importlib.reload(ads)
        importlib.reload(ads.auth)
        importlib.reload(ads.telemetry)

        with patch("oci.config.from_file", return_value=self.test_config):
            auth_info = ads.auth.default_signer()
            assert (
                auth_info["config"].get("additional_user_agent")
                == f"{LIBRARY}/version={ads.__version__}#surface={EXPECTED_RESULT['USER_AGENT_VALUE']}#api={EXPECTED_RESULT[EXTRA_USER_AGENT_INFO]}"
            )
        monkeypatch.delenv("OCI_RESOURCE_PRINCIPAL_VERSION", raising=False)
        monkeypatch.delenv(INPUT_DATA["RESOURCE_KEY"], raising=False)
        monkeypatch.delenv(EXTRA_USER_AGENT_INFO, raising=False)

    @patch("oci.config.validate_config")
    @patch("oci.signer.Signer")
    def test_user_agent_default_singer_ociservice(
        self,
        mock_signer,
        mock_validate_config,
        monkeypatch,
    ):
        monkeypatch.setenv("OCI_RESOURCE_PRINCIPAL_VERSION", "1.1")

        importlib.reload(ads.config)
        importlib.reload(ads.telemetry)

        with patch("oci.config.from_file", return_value=self.test_config):
            auth_info = ads.auth.default_signer()
            assert (
                auth_info["config"].get("additional_user_agent")
                == f"{LIBRARY}/version={ads.__version__}#surface=OCI_SERVICE#api={UNKNOWN}"
            )
        monkeypatch.delenv("OCI_RESOURCE_PRINCIPAL_VERSION", raising=False)
