#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from mock import MagicMock
import pytest
from unittest import TestCase, mock

import oci
from oci.auth.signers.ephemeral_resource_principals_signer import (
    EphemeralResourcePrincipalSigner,
)
from oci.auth.signers.security_token_signer import SecurityTokenSigner
from oci.config import DEFAULT_LOCATION
import ads
from ads import set_auth
from ads.common.utils import (
    oci_key_profile,
    oci_config_location,
)
from ads.common.auth import (
    SecurityToken,
    SecurityTokenError,
    api_keys,
    resource_principal,
    security_token,
    create_signer,
    default_signer,
    get_signer,
    AuthType,
    AuthState,
    AuthFactory,
    OCIAuthContext,
    AuthContext,
)
from ads.common.oci_logging import OCILog

MOCK_CONFIG_FROM_FILE = {
    "user": "test_user",
    "fingerprint": "test_fingerprint",
    "tenancy": "test_tenancy",
    "region": "us-ashburn-1",
    "key_file": "test_key_file"
}


class TestEDAMixin(TestCase):
    @mock.patch("oci.config.validate_config")
    @mock.patch("oci.config.from_file")
    @mock.patch("os.path.exists")
    @mock.patch("oci.signer.load_private_key_from_file")
    def test_set_auth_overwrite_profile(
        self, mock_load_key_file, mock_path_exists, mock_config_from_file, mock_validate_config
    ):
        mock_config_from_file.return_value = MOCK_CONFIG_FROM_FILE
        set_auth(profile="TEST")
        default_signer()
        mock_config_from_file.assert_called_with("~/.oci/config", "TEST")
        set_auth(profile="DEFAULT")

    @mock.patch("oci.config.validate_config")
    @mock.patch("oci.config.from_file")
    @mock.patch("os.path.exists")
    @mock.patch("oci.signer.load_private_key_from_file")
    def test_set_auth_overwrite_config_location(
        self, mock_load_key_file, mock_path_exists, mock_config_from_file, mock_validate_config
    ):
        mock_config_from_file.return_value = MOCK_CONFIG_FROM_FILE
        mock_path_exists.return_value = True
        set_auth(oci_config_location="test_path")
        default_signer()
        mock_config_from_file.assert_called_with("test_path", "DEFAULT")
        set_auth()

    @mock.patch("oci.config.validate_config")
    @mock.patch("oci.config.from_file")
    @mock.patch("oci.signer.Signer")
    def test_api_keys_using_test_profile(self, mock_signer, mock_config_from_file, mock_validate_config):
        api_keys("test_path", "TEST_PROFILE")
        mock_config_from_file.assert_called_with("test_path", "TEST_PROFILE")

    @mock.patch("oci.config.validate_config")
    @mock.patch("oci.config.from_file")
    @mock.patch("oci.signer.Signer")
    def test_api_keys_using_default_profile(self, mock_signer, mock_config_from_file, mock_validate_config):
        api_keys("test_path")
        mock_config_from_file.assert_called_with("test_path", "DEFAULT")

    @mock.patch("oci.config.validate_config")
    @mock.patch("oci.auth.signers.get_resource_principals_signer")
    @mock.patch("oci.config.from_file")
    @mock.patch("oci.signer.Signer")
    def test_get_signer_with_api_keys(
        self, mock_signer, mock_config_from_file, mock_rp_signer, mock_validate_config
    ):
        get_signer("test_path", "TEST_PROFILE")
        mock_config_from_file.assert_called_with("test_path", "TEST_PROFILE")
        get_signer()
        mock_rp_signer.assert_called_once()

    @mock.patch("oci.auth.signers.get_resource_principals_signer")
    @mock.patch.dict(os.environ, {"OCI_RESOURCE_PRINCIPAL_VERSION": "2.2"})
    def test_resource_principal(self, mock_rp_signer):
        resource_principal()
        mock_rp_signer.assert_called_once()

    @mock.patch("oci.config.validate_config")
    @mock.patch("oci.signer.load_private_key")
    def test_set_auth_with_key_content(self, mock_load_private_key, mock_validate_config):
        set_auth(
            config={
                "user": "test_user",
                "fingerprint": "test_fingerprint",
                "tenancy": "test_tenancy",
                "region": "us-ashburn-1",
                "key_content": "test_key_content"
            }
        )
        signer = default_signer()
        assert signer["config"]["user"] == "test_user"
        assert signer["config"]["fingerprint"] == "test_fingerprint"
        assert signer["config"]["tenancy"] == "test_tenancy"
        assert signer["config"]["region"] == "us-ashburn-1"
        assert signer["config"]["key_content"] == "test_key_content"
        assert "additional_user_agent" in signer["config"]
        assert signer["signer"] != None
        set_auth()


class TestOCIMixin(TestCase):
    def tearDown(self) -> None:
        with mock.patch("os.path.exists"):
            ads.set_auth(AuthType.API_KEY)
            return super().tearDown()

    @mock.patch("oci.config.validate_config")
    @mock.patch("oci.config.from_file")
    @mock.patch("os.path.exists")
    @mock.patch("oci.signer.Signer")
    @mock.patch("oci.logging.LoggingManagementClient")
    def test_api_key_auth_with_logging(
        self, client, mock_signer, mock_path_exists, mock_config_from_file, mock_validate_config
    ):
        """Tests initializing OCIMixin with default auth.
        Without any explicit config, the client should be initialized from DEFAULT OCI API key config.
        """
        # Accessing the client property will trigger the authentication.
        # By default, ADS uses API key with DEFAULT profile.
        OCILog().client
        self.assertIsNotNone(
            mock_config_from_file.call_args, "OCI config not initialized from file."
        )
        args = mock_config_from_file.call_args[0]
        config_location = args[0]
        profile = args[1]
        self.assertEqual(profile, "DEFAULT", mock_config_from_file.call_args_list)
        self.assertEqual(
            os.path.abspath(os.path.expanduser(config_location)),
            os.path.abspath(os.path.expanduser("~/.oci/config")),
        )

        # Change the profile via ads.set_auth() to use a different profile
        ads.set_auth(profile="TEST")
        OCILog().client
        args = mock_config_from_file.call_args[0]
        profile = args[1]
        self.assertEqual(profile, "TEST")
        config = client.call_args[1]["config"]
        signer = client.call_args[1]["signer"]
        self.assertIsInstance(config, mock.MagicMock)
        self.assertIsInstance(signer, mock.MagicMock)

        # Pass in a customized config
        customized_config = dict(tenancy="my_tenancy")
        OCILog(config=customized_config).client
        config = client.call_args[1]["config"]
        self.assertNotIn("signer", client.call_args[1])
        self.assertEqual(config, customized_config)

        # Pass in a customized signer
        customized_signer = oci.signer.Signer("tenancy", "user", "fingerprint", "key")
        OCILog(signer=customized_signer).client
        config = client.call_args[1]["config"]
        signer = client.call_args[1]["signer"]
        self.assertEqual(config, None)
        self.assertEqual(signer, customized_signer)

    @mock.patch("oci.auth.signers.get_resource_principals_signer")
    @mock.patch("oci.config.from_file")
    @mock.patch("oci.signer.Signer")
    def test_resource_principal_auth_with_logging(
        self, mock_signer, mock_config_from_file, mock_rp_signer
    ):
        """Tests initializing OCIMixin with resource principal."""
        oci.logging.LoggingManagementClient = mock.MagicMock()
        ads.set_auth(AuthType.RESOURCE_PRINCIPAL)
        # Accessing the client property will trigger the authentication.
        OCILog().client
        self.assertIsNone(
            mock_config_from_file.call_args,
            "oci.config.from_file() should not be called.",
        )
        self.assertGreater(len(mock_rp_signer.mock_calls), 0)

    @mock.patch("oci.auth.signers.get_resource_principals_signer")
    @mock.patch("oci.config.from_file")
    @mock.patch("oci.signer.Signer")
    def test_resource_principal_auth_with_logging_with_non_existent_config(
        self, mock_signer, mock_config_from_file, mock_rp_signer
    ):
        """Tests initializing OCIMixin with resource principal."""
        oci.logging.LoggingManagementClient = mock.MagicMock()
        import random

        ads.set_auth(
            AuthType.RESOURCE_PRINCIPAL,
            oci_config_location=f"/my/{random.random()}/path",
        )
        # Accessing the client property will trigger the authentication.
        OCILog().client
        self.assertIsNone(
            mock_config_from_file.call_args,
            "oci.config.from_file() should not be called.",
        )
        self.assertGreater(len(mock_rp_signer.mock_calls), 0)

    @mock.patch("oci.auth.signers.InstancePrincipalsSecurityTokenSigner")
    @mock.patch("oci.config.from_file")
    @mock.patch("oci.signer.Signer")
    def test_instance_principal_auth_with_logging(
        self, mock_signer, mock_config_from_file, mock_ip_signer
    ):
        """Tests initializing OCIMixin with instance principal."""
        oci.logging.LoggingManagementClient = mock.MagicMock()
        ads.set_auth(AuthType.INSTANCE_PRINCIPAL)
        # Accessing the client property will trigger the authentication.
        OCILog().client
        self.assertIsNone(
            mock_config_from_file.call_args,
            "oci.config.from_file() should not be called.",
        )
        self.assertGreater(len(mock_ip_signer.mock_calls), 0)


class TestAuthFactory(TestCase):
    def tearDown(self) -> None:
        with mock.patch("os.path.exists"):
            ads.set_auth(AuthType.API_KEY)
            return super().tearDown()

    def test_set_auth_value_errors(self):
        """
        Testing ValueErrors when running set_auth()
        """

        with pytest.raises(ValueError):
            set_auth("invalid_auth_type_value")

        with pytest.raises(ValueError):
            set_auth(config={"test": "test"}, profile="TEST_PROFILE")

        with pytest.raises(ValueError):
            AuthFactory().signerGenerator("not_existing_iam_type")

    def test_register_singer(self):
        AuthFactory().register("new_singer_type", "signer_class")
        assert "new_singer_type" in AuthFactory.classes.keys()

    @mock.patch("os.path.exists")
    @mock.patch(
        "oci.config.from_file",
        return_value={
            "tenancy": "",
            "user": "",
            "fingerprint": "",
            "key_file": "",
        },
    )
    @mock.patch("oci.config.validate_config")
    @mock.patch("oci.signer.load_private_key_from_file")
    def test_api_key_create_signer(
        self, mock_load_key_file, mock_config_from_file, mock_path_exists, mock_validate_config
    ):
        """
        Testing api key setup with set_auth() and getting it with default_signer()
        """
        set_auth(AuthType.API_KEY)
        signer = default_signer(client_kwargs={"test": "test"})
        assert "fingerprint" in signer["config"]
        assert isinstance(signer["signer"], oci.signer.Signer)
        assert "test" in signer["client_kwargs"]

        set_auth(
            AuthType.API_KEY,
            config={
                "tenancy": "test_tenancy",
                "user": "test_user",
                "fingerprint": "test_fingerprint",
                "key_file": "test_key_file",
            },
        )
        signer = default_signer(client_kwargs={"test": "test"})
        assert signer["config"]["fingerprint"] == "test_fingerprint"
        assert "test" in signer["client_kwargs"]

    @mock.patch("oci.auth.signers.get_resource_principals_signer")
    @mock.patch.dict(os.environ, {"OCI_RESOURCE_PRINCIPAL_VERSION": "2.2"})
    def test_resource_principal_create_signer(self, mock_rp_signer):
        """
        Testing resource principal setup with set_auth() and getting it with default_signer()
        """
        set_auth(AuthType.RESOURCE_PRINCIPAL)
        signer = default_signer(client_kwargs={"test": "test"})
        assert "additional_user_agent" in signer["config"]
        mock_rp_signer.assert_called_once()
        assert "test" in signer["client_kwargs"]

        test_rp_signer = oci.auth.signers.get_resource_principals_signer
        set_auth(AuthType.RESOURCE_PRINCIPAL, signer_callable=test_rp_signer)
        signer = default_signer(client_kwargs={"test": "test"})
        assert "test" in signer["client_kwargs"]
        mock_rp_signer.assert_called()

        set_auth(AuthType.RESOURCE_PRINCIPAL, signer="test_signer_instance")
        signer = default_signer()
        assert signer["signer"] == "test_signer_instance"

    @mock.patch("oci.auth.signers.InstancePrincipalsSecurityTokenSigner")
    def test_instance_principal_create_signer(self, mock_ip_signer):
        """
        Testing instance principal setup with set_auth() and getting it with default_signer()
        """
        set_auth(AuthType.INSTANCE_PRINCIPAL)
        signer = default_signer(client_kwargs={"test": "test"})
        assert "additional_user_agent" in signer["config"]
        mock_ip_signer.assert_called_once()
        assert "test" in signer["client_kwargs"]

        test_ip_signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner
        set_auth(AuthType.INSTANCE_PRINCIPAL, signer_callable=test_ip_signer)
        signer = default_signer(client_kwargs={"test": "test"})
        assert "test" in signer["client_kwargs"]
        mock_ip_signer.assert_called()

        set_auth(AuthType.INSTANCE_PRINCIPAL, signer="test_signer_instance")
        signer = default_signer()
        assert signer["signer"] == "test_signer_instance"

        mock_ip_signer.mock_reset()
        test_signer_kwargs = {"test_signer_kwargs": "test_signer_kwargs"}
        signer_args = dict(signer_kwargs=test_signer_kwargs)
        signer_generator = AuthFactory().signerGenerator(AuthType.INSTANCE_PRINCIPAL)
        signer_generator(signer_args).create_signer()
        mock_ip_signer.assert_called_with(**test_signer_kwargs)

    @mock.patch("ads.common.auth.AuthFactory")
    @mock.patch("oci.auth.signers.InstancePrincipalsSecurityTokenSigner")
    def test_create_signer(self, mock_ip_signer, mock_signer_generator):

        auth = create_signer(signer="test")
        assert auth["signer"] == "test"

        test_ip_signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner
        create_signer(signer_callable=test_ip_signer)
        mock_ip_signer.assert_called()

        test_signer_kwargs = {"test_signer_kwargs": "test_signer_kwargs"}
        create_signer(signer_callable=test_ip_signer, signer_kwargs=test_signer_kwargs)
        mock_ip_signer.assert_called_with(**test_signer_kwargs)

        create_signer(auth_type="resource_principal")
        mock_signer_generator().signerGenerator.assert_called_with("resource_principal")

        create_signer(config={"test_config": "test_config"})
        mock_signer_generator().signerGenerator.assert_called_with("api_key")

    @mock.patch("oci.config.validate_config")
    @mock.patch("oci.auth.signers.InstancePrincipalsSecurityTokenSigner")
    @mock.patch("oci.auth.signers.get_resource_principals_signer")
    @mock.patch("oci.config.from_file")
    @mock.patch("oci.signer.load_private_key_from_file")
    @mock.patch("os.path.exists", value=True)
    def test_set_auth_multiple_times_with(
        self,
        mock_path_exists,
        mock_key_from_file,
        mock_config_from_file,
        mock_rp_signer,
        mock_ip_signer,
        mock_validate_config
    ):
        """
        Testing behaviour when multiple times invoked set_auth() with different parameters and validate,
        that default_signer() returns proper signer based on saved state of auth values within AuthState().
        Checking that default_signer() runs two times in a row and returns signer based on AuthState().
        """
        mock_config_from_file.return_value = MOCK_CONFIG_FROM_FILE
        config = dict(
            user="ocid1.user.oc1..<unique_ocid>",
            fingerprint="00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
            tenancy="ocid1.tenancy.oc1..<unique_ocid>",
            region="<region>",
            key_file="<path>/<to>/<key_file>",
        )
        auth_type = AuthType.API_KEY
        set_auth(auth=auth_type, config=config, client_kwargs={"timeout": 1})
        auth_state = AuthState()
        assert auth_state.oci_config
        assert auth_state.oci_config_path == oci.config.DEFAULT_LOCATION
        assert auth_state.oci_key_profile == "DEFAULT"
        assert auth_state.oci_client_kwargs == {"timeout": 1}
        signer = default_signer()
        assert signer["config"]["user"] == config["user"]
        default_signer()
        assert signer["config"]["user"] == config["user"]

        set_auth(
            auth=auth_type,
            oci_config_location="~/path_to_config",
            client_kwargs={"timeout": 2},
        )
        auth_state = AuthState()
        assert not auth_state.oci_config
        assert auth_state.oci_config_path == "~/path_to_config"
        assert auth_state.oci_key_profile == "DEFAULT"
        assert auth_state.oci_client_kwargs == {"timeout": 2}
        default_signer()
        mock_config_from_file.assert_called_with("~/path_to_config", "DEFAULT")
        default_signer()
        mock_config_from_file.assert_called_with("~/path_to_config", "DEFAULT")

        set_auth(auth=auth_type, config=config, client_kwargs={"timeout": 3})
        auth_state = AuthState()
        assert auth_state.oci_config
        assert auth_state.oci_config_path == oci.config.DEFAULT_LOCATION
        assert auth_state.oci_key_profile == "DEFAULT"
        assert auth_state.oci_client_kwargs == {"timeout": 3}
        signer = default_signer()
        assert signer["config"]["key_file"] == config["key_file"]
        signer = default_signer()
        assert signer["config"]["key_file"] == config["key_file"]

        set_auth(auth=auth_type, profile="NOT_DEFAULT")
        auth_state = AuthState()
        assert not auth_state.oci_config
        assert auth_state.oci_config_path == oci.config.DEFAULT_LOCATION
        assert auth_state.oci_key_profile == "NOT_DEFAULT"
        default_signer()
        mock_config_from_file.assert_called_with(
            oci.config.DEFAULT_LOCATION, "NOT_DEFAULT"
        )
        default_signer()
        mock_config_from_file.assert_called_with(
            oci.config.DEFAULT_LOCATION, "NOT_DEFAULT"
        )

        set_auth(config={}, signer=mock.Mock(spec=EphemeralResourcePrincipalSigner))
        signer = default_signer()
        assert isinstance(signer["signer"], EphemeralResourcePrincipalSigner)
        signer = default_signer()
        assert isinstance(signer["signer"], EphemeralResourcePrincipalSigner)

        set_auth(
            config={"test", "test"},
            signer=mock.Mock(spec=EphemeralResourcePrincipalSigner),
        )
        auth_state = AuthState()
        assert auth_state.oci_iam_type == AuthType.API_KEY

        test_ip_signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner
        test_signer_kwargs = {"test_signer_kwargs": "test_signer_kwargs"}
        set_auth(signer_callable=test_ip_signer, signer_kwargs=test_signer_kwargs)
        default_signer()
        mock_ip_signer.assert_called_with(**test_signer_kwargs)


class TestOCIAuthContext(TestCase):
    def tearDown(self) -> None:
        with mock.patch("os.path.exists"):
            ads.set_auth(AuthType.API_KEY)
            return super().tearDown()

    @mock.patch("os.path.exists")
    def test_oci_auth_context(self, mock_path_exists):
        profile = AuthState().oci_key_profile
        mode = AuthState().oci_iam_type
        with OCIAuthContext(profile="TEST"):
            assert AuthState().oci_key_profile == "TEST"
            assert AuthState().oci_iam_type == AuthType.API_KEY
        assert AuthState().oci_key_profile == profile
        assert AuthState().oci_iam_type == mode

        with OCIAuthContext():
            assert AuthState().oci_iam_type == AuthType.RESOURCE_PRINCIPAL
        assert AuthState().oci_key_profile == profile
        assert AuthState().oci_iam_type == mode


class TestAuthContext:
    def tearDown(self) -> None:
        ads.set_auth(AuthType.API_KEY)
        return super().tearDown()

    @mock.patch("os.path.exists")
    def test_auth_context_inside_auth_context(self, mock_path_exists):
        with AuthContext():
            ads.set_auth(
                signer="signer_for_the_FIRST_context", client_kwargs={"timeout": 1}
            )
            assert AuthState().oci_signer == "signer_for_the_FIRST_context"
            assert AuthState().oci_client_kwargs == {"timeout": 1}

            with AuthContext():
                ads.set_auth(
                    signer="signer_for_the_SECOND_context", client_kwargs={"timeout": 2}
                )
                assert AuthState().oci_client_kwargs == {"timeout": 2}
                assert AuthState().oci_signer == "signer_for_the_SECOND_context"
                ads.set_auth(
                    signer="ANOTHER_signer_for_the_SECOND_context",
                    client_kwargs={"timeout": 3},
                )
                assert AuthState().oci_signer == "ANOTHER_signer_for_the_SECOND_context"
                assert AuthState().oci_client_kwargs == {"timeout": 3}
            assert AuthState().oci_signer == "signer_for_the_FIRST_context"
            assert AuthState().oci_client_kwargs == {"timeout": 1}

        with AuthContext(client_kwargs={"timeout": 4}):
            assert AuthState().oci_client_kwargs == {"timeout": 4}

    def test_with_set_auth_returns_error(self):
        with pytest.raises(ValueError):
            with AuthContext(auth="not_correct_auth_type"):
                pass


class TestSecurityToken(TestCase):

    @mock.patch("oci.auth.signers.SecurityTokenSigner.__init__")
    @mock.patch("oci.signer.load_private_key_from_file")
    @mock.patch("ads.common.auth.SecurityToken._read_security_token_file")
    @mock.patch("ads.common.auth.SecurityToken._validate_and_refresh_token")
    def test_security_token(
        self,
        mock_validate_and_refresh_token, 
        mock_read_security_token_file,
        mock_load_private_key_from_file,
        mock_security_token_signer
    ):
        config = {
            "fingerprint": "test_fingerprint",
            "tenancy": "test_tenancy",
            "region": "us-ashburn-1",
            "key_file": "test_key_file",
            "generic_headers": [1,2,3],
            "body_headers": [4,5,6]
        }

        with pytest.raises(
            ValueError,
            match="Parameter `security_token_file` must be provided for using `security_token` authentication."
        ):
            signer = security_token(
                oci_config=config,
                client_kwargs={"test_client_key":"test_client_value"}
            )

        config["security_token_file"] = "test_security_token"
        mock_security_token_signer.return_value = None
        signer = security_token(
            oci_config=config,
            client_kwargs={"test_client_key":"test_client_value"}
        )

        mock_validate_and_refresh_token.assert_called_with(config)
        mock_read_security_token_file.assert_called_with("test_security_token")
        mock_load_private_key_from_file.assert_called_with("test_key_file", None)
        assert signer["client_kwargs"] == {"test_client_key": "test_client_value"}
        assert "additional_user_agent" in signer["config"]
        assert signer["config"]["fingerprint"] == "test_fingerprint"
        assert signer["config"]["tenancy"] == "test_tenancy"
        assert signer["config"]["region"] == "us-ashburn-1"
        assert signer["config"]["security_token_file"] == "test_security_token"
        assert signer["config"]["key_file"] == "test_key_file"
        assert isinstance(signer["signer"], SecurityTokenSigner)

    @mock.patch("os.system")
    @mock.patch("oci.auth.security_token_container.SecurityTokenContainer.get_jwt")
    @mock.patch("time.time")
    @mock.patch("oci.auth.security_token_container.SecurityTokenContainer.valid")
    @mock.patch("oci.auth.security_token_container.SecurityTokenContainer.__init__")
    @mock.patch("ads.common.auth.SecurityToken._read_security_token_file")
    def test_validate_and_refresh_token(
        self, 
        mock_read_security_token_file, 
        mock_security_token_container,
        mock_valid,
        mock_time,
        mock_get_jwt,
        mock_system
    ):
        security_token = SecurityToken(
            args={
                "oci_config_location": DEFAULT_LOCATION,
                "oci_key_profile": "test_profile"
            }
        )
        mock_security_token_container.return_value = None
        
        mock_valid.return_value = False
        configuration = {
            "fingerprint": "test_fingerprint",
            "tenancy": "test_tenancy",
            "region": "us-ashburn-1",
            "key_file": "test_key_file",
            "security_token_file": "test_security_token",
            "generic_headers": [1,2,3],
            "body_headers": [4,5,6]
        }
        with pytest.raises(
            SecurityTokenError,
            match="Security token has expired. Call `oci session authenticate` to generate new session."
        ):
            security_token._validate_and_refresh_token(configuration)
        
        mock_valid.return_value = True
        mock_time.return_value = 1
        mock_get_jwt.return_value = {"exp" : 1}
        mock_system.return_value = 1
        
        security_token._validate_and_refresh_token(configuration)
        
        mock_read_security_token_file.assert_called_with("test_security_token")
        mock_security_token_container.assert_called()
        mock_time.assert_called()
        mock_get_jwt.assert_called()
        mock_system.assert_called_with(f"oci session refresh --config-file {DEFAULT_LOCATION} --profile test_profile")

    @mock.patch("builtins.open")
    @mock.patch("os.path.isfile")
    def test_read_security_token_file(self, mock_isfile, mock_open):
        security_token = SecurityToken(args={})

        mock_isfile.return_value = False
        with pytest.raises(
            ValueError,
            match="Invalid `security_token_file`. Specify a valid path."
        ):
            security_token._read_security_token_file("test_security_token")

        mock_isfile.return_value = True
        security_token._read_security_token_file("test_security_token")
        mock_open.assert_called()
