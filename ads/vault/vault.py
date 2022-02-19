#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ast
import base64
import json
import oci.vault
import uuid

from oci.vault import VaultsClientCompositeOperations
from oci.vault.models import Base64SecretContentDetails
from oci.vault.models import CreateSecretDetails
from oci.vault.models import UpdateSecretDetails

import ads
from ads.common import oci_client, auth as authutil
from ads.config import NB_SESSION_COMPARTMENT_OCID

logger = ads.getLogger("ads.vault")


class Vault:
    def __init__(
        self,
        vault_id: str = None,
        key_id: str = None,
        compartment_id=None,
        secret_client_auth=None,
        vault_client_auth=None,
        auth=None,
    ):

        """
        Parameters
        ----------
        vault_id: (str, optional). Default None
            ocid of the vault
        key_id: (str, optional). Default None
            ocid of the key that is used for encrypting the content
        compartment_id: (str, optional). Default None
            ocid of the compartment_id where the vault resides. When available in
            environment variable - `NB_SESSION_COMPARTMENT_OCID`, will defult to that.
        secret_client_auth: (dict, optional, deprecated since 2.5.1). Default None.
            deprecated since 2.5.1. Use `auth` instead
        vault_client_auth: (dict, optional, deprecated since 2.5.1). Default None.
            deprecated since 2.5.1. Use `auth` instead
        auth: (dict, optional)
            Dictionay returned from ads.common.auth.api_keys() or ads.common.auth.resource_principal(). By default, will follow what is set in `ads.set_auth`.  Use this attribute to override the default.

        """
        # vault_client_auth. User should use auth to setup the authentication

        self.id = vault_id
        self.key_id = key_id
        self.client_auth = auth or authutil.default_signer()
        if secret_client_auth:
            logger.warning(
                "Deprecated use of `secret_client_auth` since version 2.5.1. Use `auth` instead."
            )
        if vault_client_auth:
            logger.warning(
                "Deprecated use of `vault_client_auth` since version 2.5.1. Use `auth` instead."
            )
        self.secret_client_auth = secret_client_auth or self.client_auth
        self.vault_client_auth = vault_client_auth or self.client_auth

        self.compartment_id = (
            NB_SESSION_COMPARTMENT_OCID if compartment_id is None else compartment_id
        )

        self.secret_client = oci_client.OCIClientFactory(
            **self.secret_client_auth
        ).secret
        self.vaults_client_composite = VaultsClientCompositeOperations(
            oci_client.OCIClientFactory(**self.vault_client_auth).vault
        )

    def create_secret(
        self,
        value: dict,
        secret_name: str = None,
        description: str = None,
        encode=True,
        freeform_tags: dict = None,
        defined_tags: dict = None,
    ) -> str:
        """
        Saves value into vault as a secret.

        Parameters
        ----------
        value: dict
            The value to store as a secret.
        secret_name: str, optional
            The name of the secret.
        description: str, optional
            The description of the secret.
        encode: (bool, optional). Default True
            Whether to encode using the default encoding.
        freeform_tags: (dict, optional). Default None
            freeform_tags as defined by the oci sdk
        defined_tags: (dict, optional). Default None
            defined_tags as defined by the oci sdk

        Returns
        -------
        The secret ocid that correspond to the value saved as a secret into vault.
        """
        if not isinstance(self.compartment_id, str):
            raise ValueError("compartment_id needs to be a string.")

        if self.compartment_id is None:
            raise ValueError("compartment_id needs to be specified.")

        if self.id is None:
            raise ValueError("vault_id needs to be specified in the constructor.")
        if self.key_id is None:
            raise ValueError("key_id needs to be specified in the constructor.")
        # Encode the secret.
        secret_content_details = self._encode_secret(value, encode=encode)

        # Bundle the secret and metadata about it.
        secrets_details = CreateSecretDetails(
            compartment_id=self.compartment_id,
            description=description
            if description is not None
            else "Data Science service secret",
            secret_content=secret_content_details,
            secret_name=secret_name
            if secret_name is not None
            else "DataScienceSecret_{}".format(str(uuid.uuid4())[-6:]),
            vault_id=self.id,
            key_id=self.key_id,
            freeform_tags=freeform_tags,
            defined_tags=defined_tags,
        )

        # Store secret and wait for the secret to become active.
        secret = self.vaults_client_composite.create_secret_and_wait_for_state(
            create_secret_details=secrets_details,
            wait_for_states=[oci.vault.models.Secret.LIFECYCLE_STATE_ACTIVE],
        ).data
        return secret.id

    def update_secret(
        self,
        secret_id: str,
        secret_content: dict,
        encode: bool = True,
    ) -> str:
        """
        Updates content of a secret.

        Parameters
        ----------
        secret_id: str
            The secret id where the stored secret will be updated.
        secret_content: dict,
            The updated content.
        encode: (bool, optional). Default True
            Whether to encode the secret_content using default encoding

        Returns
        -------
        The secret ocid with updated content.
        """

        if not isinstance(self.compartment_id, str):
            raise ValueError("compartment_id needs to be a string.")

        if self.compartment_id is None:
            raise ValueError("compartment_id needs to be specified.")

        if self.id is None:
            raise ValueError("vault_id needs to be specified in the constructor.")
        if self.key_id is None:
            raise ValueError("key_id needs to be specified in the constructor.")
        # Encode the secret.
        secret_content_details = (
            self._encode_secret(secret_content) if encode else secret_content
        )

        # Store the details to update.
        secrets_details = UpdateSecretDetails(
            secret_content=secret_content_details,
        )

        # Create new secret version and wait for the new version to become active.
        secret_update = self.vaults_client_composite.update_secret_and_wait_for_state(
            secret_id,
            secrets_details,
            wait_for_states=[oci.vault.models.Secret.LIFECYCLE_STATE_ACTIVE],
        ).data

        return secret_update.id

    def get_secret(self, secret_id: str, decoded=True) -> dict:
        """
        Retrieve secret content based on the secret ocid provided

        Parameters
        ----------
        secret_id: str
            The secret ocid.
        decoded: (bool, optional). Default True
            Whether to decode the content that is retrieved from vault service using the default decoder.
        Returns
        -------
        The secret content as a dictionary.
        """
        secret_bundle = self.secret_client.get_secret_bundle(secret_id)
        if decoded:
            secret_content = self._secret_to_dict(
                secret_bundle.data.secret_bundle_content.content
            )
            return ast.literal_eval(secret_content)
        else:
            return secret_bundle.data.secret_bundle_content.content

    def _encode_secret(self, secret_content, encode=True):
        secret_content_details = Base64SecretContentDetails(
            content_type=oci.vault.models.SecretContentDetails.CONTENT_TYPE_BASE64,
            stage=oci.vault.models.SecretContentDetails.STAGE_CURRENT,
            content=self._dict_to_secret(secret_content) if encode else secret_content,
        )
        return secret_content_details

    @staticmethod
    def _dict_to_secret(values):
        return base64.b64encode(json.dumps(values).encode("ascii")).decode("ascii")

    @staticmethod
    def _secret_to_dict(secret_content):
        return base64.b64decode(secret_content.encode("ascii")).decode("ascii")
