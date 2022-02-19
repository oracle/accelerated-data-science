#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import ads
from ads.secrets import SecretKeeper, Secret
from dataclasses import dataclass


@dataclass
class AuthToken(Secret):
    """
    AuthToken dataclass holds `auth_token` attribute
    """

    auth_token: str


class AuthTokenSecretKeeper(SecretKeeper):
    """
    `AuthTokenSecretKeeper` uses `ads.secrets.auth_token.AuthToken` class to manage Auth Token credentials.
    The credentials are stored in Vault as a dictionary with the following format - `{"auth_token":"user provided value"}`

    Examples
    --------

    >>> from ads.secrets.auth_token import AuthTokenSecretKeeper
    >>> import ads
    >>> ads.set_auth("resource_principal") #If using resource principal for authentication
    >>> # Save Auth Tokens or Acess Keys to the vault
    >>>
    >>>
    >>> authtoken2 = AuthTokenSecretKeeper(vault_id=vault_id,
    ...            key_id=key_id,
    ...            auth_token="<your auth token>").save("my_xyz_auth_token2",
    ...                                                                 "This is my auth token for git repo xyz",
    ...                                                                 freeform_tags={"gitrepo":"xyz"})
    >>> authtoken2.export_vault_details("my_git_token_vault_info.yaml", format="yaml")
    >>> # Loading credentials
    >>> with AuthTokenSecretKeeper.load_secret(source="ocid1.vaultsecret.oc1..<unique_ID>",
    ...                                export_prefix="mygitrepo",
    ...                                export_env=True
    ...                               ) as authtoken:
    ...     import os
    ...     print("Credentials inside environment variable:", os.environ.get('mygitrepo.auth_token'))
    ...     print("Credentials inside `authtoken` object: ", authtoken)
    Credentials inside environment variable: <your auth token>
    Credentials inside `authtoken` object:  {'auth_token': '<your auth token>'}
    >>> print("Credentials inside `authtoken` object: ", authtoken)
    Credentials inside `authtoken` object:  {'auth_token': None}
    >>> print("Credentials inside environment variable:", os.environ.get('mygitrepo.auth_token'))
    Credentials inside environment variable: None

    """

    def __init__(self, auth_token=None, **kwargs):
        """
        Parameters
        ----------
        auth_token: (str, optional). Default None
            auth token string that needs to be stored in the vault
        kwargs:
            vault_id: str. OCID of the vault where the secret is stored. Required for saving secret.
            key_id: str. OCID of the key used for encrypting the secret. Required for saving secret.
            compartment_id: str. OCID of the compartment where the vault is located. Required for saving secret.
            auth: dict. Dictionay returned from ads.common.auth.api_keys() or ads.common.auth.resource_principal(). By default, will follow what is set in `ads.set_auth`.  Use this attribute to override the default.

        """
        self.data = AuthToken(auth_token)
        super().__init__(**kwargs)

    def decode(self) -> "ads.secrets.auth_token.AuthTokenSecretKeeper":
        """
        Converts the content in `self.encoded` to `AuthToken` and stores in `self.data`

        Returns
        -------
        AuthTokenSecretKeeper:
            Returns the self object after decoding `self.encoded` and updates `self.data`
        """
        content = json.loads(self._decode())
        self.data = AuthToken(**content)
        return self
