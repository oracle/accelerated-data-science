#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import Optional


from ads.common.extended_enum import ExtendedEnum

from ads.opctl.operator.lowcode.feature_store_marketplace.models.serializable_yaml_model import (
    SerializableYAMLModel,
)


class MySqlConfig(SerializableYAMLModel):
    class MySQLAuthType(ExtendedEnum):
        VAULT = "VAULT"
        BASIC = "BASIC"

    class VaultConfig(SerializableYAMLModel):
        yaml_mapping = {"vaultOcid": "vault_ocid", "secretName": "secret_name"}

        def __init__(self):
            self._vault_ocid: Optional[str] = None
            self._secret_name: Optional[str] = None

        @property
        def vault_ocid(self) -> str:
            return self._vault_ocid

        @vault_ocid.setter
        def vault_ocid(self, vault_ocid: str):
            self._vault_ocid = vault_ocid

        @property
        def secret_name(self):
            return self._secret_name

        @secret_name.setter
        def secret_name(self, secret_name: str):
            self._secret_name = secret_name

    class BasicConfig(SerializableYAMLModel):
        yaml_mapping = {
            "password": "password",
        }

        def __init__(self):
            self._password: Optional[str] = None

        @property
        def password(self) -> str:
            return self._password

        @password.setter
        def password(self, password: str):
            self._password = password

    yaml_mapping = {
        "authType": "auth_type",
        "jdbcURL": "url",
        "username": "username",
        "basic": "basic_config",
        "vault": "vault_config",
    }

    def __init__(self):
        self._url: Optional[str] = None
        self._username: Optional[str] = None
        self._vault_config: Optional[MySqlConfig.VaultConfig] = None
        self._basic_config: Optional[MySqlConfig.BasicConfig] = None
        self._auth_type: Optional[MySqlConfig.MySQLAuthType] = None

    @property
    def url(self) -> str:
        return self._url

    @url.setter
    def url(self, url: str):
        self._url = url

    @property
    def username(self) -> str:
        return self._username

    @username.setter
    def username(self, username: str):
        self._username = username

    @property
    def basic_config(self) -> BasicConfig:
        return self._basic_config

    @basic_config.setter
    def basic_config(self, basic_config: BasicConfig):
        self._basic_config = basic_config

    @property
    def vault_config(self) -> VaultConfig:
        return self._vault_config

    @vault_config.setter
    def vault_config(self, vault_config: VaultConfig):
        self._vault_config = vault_config

    @property
    def auth_type(self) -> MySQLAuthType:
        if self._auth_type is None:
            if self.basic_config is None and self.vault_config is not None:
                return self.MySQLAuthType.VAULT
            else:
                return self.MySQLAuthType.BASIC
        else:
            return self._auth_type

    @auth_type.setter
    def auth_type(self, auth_type: MySQLAuthType):
        self._auth_type = auth_type
