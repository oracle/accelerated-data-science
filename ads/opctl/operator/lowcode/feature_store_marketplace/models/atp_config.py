from typing import Optional, List

from ads.opctl.operator.lowcode.feature_store_marketplace.models.serializable_yaml_model import (
    SerializableYAMLModel,
)


class ATPConfig(SerializableYAMLModel):
    yaml_mapping = {
        "tnsNetServiceName": "tns_net_service_name",
        "walletLocation": "wallet_location",
        "walletFiles": "wallet_files",
        "vaultOcid": "vault_ocid",
        "usernameSecretName": "username_secret_name",
        "passwordSecretName": "password_secret_name",
        "walletPasswordSecretName": "wallet_password_secret_name",
    }

    def __init__(self):
        self._tns_net_service_name: Optional[str] = None
        self._wallet_location: Optional[str] = None
        self._vault_ocid: Optional[str] = None
        self._username_secret_name: Optional[str] = None
        self._password_secret_name: Optional[str] = None
        self._wallet_password_secret_name: Optional[str] = None
        self._wallet_files: Optional[List[str]] = [
            "README",
            "cwallet.sso",
            "ewallet.p12",
            "keystore.jks",
            "ojdbc.properties",
            "tnsnames.ora",
            "truststore.jks",
            "sqlnet.ora",
            "ewallet.pem",
        ]

    @property
    def tns_net_service_name(self) -> str:
        return self._tns_net_service_name

    @tns_net_service_name.setter
    def tns_net_service_name(self, tns_net_service_name: str):
        self._tns_net_service_name = tns_net_service_name

    @property
    def wallet_location(self) -> str:
        return self._wallet_location

    @wallet_location.setter
    def wallet_location(self, wallet_location: str):
        self._wallet_location = wallet_location

    @property
    def vault_ocid(self) -> str:
        return self._vault_ocid

    @vault_ocid.setter
    def vault_ocid(self, vault_ocid: str):
        self._vault_ocid = vault_ocid

    @property
    def username_secret_name(self) -> str:
        return self._username_secret_name

    @username_secret_name.setter
    def username_secret_name(self, username_secret_name: str):
        self._username_secret_name = username_secret_name

    @property
    def password_secret_name(self) -> str:
        return self._password_secret_name

    @password_secret_name.setter
    def password_secret_name(self, password_secret_name: str):
        self._password_secret_name = password_secret_name

    @property
    def wallet_password_secret_name(self) -> str:
        return self._wallet_password_secret_name

    @wallet_password_secret_name.setter
    def wallet_password_secret_name(self, wallet_password_secret_name: str):
        self._wallet_password_secret_name = wallet_password_secret_name

    @property
    def wallet_files(self) -> List[str]:
        return self._wallet_files

    @wallet_files.setter
    def wallet_files(self, wallet_files: List[str]):
        self._wallet_files = wallet_files
