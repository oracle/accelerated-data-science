#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ads
from ads.secrets import SecretKeeper, Secret
import json
import os
import tempfile
import zipfile
from tqdm.auto import tqdm

logger = ads.getLogger("ads.secrets")

from dataclasses import dataclass, field


@dataclass
class ADBSecret(Secret):
    """
    Dataclass representing the attributes managed and serialized by ADBSecretKeeper
    """

    user_name: str
    password: str
    service_name: str
    wallet_location: str = field(
        default=None, metadata={"serializable": False}
    )  # Not saved in vault
    wallet_file_name: str = field(
        default=None, repr=False
    )  # Not exposed through environment or `to_dict` function
    wallet_content: dict = field(
        default=None,
        repr=False,
        metadata={"serializable": False},  # Not saved in vault and not saved in vault
    )
    wallet_secret_ids: list = field(
        repr=False, default_factory=list
    )  # Not exposed through environment or `to_dict` function

    def __post_init__(self):
        self.wallet_file_name = (
            os.path.basename(self.wallet_location)
            if self.wallet_location
            else self.wallet_file_name
        )


class ADBSecretKeeper(SecretKeeper):
    """
    `ADBSecretKeeper` provides an interface to save ADW/ATP database credentials.
    This interface does not store the wallet file by default.
    For saving wallet file, set `save_wallet=True` while calling `ADBSecretKeeper.save` method.


    Examples
    --------
    >>> # Saving credentials without saving the wallet file
    >>> from ads.secrets.adw import ADBSecretKeeper
    >>> vault_id = "ocid1.vault.oc1..<unique_ID>"
    >>> key_id = "ocid1.key..<unique_ID>"

    >>> import ads
    >>> ads.set_auth("resource_principal") # If using resource principal for authentication
    >>> connection_parameters={
    ...     "user_name":"admin",
    ...     "password":"<your password>",
    ...     "service_name":"service_name_{high|low|med}",
    ...     "wallet_location":"/home/datascience/Wallet_xxxx.zip"
    ... }
    >>> adw_keeper = ADBSecretKeeper(vault_id=vault_id, key_id=key_id, **connection_parameters)
    >>> adw_keeper.save("adw_employee", "My DB credentials", freeform_tags={"schema":"emp"}) # Does not save the wallet file
    >>> print(adw_keeper.secret_id) # Prints the secret_id of the stored credentials
    >>> adw_keeper.export_vault_details("adw_employee_att.json", format="json") # Save the secret id and vault info to a json file

    >>> # Loading credentails
    >>> import ads
    >>> ads.set_auth("resource_principal") # If using resource principal for authentication
    >>> from ads.secrets.adw import ADBSecretKeeper
    >>> secret_id = "ocid1.vaultsecret.oc1..<unique_ID>"
    >>> with ADBSecretKeeper.load_secret(source=secret_id,
                                 wallet_location='/home/datascience/Wallet_xxxxxx.zip') as adw_creds:
    ...     import pandas as pd
    ...     df = pd.DataFrame.ads.read_sql("select * from EMPLOYEE", connection_parameters=adw_creds)


    >>> myadw_creds = ADBSecretKeeper.load_secret(source='adw_employee_att.json', format="json"
    ...                          wallet_location='/home/datascience/Wallet_xxxxxx.zip')
    >>> pd.DataFrame.ads.read_sql("select * from ATTRITION_DATA", connection_parameters=myadw_creds.to_dict()).head(2)

    >>> # Saving and loading credentials with wallet storage
    >>> # Saving credentials
    >>> from ads.secrets.adw import ADBSecretKeeper
    >>> vault_id = "ocid1.vault.oc1..<unique_ID>"
    >>> key_id = "ocid1.key.oc1..<unique_ID>"

    >>> import ads
    >>> ads.set_auth("resource_principal") # If using resource principal for authentication
    >>> connection_parameters={
    ...     "user_name":"admin",
    ...     "password":"<your password>",
    ...     "service_name":"service_name_{high|low|med}",
    ...     "wallet_location":"/home/datascience/Wallet_xxxx.zip"
    ... }
    >>> adw_keeper = ADBSecretKeeper(vault_id=vault_id, key_id=key_id, **connection_parameters)
    >>> adw_keeper.save("adw_employee", "My DB credentials", freeform_tags={"schema":"emp"}, save_wallet=True)
    >>> print(adw_keeper.secret_id) # Prints the secret_id of the stored credentials
    >>> adw_keeper.export_vault_details("adw_employee_att.json") # Save the secret id and vault info to a json file

    >>> # Loading credentails
    >>> import ads
    >>> ads.set_auth("resource_principal") # If using resource principal for authentication
    >>> from ads.secrets.adw import ADBSecretKeeper
    >>> secret_id = "ocid1.vaultsecret.oc1..<unique_ID>"
    >>> with ADBSecretKeeper.load_secret(source=secret_id) as adw_creds:
    ...     import pandas as pd
    ...     df = pd.DataFrame.ads.read_sql("select * from EMPLOYEE", connection_parameters=adw_creds)


    >>> myadw_creds = ADBSecretKeeper.load_secret(source='adw_employee_att.json', format='json')
    >>> pd.DataFrame.ads.read_sql("select * from ATTRITION_DATA", connection_parameters=myadw_creds.to_dict()).head(2)
    """

    def __init__(
        self,
        user_name: str = None,
        password: str = None,
        service_name: str = None,
        wallet_location: str = None,
        wallet_dir: str = None,
        repository_path: str = None,
        repository_key: str = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        user_name: (str, optioanl). Default None
            user_name of the databse
        password: (str, optional). Default None
            password  for connecting to the database
        service_name: (str, optional). Default None
            service name of the ADB instance
        wallet_location: (str, optional). Default None
            full path to the wallet zip file used for connecting to ADB instance.
        wallet_dir: (str, optional). Default None
            local directory where the extracted wallet content is saved
        repository_path: (str, optional). Default None.
            Path to credentials repository. For more details refer `ads.database.connection`
        repository_key: (str, optional). Default None.
            Configuration key for loading the right configuration from repository. For more details refer `ads.database.connection`
        kwargs:
            vault_id: str. OCID of the vault where the secret is stored. Required for saving secret.
            key_id: str. OCID of the key used for encrypting the secret. Required for saving secret.
            compartment_id: str. OCID of the compartment where the vault is located. Required for saving secret.
            auth: dict. Dictionay returned from ads.common.auth.api_keys() or ads.common.auth.resource_principal(). By default, will follow what is set in `ads.set_auth`.  Use this attribute to override the default.
        """
        if repository_path and repository_key:
            from ads.database import connection

            config_from_repo = connection.get_repository(
                repository_key, repository_path
            )
            if user_name is None:
                user_name = config_from_repo.get("username") or config_from_repo.get(
                    "user_name"
                )
            if password is None:
                password = config_from_repo.get("password")
            if service_name is None:
                service_name = config_from_repo.get("service_name")
            if wallet_location is None:
                wallet_location = config_from_repo.get("wallet_location")

        self.data = ADBSecret(
            user_name=user_name,
            password=password,
            service_name=service_name,
            wallet_location=wallet_location,
        )
        self.wallet_dir = wallet_dir

        super().__init__(**kwargs)

    def encode(
        self, serialize_wallet: bool = False
    ) -> "ads.secrets.adb.ADBSecretKeeper":
        """
        Prepares content to save in vault. The user_name, password and service_name and the individual files inside the wallet zip file are base64 encoded and stored in `self.secret`

        Parameters
        ----------
        serialize_wallet: bool, optional
            When set to True, loads the wallet zip file and encodes the content of each file in the zip file.

        Returns
        -------
        ADBSecretKeeper:
            Returns self object
        """

        if serialize_wallet:
            if not self.data.wallet_location:
                raise ValueError(
                    "Missing path to wallet zip file. Required wallet zip file path to be set in `wallet_location` "
                )
            with tempfile.TemporaryDirectory() as tmpdir:
                zipfile.ZipFile(self.data.wallet_location).extractall(tmpdir)
                wallet_info = {}
                with tqdm(os.listdir(tmpdir), leave=False) as pbar:
                    for file in pbar:
                        pbar.set_description(f"Encoding: {file}")
                        with open(os.path.join(tmpdir, file), "rb") as f:
                            encoded = self._encode(f.read())
                            wallet_info[file] = encoded
                self.wallet_content = wallet_info
                self.secret = {
                    **self.data.serialize(),
                    "wallet_content": self.wallet_content,
                }
                self.encoded = self.secret
        else:
            self.data.wallet_file_name = None
            super().encode()
        return self

    def decode(self) -> "ads.secrets.adb.ADBSecretKeeper":
        """
        Converts the content in `self.secret` to `ADBSecret` and stores in `self.data`

        If the `wallet_location` is passed through the constructor, then retain it. We do not want to override what user has passed in
        If the `wallet_location` was not passed, but the sercret has `wallet_secret_ids`, then we generate the wallet zip file in the location specified by `wallet_dir` in the constructor

        Returns
        -------
        ADBSecretKeeper:
            Returns self object
        """
        content = json.loads(self._decode())
        logger.debug(f"Decoded secret contains following keys - {list(content.keys())}")
        data = ADBSecret(
            **content,
        )

        if hasattr(self, "data") and self.data.wallet_location:
            # If the wallet location is passed in as a keyword argument inside
            # `load_secret` method, then we need to retain that information after
            # the secret was deserialized.
            logger.debug(f"Setting wallet file to {self.data.wallet_location}")
            data.wallet_location = self.data.wallet_location
        elif data.wallet_secret_ids and len(data.wallet_secret_ids) > 0:
            logger.debug(f"Secret ids corresponding to the wallet files found.")
            # If the secret ids for wallet files are available in secret, then we
            # can generate the wallet file.

            if self.wallet_dir:
                os.makedirs(self.wallet_dir, exist_ok=True)
            data.wallet_location = (
                os.path.join(self.wallet_dir or "", data.wallet_file_name)
                if data.wallet_file_name
                else None
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                wallet_files = []
                with tqdm(data.wallet_secret_ids, leave=False) as pbar:
                    for wsec_id in pbar:
                        pbar.set_description(f"Fetching secret content for {wsec_id}")
                        content = json.loads(
                            self._decode(self.get_secret(wsec_id, decoded=False))
                        )
                        wallet_files.append(os.path.join(tmpdir, content["filename"]))
                        with open(
                            os.path.join(tmpdir, content["filename"]), "wb"
                        ) as wf:
                            wf.write(
                                self._decode(content["content"], str_encoding=None)
                            )
                logger.debug(f"Creating wallet zip file at {data.wallet_location}")
                with zipfile.ZipFile(data.wallet_location, "w") as wzipf:
                    for file in wallet_files:
                        wzipf.write(file, os.path.basename(file))
                logger.debug(
                    f"Wallet zip file create successully at {data.wallet_location}"
                )
        else:
            logger.info(
                "Wallet information not found. If you have a local wallet zip file, you can pass that as a keyword argument `wallet_location=<path to the wallet zip file>` inside the `ADBSecretKeeper.load_secret` method"
            )
        self.data = data
        return self

    def save(
        self,
        name: str,
        description: str,
        freeform_tags: dict = None,
        defined_tags: dict = None,
        save_wallet: bool = False,
    ) -> "ads.secrets.adb.ADBSecretKeeper":
        """Saves credentials to Vault and returns self.

        Parameters
        ----------
        name : str
            Name of the secret when saved in the Vault.
        description : str
            Description of the secret when saved in the Vault.
        freeform_tags : (dict, optional). Default is None
            freeform_tags to be used for saving the secret in OCI console.
        defined_tags: (dict, optional). Default is None
            Save the tags under predefined tags in OCI console.
        save_wallet: (bool, optional). Default is False
                If set to True, saves the contents of the wallet file as separate secret.

        Returns
        -------
        ADBSecretKeeper:
            Returns self object
        """
        if not save_wallet:
            logger.info(
                "`save_wallet` set to False. Not saving wallet file content to vault"
            )
        logger.debug(f"Encoding secrets. Save wallet is set to {save_wallet}")
        self.encode(serialize_wallet=save_wallet)
        secret_id_list = []
        if save_wallet:
            with tqdm(self.encoded["wallet_content"], leave=False) as pbar:
                for file in pbar:
                    pbar.set_description(f"Saving {file}")
                    secret_id = self.create_secret(
                        self._encode(
                            json.dumps(
                                {
                                    "filename": file,
                                    "content": self.encoded["wallet_content"][file],
                                }
                            ).encode("utf-8")
                        ),
                        encode=False,
                        secret_name=f"{name}_{file}",
                        description=f"{description}\n {file}",
                        freeform_tags=freeform_tags,
                        defined_tags=defined_tags,
                    )
                    secret_id_list.append(secret_id)

        self.data.wallet_secret_ids = secret_id_list
        with tqdm(
            [1], desc="Saving credentials and file secret ids", leave=False
        ) as pbar:
            for _ in pbar:
                self.secret_id = self.create_secret(
                    self._encode(json.dumps(self.data.serialize()).encode("utf-8")),
                    encode=False,
                    secret_name=name,
                    description=description,
                    freeform_tags=freeform_tags,
                    defined_tags=defined_tags,
                )
        return self
