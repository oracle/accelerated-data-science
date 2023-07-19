#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ads
from ads.vault import Vault
from base64 import b64encode, b64decode
import os
import json
from contextlib import ContextDecorator
from tqdm.auto import tqdm
import yaml
import fsspec
from dataclasses import dataclass, fields, asdict
from typing import Union

logger = ads.getLogger("ads.secrets")


@dataclass
class Secret:
    """Base class

    Methods
    -------
    serialize(self) -> dict
        Serializes attributes as dictionary. Returns dictionary with the keys that are serializable.
    to_dict(self) -> dict
        returns dictionarry with the keys that has `repr` set to True and the value is not None or empty
    export_dict -> dict
        returns dictionary with the keys that has `repr` set tp True
    export_options -> dcit
        returns list of attributes with the fields that has `repr` set to True
    """

    def serialize(self) -> dict:
        """Serializes attributes as dictionary. An attribute can be marked as not serializable by using
        `metadata` field of the `field` constructor provided by the dataclasses module.

        Returns
        -------
        dict
            returns dictionay of key/value pair where the value of the attribute
            is not None and not empty and the field does not have `metadata` = {"serializable":False}.
            Refer dataclass python documentation for more details about `metadata`
        """
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if ("serializable" not in f.metadata or f.metadata["serializable"])
            and getattr(self, f.name)
        }

    def to_dict(self) -> dict:
        """Serializes attributes as dictionary. Returns only non empty attributes.

        Returns
        -------
        dict
            returns dictionary of key/value pair where the value of the attribute is not None or empty
        """
        return asdict(self, dict_factory=lambda x: {k: v for (k, v) in x if v})

    def export_options(self) -> list:
        """Returns list of attributes that have `repr=True`.

        Returns
        -------
        list
            returns list of fields that does not have `repr=False`
        """
        return [f.name for f in fields(self) if f.repr]

    def export_dict(self) -> dict:
        """Serializes attributes as dictionary.

        Returns
        -------
        dict
            returns dictionary of key/value pair where the value of the attribute is
            not None and the field does not have `repr`=`False`
        """
        return {k: getattr(self, k) for k in self.export_options()}


class SecretKeeper(Vault, ContextDecorator):

    """
    SecretKeeper defines APIs required to serialize and deserialize secrets. Services
    such as Database, Streaming, and Git require users to provide credentials.
    These credentials need to be safely accessed at runtime.
    OCI Vault provides a mechanism for safe storage and access. SecretKeeper uses
    OCI Vault as a backend to store and retrieve the credentials.

    The exact data structure of the credentials varies from service to service.
    """

    required_keys = ["secret_id"]

    def __init__(
        self,
        content: bytes = None,
        encoded: str = None,
        secret_id: str = None,
        export_prefix: str = "",
        export_env: bool = False,
        **kwargs,
    ):
        self.encoded = encoded
        self.secret_id = secret_id
        self.content = content
        self.export_prefix = export_prefix
        self.export_env = export_env

        super().__init__(**kwargs)

    def _encode(self, content: bytes = None, str_encoding="utf-8"):
        return b64encode(content).decode(str_encoding) if content else None

    def _decode(self, content=None, str_encoding="utf-8"):
        value = b64decode(content if content else self.encoded)
        return value.decode(str_encoding) if str_encoding else value

    def encode(self):
        """
        Stores the secret in `self.secret` by calling `serialize` method on self.data.
        Stores base64 encoded string of `self.secret` in `self.encoded`.
        """
        if not hasattr(self, "data"):
            raise ValueError("No payload to encode")
        self.secret = self.data.serialize()
        self.encoded = self._encode(json.dumps(self.secret).encode("utf-8"))
        return self

    def decode(self) -> "ads.secrets.SecretKeeper":
        """
        Decodes the content in self.encoded and sets the vaule in self.secret.
        """
        self.secret = self._decode()
        return self

    def save(
        self,
        name: str,
        description: str,
        freeform_tags: dict = None,
        defined_tags: dict = None,
    ) -> "ads.secrets.SecretKeeper":
        """Saves credentials to Vault and returns self.

        Parameters
        ----------
        name : str
            Name of the secret when saved in the Vault.
        description : str
            Description of the secret when saved in the Vault.
        freeform_tags : dict, optional
            freeform_tags to be used for saving the secret in OCI console.
        defined_tags: dict, optional.
            Save the tags under predefined tags in OCI console.

        Returns
        -------
        SecretKeeper:
            Returns self object.
        """
        with tqdm(total=2, leave=False) as pbar:
            pbar.set_description("Encoding credentials")
            self.encode()
            pbar.update()
            pbar.set_description("Saving secrets to Vault")
            logger.debug(
                f"Saving to vault with following information - Vault ID: {self.id}, Key ID: {self.key_id}, name: {name}, description:{description}, freeform_tags:{freeform_tags}, defined_tags:{defined_tags}"
            )
            self.secret_id = self.create_secret(
                self.encoded,
                encode=False,
                secret_name=name,
                description=description,
                freeform_tags=freeform_tags,
                defined_tags=defined_tags,
            )
            pbar.update()
            logger.info(f"Saved secret. Secret ID: {self.secret_id}")
        return self

    @classmethod
    def load_secret(
        cls,
        source: str,
        format: str = "ocid",
        export_env: bool = False,
        export_prefix: str = "",
        auth=None,
        storage_options: dict = None,
        **kwargs,
    ) -> Union[dict, "ads.secrets.SecretKeeper"]:
        """Loads secret from vault using secret_id.

        Parameters
        ----------
        source : str
            Source could be one of the following:

            - OCID of the secret that has the secret content.
            - file path that is json or yaml format with the key - `secret_id: ocid1.vaultsecret..<unique_ID>`

        format : str
            Defult is `ocid`. When `ocid`, the source must be a secret id
            Value values:

            - `ocid` - source is expected to be ocid of the secret
            - `yaml` or `yml` - source is expected to be a path to a valid yaml file
            - `json` - source is expected to be a path to a valid json file

        export_env : str, Default False
            When set to true, the credentails will be exported to the environment variable.
            When `load_secret` is invoked using `with` statement, information exported as
            environment variable is unset before leaving the `with` scope
        export_prefix: str, Default ""
            Prefix to the environment variable that is exported.
        auth: dict, optional
            By default authentication will follow what is configured using ads.set_auth API.
            Accepts dict returned from `ads.common.auth.api_keys()` or
            `ads.common.auth.resource_principal()`.
        storage_options: dict, optional
            storage_options dict as required by `fsspec` library
        kwargs:
            key word arguments accepted by the constructor of the class
            from which this method is invoked.

        Returns
        -------
        dict:
            When called from within `with` block, Returns a dictionary containing the secret
        ads.secrets.SecretKeeper:
            When called without using `with` operator.

        Examples
        --------

        >>> from ads.secrets import APIKeySecretKeeper
        >>> with APIKeySecretKeeper.load_secret(source="ocid1.vaultsecret.**<unique_ID>**",
        ...                         export_prefix="mykafka",
        ...                         export_env=True
        ...                        ) as apisecret:
        ...     import os
        ...     print("Credentials inside environment variable:",
        ...             os.environ.get('mykafka.api_key'))
        ...     print("Credentials inside `apisecret` object: ", apisecret)
        Credentials inside environment variable: <your api key>
        Credentials inside `apisecret` object:  {'api_key': 'your api key'}

        >>> from ads.secrets import ADBSecretKeeper
        >>> with ADBSecretKeeper.load_secret("ocid1.vaultsecret.**<unique_ID>**") as adw_creds2:
        ...     import pandas as pd
        ...     df2 = pd.DataFrame.ads.read_sql("select * from ATTRITION_DATA",
        ...                 connection_parameters=adw_creds2)
        ...     print(df2.head(2))
                    JOBFUNCTION ATTRITION
        0  Product Management        No
        1  Software Developer        No
        """
        logger.info("Fetching secret from Vault..")
        if format.lower() == "ocid":
            secret_id = source
        else:
            vault_info = {}
            uri = source
            with fsspec.open(uri, storage_options=storage_options) as vf:
                if format.lower() == "json":
                    vault_info = json.load(vf)
                elif format.lower() in ["yaml", "yml"]:
                    vault_info = yaml.load(vf, Loader=yaml.FullLoader)
                if not cls._validate_required_vault_attributes(vault_info):
                    logger.error(
                        f"Missing required Attributes in file {uri}: {cls.required_keys}"
                    )
                    raise ValueError(
                        f"The file: {uri} does not contain all the required attributes - {','.join(cls.required_keys)}."
                    )
                secret_id = vault_info["secret_id"]
        logger.debug(f"Fetch with details: secret_id:{secret_id}")
        secret = cls(auth=auth).get_secret(secret_id, decoded=False)
        logger.info("Fetched secret from Vault successfully")
        return cls(
            secret_id=secret_id,
            encoded=secret,
            export_env=export_env,
            export_prefix=export_prefix,
            **kwargs,
        ).decode()

    @classmethod
    def _validate_required_vault_attributes(cls, vault_info):
        return len([k for k in cls.required_keys if k in vault_info]) == len(
            cls.required_keys
        )

    def __enter__(self, *args, **kwargs):
        self.exported_keys = []
        if self.export_env:
            secret_data = self.data.export_dict()
            export_dict = {
                f"{self.export_prefix}.{key}"
                if self.export_prefix
                else key: secret_data[key]
                for key in secret_data
                if secret_data[key]
            }

            # Validate if the keys are already in the environment variable. If they are there
            # Throw an error and ask user to provide a prefix

            for key in export_dict:
                if os.environ.get(key):
                    logger.error(
                        f"Environment variable {key} already exists. Please consider providing `export_prefix` to avoid name collision."
                    )
                    raise Exception(
                        f"Environment variable {key} already exists. Please consider providing `export_prefix` to avoid name collision."
                    )
            os.environ.update(export_dict)
            self.exported_keys = list(export_dict.keys())
            logger.debug(
                f"Exporting credentials to environment. Keys: {self.exported_keys}"
            )
        else:
            logger.debug(f"Skipping Export credentials to environment.")

        self.context_secret = self.data.export_dict()
        return self.context_secret

    def __exit__(self, *args, **kwargs):
        if hasattr(self, "exported_keys"):
            logger.debug(f"Clearing credetials from environment")
            for key in self.exported_keys:
                logger.debug(f"Removed {key}")
                os.environ.pop(key, None)
        logger.debug(f"Clearing credetials from dictionary")
        for key in self.data.export_options():
            setattr(self, key, None)
            self.context_secret[key] = None

    def to_dict(self) -> dict:
        """Returns dict of credentials retrieved from the vault or set through constructor arguments.

        Returns
        -------
        dict
            dict of credentials retrieved from the vault or set through constructor.
        """
        return {key: getattr(self.data, key) for key in self.data.export_options()}

    def export_vault_details(
        self, filepath: str, format: str = "json", storage_options: dict = None
    ):  # Consider other name.
        """Save secret_id in a json file

        Parameters
        ----------
        filepath : str
            Filepath to save the file.
        format : str
            Default is `json`. Valid values:

            - `yaml` or `yml` - to store vault details in a yaml file
            - `json` - to store vault details in a json file

        storage_options: dict, optional.
            storage_options dict as required by `fsspec` library

        Returns
        -------
        None
            Returns None
        """
        import fsspec

        with fsspec.open(
            filepath, "w", storage_options=storage_options
        ) as cred_info_file:
            if format.lower() == "json":
                json.dump(
                    {
                        "vault_id": self.id,
                        "key_id": self.key_id,
                        "secret_id": self.secret_id,
                    },
                    cred_info_file,
                )
            elif format.lower() in ["yaml", "yml"]:
                yaml.dump(
                    {
                        "vault_id": self.id,
                        "key_id": self.key_id,
                        "secret_id": self.secret_id,
                    },
                    cred_info_file,
                )
            else:
                logger.error(
                    f"Unrecognized format: {format}. Value values are - json, yaml, yml"
                )
                raise ValueError(
                    f"Unrecognized format: {format}. Value values are - json, yaml, yml"
                )
