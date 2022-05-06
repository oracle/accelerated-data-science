#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ads
from ads.secrets import SecretKeeper, Secret
from ads.bds.auth import DEFAULT_KRB5_CONFIG_PATH
import json
import os
import base64
from tqdm.auto import tqdm
from dataclasses import dataclass, field

logger = ads.getLogger("ads.secrets")


@dataclass
class BDSSecret(Secret):
    """
    Dataclass representing the attributes managed and serialized by BDSSecretKeeper.

    Attributes
    ----------
    principal: str
        The unique identity to which Kerberos can assign tickets.
    hdfs_host: str
        hdfs host name from the bds cluster.
    hive_host: str
        hive host name from the bds cluster.
    hdfs_port: str
        hdfs port from the bds cluster.
    hive_port: str
        hive port from the bds cluster.
    kerb5_path: str
        krb5.conf file path.
    kerb5_content: dict
        Content of the krb5.conf.
    keytab_path: str
        Path to the keytab file.
    keytab_content: dict
        Content of the keytab file.
    secret_id: str
        secret id where the BDSSecret is stored.
    """

    principal: str
    hdfs_host: str
    hive_host: str
    hdfs_port: str
    hive_port: str

    kerb5_path: str = field(
        default=None,
    )
    kerb5_content: dict = field(
        default=None,
        repr=False,
        metadata={"serializable": False},
    )
    keytab_path: str = field(
        default=None,
    )
    keytab_content: dict = field(
        default=None,
        repr=False,
        metadata={"serializable": False},
    )
    secret_id: str = field(repr=False, default_factory=str)


class BDSSecretKeeper(SecretKeeper):
    """
    `BDSSecretKeeper` provides an interface to save BDS hdfs and hive credentials.
    This interface does not store the wallet file by default.
    For saving keytab and krb5.cofig file, set `save_files=True` while calling `BDSSecretKeeper.save` method.

    Attributes
    ----------
    principal: str
        The unique identity to which Kerberos can assign tickets.
    hdfs_host: str
        hdfs host name from the bds cluster.
    hive_host: str
        hive host name from the bds cluster.
    hdfs_port: str
        hdfs port from the bds cluster.
    hive_port: str
        hive port from the bds cluster.
    kerb5_path: str
        krb5.conf file path.
    kerb5_content: dict
        Content of the krb5.conf.
    keytab_path: str
        Path to the keytab file.
    keytab_content: dict
        Content of the keytab file.
    secret_id: str
        secret id where the BDSSecret is stored.

    kwargs
    ------
    vault_id: str. OCID of the vault where the secret is stored. Required for saving secret.
    key_id: str. OCID of the key used for encrypting the secret. Required for saving secret.
    compartment_id: str. OCID of the compartment where the vault is located. Required for saving secret.
    auth: dict. Dictionay returned from ads.common.auth.api_keys() or ads.common.auth.resource_principal(). By default, will follow what is set in `ads.set_auth`.  Use this attribute to override the default.
    """

    def __init__(
        self,
        principal: str = None,
        hdfs_host: str = None,
        hive_host: str = None,
        hdfs_port: str = None,
        hive_port: str = None,
        kerb5_path: str = None,
        kerb5_content: str = None,
        keytab_path: str = None,
        keytab_content: str = None,
        keytab_dir: str = None,
        secret_id: str = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        principal: str
            The unique identity to which Kerberos can assign tickets.
        hdfs_host: str
            hdfs host name from the bds cluster.
        hive_host: str
            hive host name from the bds cluster.
        hdfs_port: str
            hdfs port from the bds cluster.
        hive_port: str
            hive port from the bds cluster.
        kerb5_path: str
            krb5.conf file path.
        kerb5_content: dict
            Content of the krb5.conf.
        keytab_path: str
            Path to the keytab file.
        keytab_content: dict
            Content of the keytab file.
        keytab_dir: (str, optional).
            Default None. Local directory where the extracted keytab content is saved.
        secret_id: str
            secret id where the BDSSecret is stored.


        kwargs
        ------
        vault_id: str. OCID of the vault where the secret is stored. Required for saving secret.
        key_id: str. OCID of the key used for encrypting the secret. Required for saving secret.
        compartment_id: str. OCID of the compartment where the vault is located. Required for saving secret.
        auth: dict. Dictionay returned from ads.common.auth.api_keys() or ads.common.auth.resource_principal(). By default, will follow what is set in `ads.set_auth`.  Use this attribute to override the default.

        """
        self.data = BDSSecret(
            principal=principal,
            hdfs_host=hdfs_host,
            hive_host=hive_host,
            hdfs_port=hdfs_port,
            hive_port=hive_port,
            kerb5_path=kerb5_path,
            kerb5_content=kerb5_content,
            keytab_path=keytab_path,
            keytab_content=keytab_content,
            secret_id=secret_id,
        )
        self.keytab_dir = keytab_dir
        super().__init__(**kwargs)

    def encode(self, serialize: bool = True) -> "ads.secrets.bds.BDSSecretKeeper":
        """
        Prepares content to save in vault. The port, host name and the keytab and krb5.config files are base64 encoded and stored in `self.secret`

        Parameters
        ----------
        serialize: bool, optional
            When set to True, loads the keytab and krb5.config file and encodes the content of both files.

        Returns
        -------
        BDSSecretKeeper:
            Returns self object
        """
        if serialize:
            if not self.data.keytab_path and not self.data.kerb5_path:
                raise ValueError(
                    "Missing path to keytab or krb5.config file. Required keytab file path to be set in `keytab_path` and krb5.config file path to be set in `kerb5_path`."
                )
            if self.data.keytab_path:
                with open(os.path.expanduser(self.data.keytab_path), "rb") as f:
                    encoded = self._encode(f.read())
                    self.keytab_content = encoded

            if self.data.kerb5_path:
                with open(os.path.expanduser(self.data.kerb5_path), "rb") as f:
                    encoded = self._encode(f.read())
                    self.kerb5_content = encoded

            self.secret = {
                **self.data.serialize(),
                "keytab_content": self.keytab_content,
                "kerb5_content": self.kerb5_content,
            }
            self.encoded = self.secret
        else:
            super().encode()

        return self

    def decode(self, save_files: bool = True) -> "ads.secrets.bds.BDSSecretKeeper":
        """
        Converts the content in `self.secret` to `BDSSecret` and stores in `self.data`

        If the `keytab_path` and `kerb5_path` are passed through the constructor, then retain it. We do not want to override what user has passed in
        If the `keytab_path` and `kerb5_path` are not passed, but the sercret has `secret_id`, then we generate the keytab file in the location specified by `keytab_path` in the constructor.

        Returns
        -------
        BDSSecretKeeper:
            Returns self object
        """
        content = json.loads(self._decode())
        data = BDSSecret(
            **content,
        )
        krb5_file_path = os.path.expanduser(DEFAULT_KRB5_CONFIG_PATH)
        if hasattr(self, "data") and self.data.keytab_path and self.data.kerb5_path:
            # If the keytab and kerb5 locations are passed in as a keyword argument inside
            # `load_secret` method, then we need to retain that information after
            # the secret was deserialized.
            logger.debug(f"Setting keytab file to {data.keytab_path}")
            data.keytab_path = self.data.keytab_path
            data.kerb5_path = self.data.kerb5_path
            if save_files:
                os.makedirs(os.path.dirname(krb5_file_path), exist_ok=True)
                if not os.path.exists(self.data.kerb5_path):
                    raise FileNotFoundError(
                        f"`kerb5_path` does not exists. Cannot load the kerb5 config from this location {self.data.kerb5_path}."
                    )
                with open(self.data.kerb5_path) as rf:
                    with open(krb5_file_path, "wb") as wf:
                        wf.write(rf.read())
        elif data.secret_id:
            logger.debug(f"Secret ids corresponding to the keytab file found.")

            if data.secret_id:
                content = json.loads(
                    self._decode(self.get_secret(data.secret_id, decoded=False))
                )

                if save_files:
                    if content["keytab_content"]:
                        if data.keytab_path:
                            if self.keytab_dir:
                                data.keytab_path = os.path.abspath(
                                    os.path.expanduser(
                                        os.path.join(
                                            self.keytab_dir,
                                            os.path.basename(data.keytab_path),
                                        )
                                    )
                                )
                            os.makedirs(
                                os.path.dirname(data.keytab_path), exist_ok=True
                            )
                        else:
                            raise ValueError(
                                "`keytab_path` field not found. `keytab_path` cannot be None when `save_files` = True."
                            )
                        with open(
                            os.path.abspath(os.path.expanduser(data.keytab_path)), "wb"
                        ) as wf:
                            wf.write(base64.b64decode(content["keytab_content"]))

                    if content["kerb5_content"]:
                        os.makedirs(os.path.dirname(krb5_file_path), exist_ok=True)
                        with open(
                            os.path.abspath(os.path.expanduser(krb5_file_path)), "wb"
                        ) as wf:
                            wf.write(base64.b64decode(content["kerb5_content"]))
        else:
            logger.warning(
                "Keytab and krb5 config information not found. If you have local keytab and krb5.config files, you can pass that as a keyword argument `keytab_path=<path to the keytab file>` and `kerb5_path=<path to the kerb5 file>` inside the `BDSSecretKeeper.load_secret` method"
            )
        self.data = data
        return self

    def save(
        self,
        name: str,
        description: str,
        freeform_tags: dict = None,
        defined_tags: dict = None,
        save_files: bool = True,
    ) -> "ads.secrets.bds.BDSSecretKeeper":
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
        save_files: (bool, optional). Default is False
            If set to True, saves the contents of the keytab and krb5 file as separate secret.

        Returns
        -------
        BDSSecretKeeper:
            Returns self object
        """
        if not save_files:
            logger.info(
                "`save_files` set to False. Not saving keytab or krb5.config file content to vault."
            )
        logger.debug(f"Encoding secrets. Save files is set to {save_files}")
        self.encode(serialize=save_files)

        secret_id = None
        n = 2 if save_files else 1
        with tqdm(total=n, leave=False) as pbar:
            if save_files:
                pbar.set_description(
                    f"Saving the keytab file and the krb5.config file."
                )
                secret_id = self.create_secret(
                    self._encode(
                        json.dumps(
                            {
                                "keytab_content": self.encoded["keytab_content"],
                                "kerb5_content": self.encoded["kerb5_content"],
                            }
                        ).encode("utf-8")
                    ),
                    encode=False,
                    secret_name=f"{name}_keytab_krb5",
                    description=f"Keytab file and krb5 config file.",
                    freeform_tags=freeform_tags,
                    defined_tags=defined_tags,
                )
                pbar.update()

            self.data.secret_id = secret_id

            pbar.set_description(f"Saving the credentials.")
            self.secret_id = self.create_secret(
                self._encode(json.dumps(self.data.serialize()).encode("utf-8")),
                encode=False,
                secret_name=name,
                description=description,
                freeform_tags=freeform_tags,
                defined_tags=defined_tags,
            )
            pbar.update()

        return self
