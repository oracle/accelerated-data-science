#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ads
from ads.common.utils import MYSQL_DEFAULT_PORT
from ads.secrets import SecretKeeper, Secret
import json


logger = ads.getLogger("ads.secrets")

from dataclasses import dataclass, field


@dataclass
class MySQLDBSecret(Secret):
    """
    Dataclass representing the attributes managed and serialized by MySQLDBSecretKeeper
    """

    user_name: str
    password: str
    host: str
    port: str
    database: str = field(default=None)


class MySQLDBSecretKeeper(SecretKeeper):
    """
    `MySQLDBSecretKeeper` provides an interface to save MySQL database credentials.
    If you use Wallet file for connnecting to the database, please use ``ADBSecretKeeper``.


    Examples
    --------
    >>> from ads.secrets.mysqldb import MySQLDBSecretKeeper
    >>> vault_id = "ocid1.vault.oc1..<unique_ID>"
    >>> key_id = "ocid1.key..<unique_ID>"

    >>> import ads
    >>> ads.set_auth("resource_principal") # If using resource principal for authentication
    >>> connection_parameters={
    ...     "user_name":"<your user name>",
    ...     "password":"<your password>",
    ...     "host":"<db host>",
    ...     "port":"<db port>",
    ...     "database":"<database>",
    ... }
    >>> mysqldb_keeper = MySQLDBSecretKeeper(vault_id=vault_id, key_id=key_id, **connection_parameters)
    >>> mysqldb_keeper.save("mysqldb_employee", "My DB credentials", freeform_tags={"schema":"emp"})
    >>> print(mysqldb_keeper.secret_id) # Prints the secret_id of the stored credentials
    >>> mysqldb_keeper.export_vault_details("mysqldb_employee_att.json") # Save the secret id and vault info to a json file

    >>> # Loading credentails
    >>> import ads
    >>> ads.set_auth("resource_principal") # If using resource principal for authentication
    >>> from ads.secrets.mysqldb import MySQLDBSecretKeeper
    >>> secret_id = "ocid1.vaultsecret.oc1..<unique_ID>"
    >>> with MySQLDBSecretKeeper.load_secret(source=secret_id) as mysqldb_creds:
    ...     import pandas as pd
    ...     df = pd.DataFrame.ads.read_sql("select * from EMPLOYEE", connection_parameters=mysqldb_creds, engine="mysql")


    >>> mymysqldb_creds = MySQLDBSecretKeeper.load_secret(source='mysqldb_employee_att.json', format="json")
    >>> pd.DataFrame.ads.read_sql("select * from ATTRITION_DATA", connection_parameters=mymysqldb_creds.to_dict(), engine="mysql").head(2)

    """

    def __init__(
        self,
        user_name: str = None,
        password: str = None,
        host: str = None,
        port: str = MYSQL_DEFAULT_PORT,
        database: str = None,
        repository_path: str = None,
        repository_key: str = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        user_name: (str, optional). Default None
            user_name of the database
        password: (str, optional). Default None
            password  for connecting to the database
        host: (str, optional). Default None
            Database host name
        port: (str, optional). Default 1521
            Port number
        database: (str, optional). Default None
            database name
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
            if database is None:
                database = config_from_repo.get("database")
            if host is None:
                host = config_from_repo.get("host")
            if port is None:
                port = config_from_repo.get("port")

        self.data = MySQLDBSecret(
            user_name=user_name,
            password=password,
            host=host,
            port=port,
            database=database,
        )

        super().__init__(**kwargs)

    def decode(self) -> "ads.secrets.mysqldb.MySQLDBSecretKeeper":
        """
        Converts the content in `self.encoded` to `MySQLDBSecret` and stores in `self.data`

        Returns
        -------
        MySQLDBSecretKeeper:
            Returns self object
        """
        content = json.loads(self._decode())
        logger.debug(f"Decoded secret contains following keys - {list(content.keys())}")
        self.data = MySQLDBSecret(
            **content,
        )
        return self
