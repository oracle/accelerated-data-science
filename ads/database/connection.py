#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ast
import json
import oci
import os
import pathlib
import re
import shutil

from oci.exceptions import ServiceError
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

from oci.secrets import SecretsClient
from oci.config import from_file

from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common import utils
from ads.common import auth as authutil
from ads.common import oci_client as oc
from ads.vault.vault import Vault


class Connector:
    def __init__(
        self,
        secret_id: str = None,
        key: str = None,
        repository_path: str = None,
        **kwargs,
    ):

        """
        Validate that a connection could be made for the given set of connection parameters, and contruct a Connector object provided that the
        validation is successful.

        Parameters
        ----------
        secret_id: str, optional
            The ocid of the secret to retrieve from Oracle Cloud Infrastructure Vault.
        key: str, optional
            The key to find the database directory.
        repository_path: str, optional
            The local database information store, default to ~/.database unless specified otherwise.
        kwargs: dict, optional
            Name-value pairs that are to be added to the list of connection parameters.
            For example, database_name="mydb", database_type="oracle", username = "root", password = "example-password".

        Returns
        -------
        A Connector object.
        """

        prio_dict = {}

        if kwargs:

            command_creds = {}
            for input_key in kwargs.keys():
                command_creds[input_key] = kwargs.get(input_key)

            # declaring priority order
            prio_dict[1] = command_creds

        # get creds content from vault if secret id is provided
        if secret_id:
            if not bool(re.match("^ocid[0-9]?\.vaultsecret.*", secret_id)):
                raise ValueError(f"{secret_id} is not a valid secret id.")

            auth = authutil.default_signer()
            self.secret_client = oc.OCIClientFactory(**auth).secret

            secret_bundle = self.secret_client.get_secret_bundle(secret_id)
            secret_content = ast.literal_eval(
                Vault._secret_to_dict(secret_bundle.data.secret_bundle_content.content)
            )
            # declaring priority order
            prio_dict[2] = secret_content

        # get creds content from local if key is provided
        repository_path = _get_repository_path(repository_path=repository_path)
        if not os.path.exists(repository_path):
            raise ValueError(f"{repository_path} does not exist.")

        if key:
            if _not_valid_key(key=key):
                raise ValueError(f"{key} is not a valid directory name.")
            db_path = _get_db_path(repository_path=repository_path, key=key)
            if not os.path.exists(db_path):
                raise ValueError(f"{db_path} does not exist.")
            local_content = get_repository(key=key, repository_path=repository_path)
            # declaring priority order
            prio_dict[3] = local_content

        # Combine dictionary with priority using ** operator
        config = {}
        for k in reversed(sorted(prio_dict.keys())):
            config.update(**prio_dict[k])
        self.config = config

        # check database types
        valid_database_types = ["oracle"]

        if "database_type" not in self.config:
            raise ValueError(
                f"The database_type needs to be specified. "
                f"Valid database types are {valid_database_types}"
            )

        if self.config["database_type"] not in valid_database_types:
            raise ValueError(
                f"{self.config['database_type']} is not a valid database type. "
                f"Valid database types are {valid_database_types}"
            )

        if self.config["database_type"] == "oracle":
            connector = OracleConnector(self.config)
            self.uri = connector.uri
            self.engine = connector.engine

    def connect(self):
        return self.engine.connect()

    def __enter__(self):
        self.db_connection = self.engine.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self

    def __repr__(self):  # pragma: no cover
        return str(self.engine.url)


class OracleConnector:
    @runtime_dependency(module="sqlalchemy", install_from=OptionalDependency.DATA)
    def __init__(self, oracle_connection_config):
        self.config = oracle_connection_config

        # sanity check on valid keys before making a connection
        valid_keys = ["username", "password", "database_name"]
        for vk in valid_keys:
            if vk not in self.config.keys():
                raise ValueError(f"{vk} is a required parameter to connect.")

        self.uri = _create_connection_str(config=self.config)
        self.engine = sqlalchemy.create_engine(self.uri)


def update_repository(
    value: dict, key: str, replace: bool = True, repository_path: str = None
) -> dict:
    """
    Saves value into local database store.

    Parameters
    ----------
    value: dict
        The values to store locally.
    key: str
        The key to find the local database directory.
    replace: bool, default to True
        If set to false, updates the stored value.
    repository_path:str: str, optional
        The local database store, default to ~/.database unless specified otherwise.

    Returns
    -------
    A dictionary of all values in the repository for the given key.
    """
    if _not_valid_key(key=key):
        raise ValueError(f"{key} is not a valid directory name.")

    # make local database directory
    repository_path = _get_repository_path(repository_path=repository_path)
    pathlib.Path(repository_path).mkdir(parents=True, exist_ok=True)

    db_path = _get_db_path(repository_path=repository_path, key=key)
    pathlib.Path(db_path).mkdir(parents=True, exist_ok=True)

    db_config_path = os.path.join(db_path, "config.json")

    if not replace:
        value = _update(value, db_config_path)

    with open(db_config_path, "w") as fp:
        json.dump(value, fp)

    return value


def _update(new_value, db_config_path):
    # update existing key's values if found different and add new key-value pairs
    with open(db_config_path) as f:
        old_value = json.load(f)
    prio_dict = {1: old_value, 2: new_value}
    return {**prio_dict[2], **prio_dict[1]}


def get_repository(key: str, repository_path: str = None) -> dict:
    """
    Get all values from local database store.

    Parameters
    ----------
    key: str
        The key to find the database directory.
    repository_path: str, optional
        The path to local database store, default to ~/.database unless specified otherwise.

    Returns
    -------
    A dictionary of all values in the store.
    """
    if _not_valid_key(key=key):
        raise ValueError(f"{key} is not a valid directory name.")

    # check whether repository_path exists
    repository_path = _get_repository_path(repository_path=repository_path)
    if not os.path.exists(repository_path):
        raise ValueError(f"{repository_path} does not exist.")
    # check whether db_path exists
    db_path = _get_db_path(repository_path=repository_path, key=key)
    if not os.path.exists(db_path):
        raise ValueError(f"{db_path} does not exist.")

    db_config_path = os.path.join(db_path, "config.json")
    with open(db_config_path) as f:
        return json.load(f)


def import_wallet(wallet_path: str, key: str, repository_path: str = None) -> None:
    """
    Saves wallet to local database store.
    Unzip the wallet zip file, update sqlnet.ora and store wallet files.

    Parameters
    ----------
    wallet_path: str
        The local path to the downloaded wallet zip file.
    key: str
        The key to find the database directory.
    repository_path: str, optional
        The local database store, default to ~/.database unless specified otherwise.

    """
    if _not_valid_key(key=key):
        raise ValueError(f"{key} is not a valid directory name.")

    # checking paths are valid
    repository_path = _get_repository_path(repository_path=repository_path)
    db_path = _get_db_path(repository_path=repository_path, key=key)

    if not os.path.exists(db_path):
        raise ValueError(f"{db_path} does not exist.")
    if not os.path.exists(wallet_path):
        raise ValueError(f"{wallet_path} does not exist.")

    # Create a ZipFile Object and load wallet zip in it
    with ZipFile(wallet_path, "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(
            os.path.join(os.path.expanduser("~"), f"{repository_path}/{key}")
        )

    # Add TNS_ADMIN to the environment
    os.environ["TNS_ADMIN"] = db_path

    # Update the sqlnet.ora
    _update_sqlnet(db_path=db_path)

    # Update the config.json file so that the 'database_type' is set to oracle.
    db_config_path = os.path.join(db_path, "config.json")
    value = _update({"database_type": "oracle"}, db_config_path)
    with open(db_config_path, "w") as fp:
        json.dump(value, fp)


def _update_sqlnet(db_path):
    sqlnet_path = os.path.join(db_path, "sqlnet.ora")
    sqlnet_original_path = os.path.join(db_path, "sqlnet.ora.original")
    sqlnet_backup_path = os.path.join(db_path, "sqlnet.ora.backup")

    if not os.path.exists(sqlnet_original_path):
        shutil.copy(sqlnet_path, sqlnet_original_path)
    if os.path.exists(sqlnet_path):
        shutil.copy(sqlnet_path, sqlnet_backup_path)
    sqlnet_re = re.compile(
        '(WALLET_LOCATION\s*=.*METHOD_DATA\s*=.*DIRECTORY\s*=\s*")(.*)(".*)',
        re.IGNORECASE,
    )
    tmp = NamedTemporaryFile()

    with open(sqlnet_path, "rt") as sqlnet:
        for line in sqlnet:
            tmp.write(
                bytearray(
                    sqlnet_re.subn(r"\1{}\3".format(db_path), line)[0], encoding="utf-8"
                )
            )
    tmp.flush()
    shutil.copy(tmp.name, sqlnet_path)
    tmp.close()


def _get_repository_path(repository_path):
    return (
        os.path.join(os.path.expanduser("~"), ".database")
        if repository_path is None
        else repository_path
    )


def _get_db_path(repository_path, key):
    return os.path.join(repository_path, key)


def _not_valid_key(key):
    return bool(re.search("[../.]", key))


def _create_connection_str(config):
    return (
        "oracle+cx_oracle://"
        + config["username"]
        + ":"
        + config["password"]
        + "@"
        + config["database_name"]
    )
