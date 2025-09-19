#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
import os
from typing import Dict, Tuple

import fsspec
import yaml
from cerberus import DocumentError, Validator

from ads.common.object_storage_details import ObjectStorageDetails

MODEL_PROVENANCE_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "schemas",
    "model_provenance_schema.yaml",
)
INFERENCE_ENV_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "schemas",
    "inference_env_info_schema.yaml",
)
TRAINING_ENV_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "schemas",
    "training_env_info_schema.yaml",
)
SERVICE_PACKS = "service_packs"


class SchemaValidator:
    """
    Base Schema Validator which validate yaml file.
    """

    def __init__(self, schema_file_path: str) -> "SchemaValidator":
        """Initiate a SchemaValidator instance.

        Parameters
        ----------
        schema_file_path: (str)
            schema file path. The schema is used to validate the yaml file.

        Returns
        -------
        SchemaValidator
            A SchemaValidator instance.
        """
        self.schema_file_path = schema_file_path
        self._schema_validator = self._load_schema_validator()

    def validate(self, document: Dict) -> bool:
        """Validate the schema.

        Parameters
        ----------
        document: (Dict)
            yaml file content to validate.

        Raises
        ------
        DocumentError: Raised when the validation schema is missing, has the wrong format or contains errors.

        Returns
        -------
        bool
            validation result.
        """
        v = Validator()
        res = v.validate(document, self._schema_validator)
        if not res:
            raise DocumentError(v.errors)
        return res

    def _load_schema_validator(self):
        """load the schema validator to validate the schema.

        Returns
        -------
        dict
            Schema validator.
        """
        with open(self.schema_file_path, encoding="utf-8") as schema_file:
            ext = os.path.splitext(self.schema_file_path)[-1].lower()
            if ext in [".yaml", ".yml"]:
                schema_validator = yaml.load(schema_file, yaml.FullLoader)
            elif ext in [".json"]:
                schema_validator = json.load(schema_file)
            else:
                raise NotImplementedError(f"{ext} format schema is not supported.")
        return schema_validator


def _get_index_json_through_bucket(
    namespace: str, bucketname: str, auth: dict = None
) -> list:
    """get the index json from the object storage.

    Parameters
    ----------
    namespace: str
        The Object Storage namespace.
    bucketname: str
        The Object Storage bucketname.
    auth: (Dict, optional). Defaults to None.
        The default authentication is set using `ads.set_auth` API. If you need to override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
        authentication signer and kwargs required to instantiate IdentityClient object.

    Returns
    -------
    list: A list of dictionary which contains service packs information.
    """
    auth = auth or {}
    service_pack_list = []
    try:
        uri = f"oci://{bucketname}@{namespace}/service_pack/index.json"
        with fsspec.open(uri, "r", **auth) as f:
            service_packs = json.loads(f.read())
        service_pack_list = service_packs.get(SERVICE_PACKS)
    except Exception as e:
        logging.error(
            f"Error occurred in attempt to extract the list of the service conda environments "
            f"from the object storage for bucket '{bucketname}' and namespace '{namespace}'. "
            f"Please make sure that you've provided correct bucket and namespace."
        )
    return service_pack_list


def get_service_packs(
    namespace: str, bucketname: str, auth: dict = None
) -> Tuple[Dict, Dict]:
    """Get the service pack path mapping and service pack slug mapping.
    Note: deprecated packs are also included.

    Parameters
    ----------
    namespace: str
        namespace of the service pack.
    bucketname: str
        bucketname of the service pack.
    auth: (Dict, optional). Defaults to None.
        The default authentication is set using `ads.set_auth` API. If you need to override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
        authentication signer and kwargs required to instantiate IdentityClient object.

    Returns
    -------
    (Dict, Dict)
        Service pack path mapping(service pack path -> (slug, python version))
        and the service pack slug mapping(service pack slug -> (pack path, python version)).
    """
    service_pack_path_mapping = {}
    service_pack_slug_mapping = {}

    try:
        service_pack_list = _get_index_json_through_bucket(
            namespace=namespace,
            bucketname=bucketname,
            auth=auth
        )
    except Exception as e:
        logging.error(
            "Failed to fetch service packs index from namespace '%s' and bucket '%s': %s",
            namespace,
            bucketname,
            str(e),
        )
        return service_pack_path_mapping, service_pack_slug_mapping

    for service_pack in service_pack_list:
        # Here we need to replace the namespace and bucketname
        # with the bucket and namespace of the region that
        # user is in. The reason is that the mapping is generated
        # from the index.json file which has static namespace
        # and bucket of prod. however, namespace will change based
        # on the region. also, dev has different bucketname.
        pack_path = ObjectStorageDetails(
            bucket=bucketname,
            namespace=namespace,
            filepath=ObjectStorageDetails.from_path(
                service_pack.get("pack_path")
            ).filepath,
        ).path
        service_pack_path_mapping[pack_path] = (
            service_pack.get("slug"),
            service_pack.get("python"),
        )
        service_pack_slug_mapping[service_pack.get("slug")] = (
            pack_path,
            service_pack.get("python"),
        )
    return service_pack_path_mapping, service_pack_slug_mapping
