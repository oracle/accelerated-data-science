#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict

from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.serializer import DataClassSerializable
from ads.config import CONDA_BUCKET_NAME, CONDA_BUCKET_NS
from ads.model.runtime.utils import (
    INFERENCE_ENV_SCHEMA_PATH,
    TRAINING_ENV_SCHEMA_PATH,
    SchemaValidator,
    get_service_packs,
)


DEFAULT_CONDA_BUCKET_NAME = "service-conda-packs"


class PACK_TYPE(Enum):
    """Conda Pack Type"""

    SERVICE_PACK = "data_science"
    USER_CUSTOM_PACK = "published"


class EnvInfo(ABC):
    """Env Info Base class."""

    @classmethod
    @abstractmethod
    def _populate_env_info(
        cls, env_slug: str, env_type: str, env_path: str, python_version: str
    ) -> "EnvInfo":
        """Populate the EnvInfo instance.

        Parameters
        ----------
        env_slug: (str)
            conda pack slug.
        env_type: (str)
            conda pack type: data_science or published.
        env_path: (str)
            conda pack object storage path.
        python_version: (str)
            python version of the conda pack.

        Returns
        -------
        EnvInfo
            An EnvInfo instance.
        """
        pass

    @classmethod
    def from_slug(
        cls,
        env_slug: str,
        namespace: str = CONDA_BUCKET_NS,
        bucketname: str = CONDA_BUCKET_NAME,
        auth: dict = None,
    ) -> "EnvInfo":
        """Initiate an EnvInfo object from a slug. Only service pack is allowed to use this method.

        Parameters
        ----------
        env_slug: str
            service pack slug.
        namespace: (str, optional)
            namespace of region.
        bucketname: (str, optional)
            bucketname of service pack.
        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        EnvInfo
            An EnvInfo instance.
        """
        if "oci://" not in env_slug:
            warnings.warn("slug will be deprecated. Provide conda pack path instead.")

        if not bucketname:
            warnings.warn(
                f"`bucketname` is not provided, defaults to `{DEFAULT_CONDA_BUCKET_NAME}`."
            )
            bucketname = DEFAULT_CONDA_BUCKET_NAME
        _, service_pack_slug_mapping = get_service_packs(
            namespace, bucketname, auth=auth
        )
        env_type, env_path, python_version = None, None, None
        if service_pack_slug_mapping:
            if env_slug in service_pack_slug_mapping:
                env_type = PACK_TYPE.SERVICE_PACK.value
                env_path, python_version = service_pack_slug_mapping[env_slug]
            else:
                warnings.warn(
                    f"The {env_slug} is not a service pack. Use `from_path` method by passing in the object storage path."
                )

        return cls._populate_env_info(
            env_slug=env_slug,
            env_type=env_type,
            env_path=env_path,
            python_version=python_version,
        )

    @classmethod
    def from_path(cls, env_path: str, auth: dict = None) -> "EnvInfo":
        """Initiate an object from a conda pack path.

        Parameters
        ----------
        env_path: str
            conda pack path.
        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        EnvInfo
            An EnvInfo instance.
        """
        bucketname, namespace, _ = ObjectStorageDetails.from_path(env_path).to_tuple()
        env_type = ""
        python_version = ""
        env_slug = ""
        service_pack_path_mapping = {}
        service_pack_path_mapping, _ = get_service_packs(
            namespace, bucketname, auth=auth
        )
        if env_path.startswith("oci://") and service_pack_path_mapping:
            if env_path in service_pack_path_mapping:
                env_type = PACK_TYPE.SERVICE_PACK.value
                (
                    env_slug,
                    python_version,
                ) = service_pack_path_mapping[env_path]
            else:
                env_type = PACK_TYPE.USER_CUSTOM_PACK.value
                try:
                    metadata_json = ObjectStorageDetails.from_path(
                        env_path
                    ).fetch_metadata_of_object()
                    python_version = metadata_json.get("python", None)
                    env_slug = metadata_json.get("slug", None)
                    if not python_version:
                        raise ValueError(
                            f"The manifest metadata of {env_path} doesn't contains inforamtion for python version."
                        )
                except Exception as e:
                    logging.debug(e)
                    logging.debug(
                        "python version and slug are not found from the manifest metadata."
                    )

        return cls._populate_env_info(
            env_slug=env_slug,
            env_type=env_type,
            env_path=env_path,
            python_version=python_version,
        )

    @staticmethod
    def _validate(obj_dict: Dict, schema_file_path: str) -> bool:
        """Validate the content in the ditionary format from the yaml file.

        Parameters
        ----------
        obj_dict: (Dict)
            yaml file content to validate.

        Returns
        -------
        bool
            Validation result.
        """
        validator = SchemaValidator(schema_file_path=schema_file_path)
        return validator.validate(document=obj_dict)


@dataclass(repr=False)
class TrainingEnvInfo(EnvInfo, DataClassSerializable):
    """Training conda environment info."""

    training_env_slug: str = ""
    training_env_type: str = ""
    training_env_path: str = ""
    training_python_version: str = ""

    @classmethod
    def _populate_env_info(
        cls, env_slug: str, env_type: str, env_path: str, python_version: str
    ) -> "TrainingEnvInfo":
        """Populate the TrainingEnvInfo instance.

        Parameters
        ----------
        env_slug: (str)
            conda pack slug.
        env_type: (str)
            conda pack type: data_science or published.
        env_path: (str)
            conda pack object storage path.
        python_version: (str)
            python version of the conda pack.

        Returns
        -------
        TrainingEnvInfo
            An TrainingEnvInfo instance.
        """
        return cls(
            training_env_slug=env_slug,
            training_env_type=env_type,
            training_env_path=env_path,
            training_python_version=python_version,
        )

    @classmethod
    def _validate_dict(cls, obj_dict: Dict) -> bool:
        """Validate the content in the dictionary format from the yaml file.

        Parameters
        ----------
        obj_dict: (Dict)
            yaml file content to validate.

        Returns
        -------
        bool
            Validation result.
        """
        return EnvInfo._validate(
            obj_dict=obj_dict, schema_file_path=TRAINING_ENV_SCHEMA_PATH
        )


@dataclass(repr=False)
class InferenceEnvInfo(EnvInfo, DataClassSerializable):
    """Inference conda environment info."""

    inference_env_slug: str = ""
    inference_env_type: str = ""
    inference_env_path: str = ""
    inference_python_version: str = ""

    @classmethod
    def _populate_env_info(
        cls, env_slug: str, env_type: str, env_path: str, python_version: str
    ) -> "InferenceEnvInfo":
        """Populate the InferenceEnvInfo instance.

        Parameters
        ----------
        env_slug: (str)
            conda pack slug.
        env_type: (str)
            conda pack type: data_science or published.
        env_path: (str)
            conda pack object storage path.
        python_version: (str)
            python version of the conda pack.

        Returns
        -------
        InferenceEnvInfo
            An InferenceEnvInfo instance.
        """
        return cls(
            inference_env_slug=env_slug,
            inference_env_type=env_type,
            inference_env_path=env_path,
            inference_python_version=python_version,
        )

    @classmethod
    def _validate_dict(cls, obj_dict: Dict) -> bool:
        """Validate the content in the dictionary format from the yaml file.

        Parameters
        ----------
        obj_dict: (Dict)
            yaml file content to validate.

        Returns
        -------
        bool
            Validation result.
        """
        return EnvInfo._validate(
            obj_dict=obj_dict, schema_file_path=INFERENCE_ENV_SCHEMA_PATH
        )
