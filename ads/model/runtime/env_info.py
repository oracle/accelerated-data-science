#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict

from ads.common import utils
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
        if not bucketname:
            warnings.warn(
                f"`bucketname` is not provided, defaults to `{DEFAULT_CONDA_BUCKET_NAME}`."
            )
            bucketname = DEFAULT_CONDA_BUCKET_NAME
        _, service_pack_slug_mapping = get_service_packs(
            namespace, bucketname, auth=auth
        )
        if not service_pack_slug_mapping:
            raise ValueError(
                "The service conda environment list could not be extracted, so "
                f"the conda environment slug `{env_slug}` could not be resolved. "
                "Provide the full conda environment path from Environment "
                "Explorer, for example "
                "`oci://<bucket>@<namespace>/conda_environments/cpu/<env-name>/<version>/<slug>`."
            )

        if env_slug not in service_pack_slug_mapping:
            raise ValueError(
                f"The conda environment slug `{env_slug}` could not be resolved. "
                "ADS supports short slug names only for service conda "
                "environments. For custom or published conda environments, "
                "provide the full OCI path from Environment Explorer, for "
                "example "
                "`oci://<bucket>@<namespace>/conda_environments/cpu/<env-name>/<version>/<slug>`."
            )

        env_type = PACK_TYPE.SERVICE_PACK.value
        env_path, python_version = service_pack_slug_mapping[env_slug]

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
        object_storage_details = ObjectStorageDetails.from_path(
            env_path, auth=auth
        )
        cls._validate_conda_env_path(env_path, auth=auth)
        env_type = (
            PACK_TYPE.SERVICE_PACK.value
            if cls._is_service_conda_path(object_storage_details)
            else PACK_TYPE.USER_CUSTOM_PACK.value
        )
        python_version = ""
        env_slug = (
            os.path.basename(object_storage_details.filepath.rstrip("/"))
            if env_type == PACK_TYPE.SERVICE_PACK.value
            else ""
        )
        try:
            metadata_json = object_storage_details.fetch_metadata_of_object()
            python_version = metadata_json.get("python") or ""
            env_slug = metadata_json.get("slug") or env_slug
            if not python_version:
                logging.debug(
                    "The manifest metadata of %s does not contain python version.",
                    env_path,
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
    def _is_service_conda_path(object_storage_details: ObjectStorageDetails) -> bool:
        """Checks whether the full path points to the service conda bucket."""
        return (
            object_storage_details.bucket == DEFAULT_CONDA_BUCKET_NAME
            and object_storage_details.filepath.startswith("service_pack/")
        )

    @staticmethod
    def _validate_conda_env_path(env_path: str, auth: dict = None) -> None:
        """Validate that the full OCI conda path exists and is accessible."""
        try:
            if not utils.is_path_exists(env_path, auth=auth):
                raise ValueError(
                    f"The conda environment path `{env_path}` does not exist or "
                    "is not accessible. Provide a valid full conda environment "
                    "path from Environment Explorer."
                )
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"The conda environment path `{env_path}` could not be verified. "
                "Provide a valid full conda environment path from Environment "
                f"Explorer. Original error: {e}"
            ) from e

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
