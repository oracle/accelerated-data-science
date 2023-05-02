#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import warnings

warnings.warn(
    (
        "The `ads.common.model_artifact` is deprecated in `oracle-ads 2.6.9` and will be removed in `oracle-ads 3.0`."
        "Use framework specific Model utility class for saving and deploying model. "
        "Check https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/quick_start.html"
    ),
    DeprecationWarning,
    stacklevel=2,
)

import fnmatch
import importlib
import json
import os
import re
import git
import shutil
import subprocess
import sys
import textwrap
import uuid
import python_jsonschema_objects as pjs
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

import ads.dataset.factory as factory
import fsspec
import numpy as np
import oci.data_science
import oci.exceptions
import pandas as pd
import pkg_resources
import yaml

from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common import logger, utils
from ads.common import auth as authutil
from ads.common.data import ADSData
from ads.common.error import ChangesNotCommitted
from ads.model.model_introspect import (
    TEST_STATUS,
    Introspectable,
    IntrospectionNotPassed,
    ModelIntrospect,
)
from ads.model.model_metadata import (
    METADATA_SIZE_LIMIT,
    MetadataCustomCategory,
    MetadataCustomKeys,
    MetadataSizeTooLarge,
    MetadataTaxonomyKeys,
    ModelCustomMetadata,
    ModelCustomMetadataItem,
    ModelTaxonomyMetadata,
    UseCaseType,
)
from ads.common.object_storage_details import (
    InvalidObjectStoragePath,
    ObjectStorageDetails,
)
from ads.common.utils import DATA_SCHEMA_MAX_COL_NUM
from ads.config import (
    JOB_RUN_COMPARTMENT_OCID,
    JOB_RUN_OCID,
    NB_SESSION_COMPARTMENT_OCID,
    NB_SESSION_OCID,
    PROJECT_OCID,
)
from ads.common.decorator.deprecate import deprecated
from ads.feature_engineering.schema import DataSizeTooWide, Schema, SchemaSizeTooLarge
from ads.model.extractor.model_info_extractor_factory import ModelInfoExtractorFactory
from ads.model.model_version_set import ModelVersionSet
from ads.model.common.utils import fetch_manifest_from_conda_location
from git import InvalidGitRepositoryError, Repo

from oci.data_science.models import ModelProvenance

try:
    from yaml import CDumper as dumper
    from yaml import CLoader as loader
except:
    from yaml import Dumper as dumper
    from yaml import Loader as loader

MODEL_ARTIFACT_VERSION = "3.0"
INPUT_SCHEMA_FILE_NAME = "input_schema.json"
OUTPUT_SCHEMA_FILE_NAME = "output_schema.json"

_TRAINING_RESOURCE_OCID = JOB_RUN_OCID or NB_SESSION_OCID
_COMPARTMENT_OCID = NB_SESSION_COMPARTMENT_OCID or JOB_RUN_COMPARTMENT_OCID


class InvalidDataType(Exception):   # pragma: no cover
    """Invalid Data Type."""

    pass


SAMPLE_RUNTIME_YAML = f"""
MODEL_ARTIFACT_VERSION: '{MODEL_ARTIFACT_VERSION}'
MODEL_DEPLOYMENT:
    INFERENCE_CONDA_ENV:
        INFERENCE_ENV_SLUG: <slug of the conda environment>
        INFERENCE_ENV_TYPE: <data_science or published>
        INFERENCE_ENV_PATH: oci://<bucket-name>@<namespace>/<prefix>/<env>.tar.gz
        INFERENCE_PYTHON_VERSION: <python version>
"""


class ConflictStrategy(object):
    IGNORE = "IGNORE"
    UPDATE = "UPDATE"
    CREATE = "CREATE"


class PACK_TYPE(Enum):
    SERVICE_PACK = "data_science"
    USER_CUSTOM_PACK = "published"


class ModelArtifact(Introspectable):
    @deprecated(
        "2.6.6",
        details="Use framework specific Model utility class for saving and deploying model. Check https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/quick_start.html",
    )
    def __init__(
        self,
        artifact_dir,
        conflict_strategy=ConflictStrategy.IGNORE,
        install_libs=False,
        reload=True,
        create=False,
        progress=None,
        model_file_name="model.onnx",
        inference_conda_env=None,
        data_science_env=False,
        ignore_deployment_error=False,
        inference_python_version=None,
    ):
        """
        A class used to construct model artifacts.

        ...

        Attributes
        ----------
        artifact_dir: str
            Path to the model artifacts.
        conflict_strategy: ConflictStrategy, default: IGNORE
            How to handle version conflicts between the current environment and the requirements of
            model artifact.
        install_libs: bool
            Re-install the environment inwhich the model artifact were trained in.
        reload: bool
            Reload the model into the environment.
        create: bool
            Create the `runtime.yaml` file.
        progress:
            Show a progress bar.
        model_file_name: str
            Name of the model file.
        inference_conda_env: str
            The inference conda environment. If provided, the value will be set in the runtime.yaml.
            This is expected to be full oci URI format - `oci://{bucket}@{namespace}/path/to/condapack`.
        data_science_env: bool
            Is the inference conda environment managed by the Oracle Data Science service?
        ignore_deployment_error: bool
            Determine whether to turn off logging for deployment error.
            If set to True, the `.prepare()` method will ignore errors that impact model deployment.
        inference_python_version: str Optional, default None
            The version of Python to be used in inference. The value will be set in the `runtime.yaml` file


        Methods
        -------
        reload(self, model_file_name=None)
            Reload the files in the model artifact directory.
        verify(self, input_data)
            Verifies a model artifact directory.
        install_requirements(self, conflict_strategy=ConflictStrategy.IGNORE)
            Installs missing libraries listed in the model artifact.
        populate_metadata(self, model=None, use_case_type=None)
            Extracts and populate taxonomy metadata from the model.
        save(
            self,
            display_name: str = None,
            description: str = None,
            project_id: str = None,
            compartment_id: str = None,
            training_script_path: str = None,
            ignore_pending_changes: bool = False,
            auth: dict = None,
            training_id: str = None,
            timeout: int = None,
            ignore_introspection=False,
        )
            Saves this model artifact in model catalog.
        populate_schema(
            self,
            data_sample: ADSData = None,
            X_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
            y_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        )
            Populates input and output schema.
        introspect(self) -> pd.DataFrame
            Runs model introspection.

        """
        self.artifact_dir = (
            artifact_dir[:-1] if artifact_dir.endswith("/") else artifact_dir
        )
        self._introspect = ModelIntrospect(self)
        self.model = None
        self.score = None
        self.inference_conda_env = inference_conda_env
        self.data_science_env = data_science_env
        self.ignore_deployment_error = ignore_deployment_error
        self.metadata_taxonomy = ModelTaxonomyMetadata()
        self.metadata_custom = ModelCustomMetadata()
        self.schema_input = Schema()
        self.schema_output = Schema()
        self._serialization_format = None
        self.inference_python_version = inference_python_version

        if create:
            self.progress = progress
            if "CONDA_PREFIX" in os.environ and "NB_SESSION_OCID" in os.environ:
                self._generate_runtime_yaml(model_file_name=model_file_name)
            else:
                self._generate_empty_runtime_yaml(
                    model_file_name=model_file_name,
                    data_science_env=data_science_env,
                    inference_conda_env=inference_conda_env,
                    inference_python_version=inference_python_version,
                )
            self.version = MODEL_ARTIFACT_VERSION

        # This will re-install the environment inwhich the model artifact was trained in.
        if install_libs:
            self.install_requirements(conflict_strategy=conflict_strategy)
        # This will reload the model into the environment
        if reload:
            self.reload(model_file_name=model_file_name)

    def __repr__(self):
        res = "Artifact directory: %s\n" % self.artifact_dir
        res += "Contains: %s" % str(self._get_files())
        return res

    def __getattr__(self, item):
        return getattr(self.score, item)

    def __fetch_repo_details(self, training_code_info):
        repo = git.Repo(".", search_parent_directories=True)
        # get repository url
        if len(repo.remotes) > 0:
            repository_url = (
                repo.remotes.origin.url
                if repo.remotes.origin in repo.remotes
                else list(repo.remotes.values())[0].url
            )
        else:
            repository_url = "file://" + repo.working_dir  # no remote repo

        git_branch = None
        git_commit = None
        try:
            # get git branch
            git_branch = format(repo.active_branch)
            # get git commit
            git_commit = format(str(repo.head.commit.hexsha))
            training_code_info.GIT_COMMIT = git_commit
        except ValueError:
            # do not set commit if there isn't any
            pass

        training_code_info.GIT_REMOTE = repository_url
        training_code_info.GIT_BRANCH = git_branch
        training_code_info.ARTIFACT_DIRECTORY = self.artifact_dir
        return repo, training_code_info

    def __fetch_training_env_details(self, training_info):
        conda_prefix = os.environ.get("CONDA_PREFIX", None)
        pack_name = "NOT FOUND"
        try:
            manifest = fetch_manifest_from_conda_location(conda_prefix)
            manifest_type = manifest["type"]
            pack_name = manifest["pack_path"] if "pack_path" in manifest else None
            slug = manifest["slug"] if "slug" in manifest else ""

            if manifest_type == PACK_TYPE.USER_CUSTOM_PACK.value:
                if os.path.exists(
                    os.path.join(os.path.expanduser("~"), "conda", "config.yaml")
                ):
                    with open(
                        (os.path.join(os.path.expanduser("~"), "conda", "config.yaml"))
                    ) as conf:
                        user_config = yaml.load(conf, Loader=yaml.FullLoader)
                    pack_bucket = user_config["bucket_info"]["name"]
                    pack_namespace = user_config["bucket_info"]["namespace"]
                else:
                    logger.warning(
                        f"Cannot resolve the bucket name or namespace for the conda environment {conda_prefix}. "
                        f"You can set these values while saving the model or run `odsc init -b *bucket-name* -n *namespace*` and rerun the prepare step again."
                    )
            if not manifest_type or manifest_type.lower() not in [
                PACK_TYPE.USER_CUSTOM_PACK.value,
                PACK_TYPE.SERVICE_PACK.value,
            ]:
                if not self.ignore_deployment_error:
                    raise Exception(
                        f"Unknown manifest type. Manifest Type: {manifest_type or 'None'}"
                    )
            if not pack_name:
                if manifest_type == PACK_TYPE.USER_CUSTOM_PACK.value:
                    if self.data_science_env:
                        raise Exception(
                            f"For Published conda environments, assign the path of the environment in "
                            + "Object Storage to the `inference_conda_env` parameter and set the "
                            + "parameter `data_science_env` to `False`."
                        )
                    error_message = (
                        f"Pack destination is not known from the manifest file in {conda_prefix}. "
                        + "If it was cloned from another environment, consider publishing it before "
                        + "preparing the model artifact."
                    )
                    if self.ignore_deployment_error:
                        logger.warn(error_message)
                    else:
                        if not self.inference_conda_env:
                            logger.error(error_message)
                            logger.info(
                                "Provide a URI to the conda environment that you wish to use with the model "
                                "deployment service if you do not want to publish the current training environment."
                            )
                            raise Exception(
                                f"Could not resolve the path in the Object Storage for the conda environment: {conda_prefix}"
                            )
                else:
                    logger.warn(
                        f"Could not resolve the Object Storage destination of {conda_prefix}. Correct "
                        "the environment name and Object Storage details when saving."
                    )
        except Exception as e:
            raise e
        training_info.TRAINING_ENV_SLUG = slug
        if manifest_type.lower() in [
            PACK_TYPE.USER_CUSTOM_PACK.value,
            PACK_TYPE.SERVICE_PACK.value,
        ]:
            training_info.TRAINING_ENV_TYPE = manifest_type
        if pack_name:
            training_info.TRAINING_ENV_PATH = pack_name
        training_info.TRAINING_PYTHON_VERSION = sys.version.split("|")[0].strip()
        return training_info

    def __environment_details(self, model_provenance):
        model_provenance.TRAINING_REGION = os.environ.get("NB_REGION", "NOT_FOUND")
        model_provenance.TRAINING_COMPARTMENT_OCID = os.environ.get(
            "NB_SESSION_COMPARTMENT_OCID", "NOT_FOUND"
        )
        model_provenance.TRAINING_RESOURCE_OCID = os.environ.get(
            "NB_SESSION_OCID", "NOT_FOUND"
        )
        model_provenance.PROJECT_OCID = os.environ.get("PROJECT_OCID", "NOT_FOUND")
        model_provenance.TENANCY_OCID = os.environ.get("TENANCY_OCID", "NOT_FOUND")
        model_provenance.USER_OCID = os.environ.get("USER_OCID", "NOT_FOUND")
        model_provenance.VM_IMAGE_INTERNAL_ID = os.environ.get("VM_ID", "VMIDNOTSET")
        return model_provenance

    def __fetch_runtime_schema__(self):
        schema = None
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "model_artifact_schema.json"
            )
        ) as schema_file:
            schema = json.load(schema_file)

        if not schema:
            raise Exception(
                "Cannot load schema file to generate the runtime.yaml file."
            )

        builder = pjs.ObjectBuilder(schema)
        ns = builder.build_classes()
        return ns

    def _generate_empty_runtime_yaml(
        self,
        model_file_name="model.onnx",
        data_science_env=False,
        inference_conda_env=None,
        inference_python_version=None,
    ):
        if self.progress:
            self.progress.update("Creating runtime.yaml configuration.")
        logger.warning(
            "Generating runtime.yaml template. This file needs to be updated "
            "before saving it to the model catalog."
        )
        content = yaml.load(SAMPLE_RUNTIME_YAML, Loader=yaml.FullLoader)
        print(
            f"The inference conda environment is {inference_conda_env} and the Python version is {inference_python_version}."
        )
        if inference_conda_env:
            content["MODEL_DEPLOYMENT"]["INFERENCE_CONDA_ENV"][
                "INFERENCE_ENV_SLUG"
            ] = ""
            content["MODEL_DEPLOYMENT"]["INFERENCE_CONDA_ENV"][
                "INFERENCE_ENV_TYPE"
            ] = ""
            content["MODEL_DEPLOYMENT"]["INFERENCE_CONDA_ENV"][
                "INFERENCE_ENV_PATH"
            ] = inference_conda_env
        if inference_python_version:
            content["MODEL_DEPLOYMENT"]["INFERENCE_CONDA_ENV"][
                "INFERENCE_PYTHON_VERSION"
            ] = str(inference_python_version)

        content["MODEL_DEPLOYMENT"]["INFERENCE_CONDA_ENV"]["INFERENCE_ENV_TYPE"] = (
            PACK_TYPE.SERVICE_PACK.value
            if data_science_env
            else PACK_TYPE.USER_CUSTOM_PACK.value
        )

        with open(os.path.join(self.artifact_dir, "runtime.yaml"), "w") as outfile:
            yaml.dump(content, outfile)

    def _generate_runtime_yaml(self, model_file_name="model.onnx"):
        if self.progress:
            self.progress.update("Creating runtime.yaml configuration.")

        ns = self.__fetch_runtime_schema__()
        training_env_info = self.__fetch_training_env_details(ns.TrainingCondaEnv())
        model_provenance = self.__environment_details(ns.ModelProvenance())
        model_provenance.TRAINING_CONDA_ENV = training_env_info
        try:
            _, training_code_info = self.__fetch_repo_details(ns.TrainingCodeInfo())
            model_provenance.TRAINING_CODE = training_code_info
        except git.InvalidGitRepositoryError:
            pass

        if not training_env_info.TRAINING_ENV_PATH:
            logger.warning(
                "You did not publish the conda environment in which the madel was trained. Publishing the "
                "conda environment ensures that the exact training environment can be re-used later."
            )
        inference_info = ns.InferenceCondaEnv()
        if not self.inference_conda_env:
            message = "By default, the inference conda environment is the same as the training conda environment. Use the `inference_conda_env` parameter to override."
            if (
                training_env_info.TRAINING_ENV_TYPE
                and training_env_info.TRAINING_ENV_PATH
            ):
                logger.info(message)
                inference_info.INFERENCE_ENV_SLUG = training_env_info.TRAINING_ENV_SLUG
                inference_info.INFERENCE_ENV_TYPE = training_env_info.TRAINING_ENV_TYPE
                inference_info.INFERENCE_ENV_PATH = training_env_info.TRAINING_ENV_PATH
                inference_info.INFERENCE_PYTHON_VERSION = (
                    training_env_info.TRAINING_PYTHON_VERSION
                )
                self.conda_env = str(training_env_info.TRAINING_ENV_SLUG)
        else:
            self.conda_env = os.path.basename(str(self.inference_conda_env))
            if self.inference_conda_env.startswith("oci://"):
                inference_info.INFERENCE_ENV_PATH = self.inference_conda_env
                try:
                    metadata_json = ObjectStorageDetails.from_path(
                        env_path=self.inference_conda_env
                    ).fetch_metadata_of_object()
                    inference_info.INFERENCE_PYTHON_VERSION = metadata_json["python"]
                except:
                    if not self.inference_python_version:
                        if not training_env_info.TRAINING_PYTHON_VERSION:
                            raise Exception(
                                "The Python version was not specified."
                                "Pass in the Python version when preparing a model."
                            )
                        else:
                            logger.warning(
                                "The Python version could not be inferred from the conda environment. Defaulting to the Python "
                                "version that was used in training."
                            )
                            inference_info.INFERENCE_PYTHON_VERSION = (
                                training_env_info.TRAINING_PYTHON_VERSION
                            )
                    else:
                        inference_info.INFERENCE_PYTHON_VERSION = (
                            self.inference_python_version
                        )
            else:
                pass
        model_deployment_info = None
        if inference_info.INFERENCE_ENV_PATH:
            model_deployment_info = ns.ModelDeployment()
            model_deployment_info.INFERENCE_CONDA_ENV = inference_info

        if (
            not self.inference_conda_env
            and not self.data_science_env
            and inference_info.INFERENCE_ENV_TYPE == PACK_TYPE.SERVICE_PACK.value
            and training_env_info.TRAINING_ENV_PATH == inference_info.INFERENCE_ENV_PATH
        ):
            error_message = (
                f"The inference conda environment {training_env_info.TRAINING_ENV_SLUG} may have changed. "
                + "Publish the current conda environment or set the parameter `data_science_env` to `True` "
                + "in the `.prepare()` method."
            )
            if not self.ignore_deployment_error:
                raise Exception(error_message)
            else:
                logger.warning(error_message)

        if not inference_info.INFERENCE_ENV_PATH and not self.inference_conda_env:
            error_message = (
                f"The inference conda environment is missing. Set the `inference_conda_env` parameter "
                + "or publish the conda environment and run the `.prepare()` method."
            )
            if not self.ignore_deployment_error:
                raise Exception(error_message)
            else:
                logger.warn(error_message)

        if model_deployment_info:
            self._runtime_info = ns.ModelArtifactSchema(
                MODEL_ARTIFACT_VERSION=MODEL_ARTIFACT_VERSION,
                MODEL_PROVENANCE=model_provenance,
                MODEL_DEPLOYMENT=model_deployment_info,
            )
        else:
            self._runtime_info = ns.ModelArtifactSchema(
                MODEL_ARTIFACT_VERSION=MODEL_ARTIFACT_VERSION,
                MODEL_PROVENANCE=model_provenance,
            )

        with open(os.path.join(self.artifact_dir, "runtime.yaml"), "w") as outfile:
            outfile.write("# Model runtime environment\n")
            yaml.dump(self._runtime_info.as_dict(), outfile, default_flow_style=False)

    def reload(self, model_file_name: str = None):
        """
        Reloads files in model artifact directory.

        Parameters
        ----------
        model_file_name: str
            The model file name.
        """
        spec = importlib.util.spec_from_file_location(
            "score%s" % uuid.uuid4(), os.path.join(self.artifact_dir, "score.py")
        )
        score = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(score)
        self.score = score
        if os.path.exists(os.path.join(self.artifact_dir, "runtime.yaml")):
            if model_file_name:
                self.model = self.score.load_model(model_file_name)
            else:
                self.model = self.score.load_model()
            with open(os.path.join(self.artifact_dir, "runtime.yaml")) as runtime_file:
                runtime = yaml.load(runtime_file, Loader=yaml.FullLoader)
            self.version = runtime["MODEL_ARTIFACT_VERSION"]
            try:
                self.VM_ID = runtime["MODEL_PROVENANCE"]["VM_IMAGE_INTERNAL_ID"]
            except KeyError:
                self.VM_ID = None
            try:
                self.conda_env = runtime["MODEL_PROVENANCE"]["TRAINING_CONDA_ENV"][
                    "TRAINING_ENV_SLUG"
                ]
            except KeyError:
                self.conda_env = None
        elif os.path.exists(os.path.join(self.artifact_dir, "ds-runtime.yaml")):
            self.model = self.score.load_model()
            with open(
                os.path.join(self.artifact_dir, "ds-runtime.yaml")
            ) as runtime_file:
                runtime = yaml.load(runtime_file, Loader=yaml.FullLoader)
            self.version = "1.0"
            self.VM_ID = None  # get ads/mlx version?
            self.conda_env = runtime["conda-env"]
        else:
            self.model = self.score.load_model()
            self.version = "0.0"
            self.VM_ID = "UNKNOWN"
            self.conda_env = "base"
            # raise FileNotFoundError(os.path.join(self.artifact_dir, 'runtime.yaml'))
        # __pycache__ was created during model_artifact.reload() above
        if os.path.exists(os.path.join(self.artifact_dir, "__pycache__")):
            shutil.rmtree(
                os.path.join(self.artifact_dir, "__pycache__"), ignore_errors=True
            )
        # extract model serialization format as part of custom metadata
        if model_file_name:
            self._serialization_format = self._extract_model_serialization_format(
                model_file_name
            )
            if (
                MetadataCustomKeys.MODEL_SERIALIZATION_FORMAT
                in self.metadata_custom.keys
            ):
                self.metadata_custom[
                    MetadataCustomKeys.MODEL_SERIALIZATION_FORMAT
                ].value = self._serialization_format
            else:
                self.metadata_custom.add(
                    key=MetadataCustomKeys.MODEL_SERIALIZATION_FORMAT,
                    value=self._serialization_format,
                    description="The model serialization format",
                    category=MetadataCustomCategory.TRAINING_PROFILE,
                )

    @staticmethod
    def _extract_model_serialization_format(model_file_name):
        return os.path.splitext(model_file_name)[1][1:]

    def verify(self, input_data):
        """
        Verifies the contents of the  model artifact directory.

        Parameters
        ----------
        input_data : str, dict, BytesIO stream
            Data to be passed into the deployed model. It can be of type json (str), a dict object, or a BytesIO stream.
            All types get converted into a UTF-8 encoded BytesIO stream and is then sent to the handler.
            Any data handling past there is done in func.py. By default it looks for data
            under the keyword "input", and returns data under teh keyword "prediction".

        Returns
        -------
        output_data : the resulting prediction, formatted in the same way as input_data

        Example
         --------
         input_dict = {"input": train.X[:3].to_dict()}
         model_artifact.verify(input_dict)

         * returns {"prediction": [30/4, 24.8, 30.7]} *
        """

        # Fake Context obj created for Fn Handler
        class FakeCtx:
            def SetResponseHeaders(self, headers, status_code):
                return

        ctx = FakeCtx()
        from io import BytesIO

        if type(input_data) == str:
            data = BytesIO(input_data.encode("UTF-8"))
            data_type = "json"
        elif type(input_data) == dict:
            from json import dumps

            data = BytesIO(dumps(input_data).encode("UTF-8"))
            data_type = "dict"
        elif isinstance(type(input_data), type(BytesIO)):
            data = input_data
            data_type = "BytesIO"
        else:
            raise TypeError

        sys_path = sys.path.copy()
        try:
            if self.version.split(".")[0] not in ["0", "1"]:
                sys.path.insert(0, self.artifact_dir)
            else:
                sys.path.insert(0, os.path.join(self.artifact_dir, "fn-model"))
            import func

            resp = func.handler(ctx, data)
            output_json = resp.body()
        finally:
            # Reset in case func.py messes with it
            sys.path = sys_path

        if data_type == "json":
            return output_json
        output_bstream = BytesIO(resp.body().encode("UTF-8"))
        if data_type == "BytesIO":
            return output_bstream
        else:
            from json import load

            return load(output_bstream)

    @deprecated(
        "2.6.6",
        details="Use framework specific Model utility class for saving and deploying model. Check https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/quick_start.html",
    )
    def save(
        self,
        display_name: str = None,
        description: str = None,
        project_id: str = None,
        compartment_id: str = None,
        training_script_path: str = None,
        ignore_pending_changes: bool = False,
        auth: dict = None,
        training_id: str = None,
        timeout: int = None,
        ignore_introspection=True,
        freeform_tags=None,
        defined_tags=None,
        bucket_uri: Optional[str] = None,
        remove_existing_artifact: Optional[bool] = True,
        model_version_set: Optional[Union[str, ModelVersionSet]] = None,
        version_label: Optional[str] = None,
    ):
        """
        Saves the model artifact in the model catalog.

        Parameters
        ----------
        display_name : str, optional
            Model display name.
        description : str, optional
            Description for the model.
        project_id : str, optional
            Model's project OCID.
            If None, the default project OCID `config.PROJECT_OCID` would be used.
        compartment_id : str, optional
            Model's compartment OCID.
            If None, the default compartment OCID `config.NB_SESSION_COMPARTMENT_OCID` would be used.
        training_script_path : str, optional
            The training script path is either relative to the working directory,
            or an absolute path.
        ignore_pending_changes : bool, default: False
            If True, ignore uncommitted changes and use the current git HEAD commit for provenance metadata.
            This argument is used only when the function is called from a script in git managed directory.
        auth: dict
            Default is None. Default authetication is set using the `ads.set_auth()` method.
            Use the `ads.common.auth.api_keys()` or `ads.common.auth.resource_principal()` to create appropriate
            authentication signer and kwargs required to instantiate a DataScienceClient object.
        training_id: str, optional
            The training OCID for the model.
        timeout: int, default: 10
            The connection timeout in seconds.
        ignore_introspection: bool, optional
            Ignore the result of model introspection .
            If set to True, the `.save()` will ignore all model introspection errors.
        freeform_tags : dict(str, str), optional
            Freeform tags for the model.
        defined_tags : dict(str, dict(str, object)), optional
            Defined tags for the model.
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for uploading large artifacts which
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Whether artifacts uploaded to object storage bucket need to be removed or not.
        model_version_set: (Union[str, ModelVersionSet], optional). Defaults to None.
            The Model version set OCID, or name, or `ModelVersionSet` instance.
        version_label: (str, optional). Defaults to None.
            The model version label.

        Examples
        ________
        >>> from ads.common.model_artifact import ModelArtifact
        >>> from ads.config import NB_SESSION_OCID

        >>> # Getting auth details.
        >>> # If you are using API keys
        >>> auth=ads.common.auth.api_keys()

        >>> # If you are using resource principal
        >>> auth=ads.common.auth.resource_principal()

        >>> # If you have set the auth type using ads.set_auth()
        >>> auth=ads.common.auth.default_signer()

        >>> # Preparing model artifacts
        >>> model_artifact = prepare_generic_model(
        ...     "path_to_model_artifacts",
        ...     force_overwrite=True,
        ...     data_science_env=True,
        ...     model=gamma_reg_model,
        ... )

        >>> # Saving model to the model catalog
        >>> model_artifact.save(
        ...     project_id=PROJECT_ID,
        ...     compartment_id=COMPARTMENT,
        ...     display_name="RF Classifier 2",
        ...     description="A sample Random Forest classifier",
        ...     ignore_pending_changes=True,
        ...     auth=auth,
        ...     training_id=NB_SESSION_OCID,
        ...     timeout=6000,
        ...     ignore_introspection = True
        ... )
        """
        if timeout and not isinstance(timeout, int):
            raise TypeError("Timeout must be an integer.")

        runtime_yaml_file = os.path.join(self.artifact_dir, "runtime.yaml")
        if os.path.exists(runtime_yaml_file):
            with open(runtime_yaml_file, "r") as mfile:
                runtime_prep_info = yaml.load(mfile, Loader=yaml.FullLoader)
                # runtime_info['pack-info'] = deployment_pack_info
        else:
            runtime_prep_info = {}
        ns = self.__fetch_runtime_schema__()
        runtime_info = ns.ModelArtifactSchema().from_json(json.dumps(runtime_prep_info))

        training_code_info = self._training_code_info(
            ns, training_script_path, ignore_pending_changes
        )
        if not training_id:
            training_id = _TRAINING_RESOURCE_OCID
        model_provenance_metadata = ModelProvenance(
            repository_url=str(training_code_info.GIT_REMOTE),
            git_branch=str(training_code_info.GIT_BRANCH),
            git_commit=str(training_code_info.GIT_COMMIT),
            script_dir=str(training_code_info.ARTIFACT_DIRECTORY),
            training_script=str(training_code_info.TRAINING_SCRIPT),
            training_id=training_id,
        )
        if getattr(runtime_info, "MODEL_PROVENANCE", None):
            runtime_info.MODEL_PROVENANCE.TRAINING_CODE = training_code_info

        logger.info(model_provenance_metadata)

        # handle the case where project_id and/or compartment_id is not specified by the user
        if not project_id and not PROJECT_OCID:
            raise ValueError("The `project_id` must be provided.")

        if not compartment_id and not NB_SESSION_COMPARTMENT_OCID:
            raise ValueError("The `compartment_id` must be provided.")

        if os.path.exists(os.path.join(self.artifact_dir, "__pycache__")):
            shutil.rmtree(
                os.path.join(self.artifact_dir, "__pycache__"), ignore_errors=True
            )
        self.metadata_custom._add(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.MODEL_ARTIFACTS,
                value=textwrap.shorten(
                    ", ".join(self._get_files()), 255, placeholder="..."
                ),
                description="The list of files located in artifacts folder.",
                category=MetadataCustomCategory.TRAINING_ENV,
            ),
            replace=True,
        )

        client_auth = auth if auth else authutil.default_signer()

        if timeout:
            if not client_auth.get("client_kwargs"):
                client_auth["client_kwargs"] = {}
            client_auth["client_kwargs"]["timeout"] = timeout

        if freeform_tags and not isinstance(freeform_tags, dict):
            raise TypeError("Freeform tags must be a dictionary.")

        if defined_tags and not isinstance(defined_tags, dict):
            raise TypeError("Defined tags must be a dictionary.")

        self._validate_metadata()
        self._validate_schema()

        with open(runtime_yaml_file, "w") as mfile:
            yaml.dump(runtime_info.as_dict(), mfile, Dumper=dumper)

        if not ignore_introspection:
            self._introspect()
            if self._introspect.status == TEST_STATUS.NOT_PASSED:
                msg = (
                    "Model introspection not passed. "
                    "Use `.introspect()` method to get detailed information and follow the "
                    "messages to fix it. To save model artifacts ignoring introspection "
                    "use `.save(ignore_introspection=True...)`."
                )
                raise IntrospectionNotPassed(msg)
        try:
            from ads.catalog.model import ModelCatalog

            return ModelCatalog(
                compartment_id=compartment_id,
                ds_client_auth=client_auth,
                identity_client_auth=client_auth,
            ).upload_model(
                self,
                provenance_metadata=model_provenance_metadata,
                display_name=display_name,
                description=description,
                project_id=project_id,
                freeform_tags=freeform_tags,
                defined_tags=defined_tags,
                bucket_uri=bucket_uri,
                remove_existing_artifact=remove_existing_artifact,
                model_version_set=model_version_set,
                version_label=version_label,
            )
        except oci.exceptions.RequestException as e:
            if "The write operation timed out" in str(e):
                logger.error(
                    "The save operation timed out. Try to set a longer timeout e.g. save(timeout=600, ...)."
                )

    def _validate_schema(self):
        if not self._validate_schema_size(self.schema_input, INPUT_SCHEMA_FILE_NAME):
            self.schema_input.to_json_file(
                os.path.join(self.artifact_dir, INPUT_SCHEMA_FILE_NAME)
            )
        if not self._validate_schema_size(self.schema_output, OUTPUT_SCHEMA_FILE_NAME):
            self.schema_output.to_json_file(
                os.path.join(self.artifact_dir, OUTPUT_SCHEMA_FILE_NAME)
            )
        self.schema_input.validate_schema()
        self.schema_output.validate_schema()

    def _validate_metadata(self):
        self.metadata_custom.validate()
        self.metadata_taxonomy.validate()
        total_size = self.metadata_custom.size() + self.metadata_taxonomy.size()
        if total_size > METADATA_SIZE_LIMIT:
            raise MetadataSizeTooLarge(total_size)
        return True

    def _training_code_info(
        self, ns, training_script_path=None, ignore_pending_changes=False
    ):
        try:
            repo, training_code_info = self.__fetch_repo_details(ns.TrainingCodeInfo())
        except git.InvalidGitRepositoryError:
            repo = None
            training_code_info = ns.TrainingCodeInfo()
        if training_script_path is not None:
            if not os.path.exists(training_script_path):
                logger.warning(
                    f"Training script {os.path.abspath(training_script_path)} does not exists."
                )
            else:
                training_script = os.path.abspath(training_script_path)
                self._assert_path_not_dirty(
                    training_script_path, repo, ignore_pending_changes
                )
                training_code_info.TRAINING_SCRIPT = training_script

        self._assert_path_not_dirty(self.artifact_dir, repo, ignore_pending_changes)
        training_code_info.ARTIFACT_DIRECTORY = os.path.abspath(self.artifact_dir)

        return training_code_info

    def _assert_path_not_dirty(self, path, repo, ignore):
        if repo is not None and not ignore:
            path_abs = os.path.abspath(path)
            if os.path.commonpath([path_abs, repo.working_dir]) == repo.working_dir:
                path_relpath = os.path.relpath(path_abs, repo.working_dir)
                if repo.is_dirty(path=path_relpath) or any(
                    [
                        os.path.commonpath([path_relpath, untracked]) == path_relpath
                        for untracked in repo.untracked_files
                    ]
                ):
                    raise ChangesNotCommitted(path_abs)

    def install_requirements(self, conflict_strategy=ConflictStrategy.IGNORE):
        """
        Installs missing libraries listed in the model artifacts.

        Parameters
        ----------
        conflict_strategy : ConflictStrategy, default: IGNORE
            Update the conflicting dependency to the version required by the model artifact.
            Valid values: "IGNORE" or ConflictStrategy.IGNORE, "UPDATE" or ConflictStrategy.UPDATE.
            IGNORE: Use the installed version in  case of a conflict.
            UPDATE: Force update dependency to the version required by model artifact in case of conflict.
        """
        importlib.reload(pkg_resources)
        from pkg_resources import DistributionNotFound, VersionConflict

        if self.version.split(".")[0] not in ["0", "1"] and os.path.exists(
            Path(os.path.join(self.artifact_dir), "requirements.txt")
        ):
            requirements = (
                Path(os.path.join(self.artifact_dir), "requirements.txt")
                .read_text()
                .strip()
                .split("\n")
            )
        elif self.version.split(".")[0] in ["0", "1"] and Path(
            os.path.join(self.artifact_dir), "ds-requirements.txt"
        ):
            requirements = (
                Path(os.path.join(self.artifact_dir), "ds-requirements.txt")
                .read_text()
                .strip()
                .split("\n")
            )
        else:
            raise FileNotFoundError(
                "Could not find requirements.txt. Install the necessary libraries and "
                "re-construct the model artifact with install_libs=False."
            )

        version_conflicts = {}
        for requirement in requirements:
            try:
                pkg_resources.require(requirement)
            except VersionConflict as vc:
                if conflict_strategy == ConflictStrategy.UPDATE:
                    pip_install("%s%s" % (vc.req.name, vc.req.specifier), "-U")
                elif conflict_strategy == ConflictStrategy.IGNORE:
                    version_conflicts[
                        "%s==%s" % (vc.dist.key, vc.dist.parsed_version)
                    ] = "%s%s" % (vc.req.name, vc.req.specifier)
            except DistributionNotFound as dnf:
                pip_install(requirement)
                # distributions_not_found.add('%s%s' % (dnf.req.name, dnf.req.specifier))
        if len(version_conflicts) > 0:
            print(
                "\033[93m"
                + str(VersionConflictWarning(version_conflicts=version_conflicts))
                + "\033[0m"
            )

    def _get_files(self):
        if os.path.exists(os.path.join(self.artifact_dir, ".model-ignore")):
            ignore_patterns = (
                Path(os.path.join(self.artifact_dir), ".model-ignore")
                .read_text()
                .strip()
                .split("\n")
            )
        else:
            ignore_patterns = []
        file_names = []
        for root, dirs, files in os.walk(self.artifact_dir):
            for name in files:
                file_names.append(os.path.join(root, name))
            for name in dirs:
                file_names.append(os.path.join(root, name))

        for ignore in ignore_patterns:
            if not ignore.startswith("#") and ignore.strip() != "":
                matches = []
                for file_name in file_names:
                    if ignore.endswith("/"):
                        ignore = ignore[:-1] + "*"
                    if not re.search(
                        fnmatch.translate("/%s" % ignore.strip()), file_name
                    ):
                        matches.append(file_name)
                file_names = matches
        return [
            matched_file[len(self.artifact_dir) + 1 :] for matched_file in file_names
        ]

    def _save_data_from_memory(
        self,
        prefix: str,
        train_data: Union[pd.DataFrame, list, np.ndarray],
        validation_data: Union[pd.DataFrame, list, np.ndarray] = None,
        train_data_name: str = "train.csv",
        validation_data_name: str = "validation.csv",
        storage_options: dict = None,
        **kwargs,
    ):
        """
        Save data to Object Storage.
        return [
            matched_file[len(self.artifact_dir) + 1 :] for matched_file in file_names
        ]

        Parameters
        ----------
        prefix: str
            A prefix to append to the Object Storage key.
            e.g. oci://bucket_name@namespace/prefix
        train_data: Union[pd.DataFrame, list, np.ndarray].
            The training data to be stored.
        validation_data: Union[pd.DataFrame, list, np.ndarray]. Default None
            The validation data to be stored.
        train_data_name: str. Default 'train.csv'.
            Filename used to save the train data. The key is prefix/train_data_name.
        validation_data_name: str. Default 'train.csv'.
            Filename used to save the validation data. The key is prefix/validation_data_name.
        storage_options: dict. Default None
            Parameters passed on to the backend filesystem class.
            Defaults to `storage_options` set using `DatasetFactory.set_default_storage()`.

        Returns
        -------
        None
            Nothing.
        Examples
        ________
        >>> from ads.common.model_artifact import ModelArtifact
        >>> import ocifs
        >>> import oci
        >>> storage_options = {"config": oci.config.from_file(os.path.join("~/.oci", "config"))}
        >>> storage_options
        {'log_requests': False,
            'additional_user_agent': '',
            'pass_phrase': None,
            'user': 'ocid5.user.oc1..aaaaaaaab3geixlk***********************',
            'fingerprint': '05:15:2b:b1:46:8a:32:ec:e2:69:5b:32:01:**:**:**)',
            'tenancy': 'ocid5.tenancy.oc1..aaaaaaaag*************************',
            'region': 'us-ashburn-1',
            'key_file': '/home/datascience/.oci/oci_api_key.pem'}
        >>> path_to_generic_model_artifact = tempfile.mkdtemp()
        >>> df = pd.DataFrame([[1, 2], [2, 3], [3, 4], [4, 3]])
        >>> generic_model_artifact = prepare_generic_model(path_to_generic_model_artifact,
        ...                                   force_overwrite=True, data_science_env=True,
        ...                                   ignore_deployment_error=True)
        >>> generic_model_artifact._save_data_from_memory(prefix = 'oci://bucket_name@namespace/folder_name',
        ... train_data=df, storage_options=storage_options)
        """
        if not re.match(r"oci://*@*", prefix):
            raise InvalidObjectStoragePath(
                "`prefix` is not valid. It must have the pattern 'oci://bucket_name@namespace/key'."
            )
        if not storage_options:
            storage_options = factory.default_storage_options
        if not storage_options:
            storage_options = {"config": {}}

        self._save_from_memory(
            train_data, prefix, train_data_name, storage_options, "training", **kwargs
        )

        if validation_data is not None:
            self._save_from_memory(
                validation_data,
                prefix,
                validation_data_name,
                storage_options,
                "validation",
                **kwargs,
            )

    def _save_data_from_file(
        self,
        prefix: str,
        train_data_path: str = None,
        validation_data_path: str = None,
        storage_options: dict = None,
        **kwargs,
    ):
        """
        Save the data to Object Storage.

        Parameters
        ----------
        prefix: str
            The Object Storage prefix to store the data. When `train_data_path` or
            `validation_data_path` are provided, they are stored under this prefix
            with their original filenames. If the data are already stored on Object
            Storage, you can provide the path to the data. If no local data path is provided,
            no data is `prefix` is saved in the custom metadata.
        train_data_path: str. Default None.
            Local path for the training data.
        validation_data_path: str. Default None.
            Local path for the validation data.
        storage_options: dict. Default None
            Parameters passed on to the backend filesystem class.
            Defaults to `storage_options` set using `DatasetFactory.set_default_storage()`.

        Keyword Arguments
        _________________
        data_type:
            Either `training` or `validation`. Used when the data are
            already stored remotely and you want to record the path in
            `metadata_custom`. Pass the prefix of your data and `data_type`
            to indicate whether this data is of `training` or `validation` type.
            The `storage_options` is needed in this case.

        Returns
        -------
        None
            Nothing.

        Examples
        ________
        >>> from ads.common.model_artifact import ModelArtifact
        >>> import ocifs
        >>> import oci
        >>> storage_options = {"config": oci.config.from_file(os.path.join("~/.oci", "config"))}
        >>> storage_options
        {'log_requests': False,
            'additional_user_agent': '',
            'pass_phrase': None,
            'user': 'ocid5.user.oc1..aaaaaaaab3geixlk***********************',
            'fingerprint': '05:15:2b:b1:46:8a:32:ec:e2:69:5b:32:01:**:**:**)',
            'tenancy': 'ocid5.tenancy.oc1..aaaaaaaag*************************',
            'region': 'us-ashburn-1',
            'key_file': '/home/datascience/.oci/oci_api_key.pem'}
        >>> path_to_generic_model_artifact = tempfile.mkdtemp()
        >>> generic_model_artifact = prepare_generic_model(path_to_generic_model_artifact,
        ...                                   force_overwrite=True, data_science_env=True,
        ...                                   ignore_deployment_error=True)
        >>> generic_model_artifact._save_data_from_file(oci_storage_path = 'oci://bucket_name@namespace/folder_name',
        ... train_data_path = '~/orcl_attrition*.csv', storage_options=storage_options)
        """
        if not re.match(r"oci://*@*", prefix):
            raise InvalidObjectStoragePath(
                "`prefix` is not valid. It must have the pattern 'oci://bucket_name@namespace/key'."
            )
        if not storage_options:
            storage_options = factory.default_storage_options
        if not storage_options:
            storage_options = {"config": {}}

        if train_data_path is not None:
            assert isinstance(train_data_path, str), "A path to the data is required."
            self._save_from_local_file(
                prefix=prefix,
                file_path=train_data_path,
                storage_options=storage_options,
                data_type="training",
            )
        if validation_data_path is not None:
            assert isinstance(
                validation_data_path, str
            ), "A path to the data is required."
            self._save_from_local_file(
                prefix=prefix,
                file_path=validation_data_path,
                storage_options=storage_options,
                data_type="validation",
            )
        if train_data_path is None and validation_data_path is None:
            data_type = kwargs.get("data_type", "training")
            if data_type not in ("training", "validation"):
                InvalidDataType(
                    "`data_type` is not supported. Choose 'training' or 'validation'."
                )
            self._save_data_path(prefix, data_type=data_type)

    def _populate_metadata_taxonomy(self, model=None, use_case_type=None):
        """Extract and populate the taxonomy metadata from the model.

        Parameters
        ----------
        model: [sklearn, xgboost, lightgbm, automl, keras]
            The model object.
        use_case_type: str
            The use case type of the model.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        ValueError: When model not provided.
        """
        if use_case_type and use_case_type not in UseCaseType:
            raise ValueError(
                f"Invalid value of `UseCaseType`. Choose from {UseCaseType.values()}."
            )

        self.metadata_taxonomy[MetadataTaxonomyKeys.USE_CASE_TYPE].value = use_case_type
        if model is not None:
            map = ModelInfoExtractorFactory.extract_info(model)
            if map is not None:
                self.metadata_taxonomy._populate_from_map(map)
            if (
                self.metadata_taxonomy[MetadataTaxonomyKeys.HYPERPARAMETERS].size()
                > METADATA_SIZE_LIMIT
            ):
                logger.warn(
                    f"The model hyperparameters are larger than `{METADATA_SIZE_LIMIT}` "
                    "bytes and cannot be stored as model catalog metadata. It will be saved to "
                    f"{self.artifact_dir}/hyperparameters.json and removed from the metadata."
                )

                self.metadata_taxonomy[
                    MetadataTaxonomyKeys.HYPERPARAMETERS
                ].to_json_file(self.artifact_dir)
                self.metadata_taxonomy[MetadataTaxonomyKeys.HYPERPARAMETERS].update(
                    value=None
                )

    def _populate_metadata_custom(self):
        """Extracts custom metadata from the model artifact.

        Returns
        -------
        None
            Nothing
        """
        model_metadata_items = []

        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.CONDA_ENVIRONMENT,
                value=self.conda_env if hasattr(self, "conda_env") else None,
                description="The conda environment where the model was trained.",
                category=MetadataCustomCategory.TRAINING_ENV,
            )
        )
        try:
            env_type = (
                self._runtime_info.MODEL_DEPLOYMENT.INFERENCE_CONDA_ENV.INFERENCE_ENV_TYPE._value
            )
        except:
            env_type = None
        try:
            slug_name = (
                self._runtime_info.MODEL_DEPLOYMENT.INFERENCE_CONDA_ENV.INFERENCE_ENV_SLUG._value
            )
        except:
            slug_name = None
        try:
            env_path = (
                self._runtime_info.MODEL_DEPLOYMENT.INFERENCE_CONDA_ENV.INFERENCE_ENV_PATH._value
            )
        except:
            env_path = None

        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.ENVIRONMENT_TYPE,
                value=env_type,
                description="The environment type, must be a 'published' or 'data_science'.",
                category=MetadataCustomCategory.TRAINING_ENV,
            )
        )
        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.SLUG_NAME,
                value=slug_name,
                description="The slug name of the training conda environment.",
                category=MetadataCustomCategory.TRAINING_ENV,
            )
        )
        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.CONDA_ENVIRONMENT_PATH,
                value=env_path,
                description="The oci path of the training conda environment.",
                category=MetadataCustomCategory.TRAINING_ENV,
            )
        )
        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.MODEL_ARTIFACTS,
                value=textwrap.shorten(
                    ", ".join(self._get_files()), 255, placeholder="..."
                ),
                description="A list of files located in the model artifacts folder.",
                category=MetadataCustomCategory.TRAINING_ENV,
            )
        )
        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.MODEL_SERIALIZATION_FORMAT,
                value=self._serialization_format,
                description="The model serialization format.",
                category=MetadataCustomCategory.TRAINING_PROFILE,
            )
        )
        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.CLIENT_LIBRARY,
                value="ADS",
                description="",
                category=MetadataCustomCategory.OTHER,
            )
        )
        self.metadata_custom._add_many(model_metadata_items, replace=True)

    def populate_metadata(self, model=None, use_case_type=None):
        """Extracts and populate taxonomy metadata from given model.

        Parameters
        ----------
        model: [sklearn, xgboost, lightgbm, automl, keras]
            The model object.

        use_case_type:
            The use case type of the model.
        model: (Any, optional). Defaults to None.
            This is an optional model object which is only used to extract taxonomy metadata.
            Supported models: keras, lightgbm, pytorch, sklearn, tensorflow, and xgboost.
            If the model is not under supported frameworks, then extracting taxonomy metadata will be skipped.
        use_case_type: (str, optional). Default to None.
            The use case type of the model.

        Returns
        -------
        None
            Nothing.
        """
        if model is None and self.metadata_taxonomy["Algorithm"].value is None:
            logger.info(
                "To auto-extract taxonomy metadata the model must be provided. Supported models: automl, keras, lightgbm, pytorch, sklearn, tensorflow, and xgboost."
            )
        if use_case_type is None:
            use_case_type = self.metadata_taxonomy[
                MetadataTaxonomyKeys.USE_CASE_TYPE
            ].value
        self._populate_metadata_taxonomy(model, use_case_type)
        self._populate_metadata_custom()

    def _save_from_memory(
        self,
        data,
        prefix,
        data_file_name,
        storage_options,
        data_type="training",
        **kwargs,
    ):
        """
        Save the data to Object Storage.
        """
        oci_storage_path = os.path.join(prefix, data_file_name)
        if isinstance(data, np.ndarray) or isinstance(data, list):
            data = pd.DataFrame(data)
            data.to_csv(oci_storage_path, storage_options=storage_options, **kwargs)
        elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            data.to_csv(oci_storage_path, storage_options=storage_options, **kwargs)
        elif isinstance(data, ADSData):
            data = pd.concat([data.X, data.y], axis=1)
            data.to_csv(oci_storage_path, storage_options=storage_options, **kwargs)
        else:
            raise NotImplementedError(
                f"`{type(data)}` is not supported. Use a Pandas DataFrame."
            )

        self._save_data_path(oci_storage_path, data_type)
        self._save_data_shape(data, data_type)

    def _save_from_local_file(
        self, prefix, file_path, storage_options, data_type="training"
    ):
        """Save local file to Object Storage."""
        file_path = os.path.expanduser(file_path)
        import glob

        if len(glob.glob(file_path)) == 0:
            raise FileExistsError(f"No files were found in `{file_path}`.")
        oci_storage_paths = []
        with fsspec.open_files(file_path, mode="r") as fhs:
            for fh in fhs:
                oci_storage_path = os.path.join(prefix, os.path.basename(fh.name))
                with fsspec.open(
                    oci_storage_path,
                    mode="w",
                    **(storage_options),
                ) as f:
                    f.write(fh.read())
                oci_storage_paths.append(oci_storage_path)
                self._save_file_size(
                    os.path.join(os.path.dirname(file_path), os.path.basename(fh.name)),
                    data_type,
                )
        self._save_data_path(",  ".join(oci_storage_paths), data_type)

    def _save_data_path(self, oci_storage_path, data_type):

        key = (
            MetadataCustomKeys.TRAINING_DATASET
            if data_type == "training"
            else MetadataCustomKeys.VALIDATION_DATASET
        )
        self.metadata_custom._add(
            ModelCustomMetadataItem(
                key=key,
                value=oci_storage_path,
                description=f"The path to where the {data_type} dataset is stored on Object Storage.",
                category=MetadataCustomCategory.TRAINING_AND_VALIDATION_DATASETS,
            ),
            replace=True,
        )

    def _save_data_shape(self, data, data_type):
        key = (
            MetadataCustomKeys.TRAINING_DATASET_SIZE
            if data_type == "training"
            else MetadataCustomKeys.VALIDATION_DATASET_SIZE
        )
        self.metadata_custom._add(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.TRAINING_DATASET_SIZE
                if data_type == "training"
                else MetadataCustomKeys.VALIDATION_DATASET_SIZE,
                value=str(data.shape),
                description=f"The size of the {data_type} dataset in bytes.",
                category=MetadataCustomCategory.TRAINING_AND_VALIDATION_DATASETS,
            ),
            replace=True,
        )

    def _save_file_size(self, file_path, data_type):
        self.metadata_custom._add(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.TRAINING_DATASET_SIZE
                if data_type == "training"
                else MetadataCustomKeys.VALIDATION_DATASET_SIZE,
                value=str(os.stat(file_path).st_size) + " bytes",
                description=f"The {data_type} dataset size in bytes.",
                category=MetadataCustomCategory.TRAINING_AND_VALIDATION_DATASETS,
            ),
            replace=True,
        )

    def _prepare_data_for_schema(
        self,
        X_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        y_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
    ):
        """
        Any Framework-specific work before generic schema generation.
        """
        return X_sample, y_sample

    def populate_schema(
        self,
        data_sample: ADSData = None,
        X_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        y_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        max_col_num: int = DATA_SCHEMA_MAX_COL_NUM,
    ):
        """
        Populate the input and output schema.
        If the schema exceeds the limit of 32kb, save as json files to the artifact directory.

        Parameters
        ----------
        data_sample: ADSData
            A sample of the data that will be used to generate input_schema and output_schema.
        X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]
            A sample of input data that will be used to generate input schema.
        y_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]
            A sample of output data that will be used to generate output schema.
        max_col_num: (int, optional). Defaults to utils.DATA_SCHEMA_MAX_COL_NUM.
            The maximum column size of the data that allows to auto generate schema.
        """
        if data_sample is not None:
            assert isinstance(
                data_sample, ADSData
            ), "`data_sample` expects data of ADSData type. \
            Pass in to `X_sample` and `y_sample` for other data types."
            X_sample = data_sample.X
            y_sample = data_sample.y
        X_sample, y_sample = self._prepare_data_for_schema(X_sample, y_sample)
        self.schema_input = self._populate_schema(
            X_sample,
            schema_file_name=INPUT_SCHEMA_FILE_NAME,
            max_col_num=max_col_num,
        )
        self.schema_output = self._populate_schema(
            y_sample,
            schema_file_name=OUTPUT_SCHEMA_FILE_NAME,
            max_col_num=max_col_num,
        )

    def _populate_schema(
        self,
        data: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame],
        schema_file_name: str,
        max_col_num: int,
    ):
        """
        Populate schema and if the schema exceeds the limit of 32kb, save as a json file to artifact_dir.

        Parameters
        ----------
        data: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]
            A sample of input data that will be used to generate input schema.
        schema_file_name: str
            schema file name to be saved as.
        max_col_num : int
            The maximum column size of the data that allows to auto generate schema.

        Returns
        -------
        Schema
            The schema.
        """
        result = None

        try:
            if data is not None:
                data = utils.to_dataframe(data)
                schema = data.ads.model_schema(max_col_num=max_col_num)
                schema.to_json_file(os.path.join(self.artifact_dir, schema_file_name))
                if self._validate_schema_size(schema, schema_file_name):
                    result = schema
        except DataSizeTooWide:
            logger.warning(
                f"The data has too many columns and "
                f"the maximum allowable number of columns is `{max_col_num}`. "
                "The schema was not auto generated. Increase allowable number of columns."
            )

        return result or Schema()

    def _validate_schema_size(self, schema, schema_file_name):
        result = False
        try:
            result = schema.validate_size()
        except SchemaSizeTooLarge:
            logger.warn(
                f"The {schema_file_name.replace('.json', '')} is larger than "
                f"`{METADATA_SIZE_LIMIT}` bytes and cannot be stored as model catalog metadata."
                f"It will be saved to {self.artifact_dir}/{schema_file_name}."
            )

        return result

    def introspect(self) -> pd.DataFrame:
        """Runs model introspection.

        Returns
        -------
        pd.DataFrame
           The introspection result in a dataframe format.
        """
        return self._introspect()

    @classmethod
    def from_model_catalog(
        cls,
        model_id: str,
        artifact_dir: str,
        model_file_name: Optional[str] = "model.onnx",
        auth: Optional[Dict] = None,
        force_overwrite: Optional[bool] = False,
        install_libs: Optional[bool] = False,
        conflict_strategy=ConflictStrategy.IGNORE,
        bucket_uri: Optional[str] = None,
        remove_existing_artifact: Optional[bool] = True,
        **kwargs,
    ) -> "ModelArtifact":
        """Download model artifacts from the model catalog to the target artifact directory.

        Parameters
        ----------
        model_id: str
            The model OCID.
        artifact_dir: str
            The artifact directory to store the files needed for deployment.
            Will be created if not exists.
        model_file_name: (str, optional). Defaults to "model.onnx".
            The name of the serialized model.
        auth: (Dict, optional). Defaults to None.
            Default authetication is set using the `ads.set_auth()` method.
            Use the `ads.common.auth.api_keys()` or `ads.common.auth.resource_principal()` to create appropriate
            authentication signer and kwargs required to instantiate a IdentityClient object.
        force_overwrite: (bool, optional). Defaults to False.
            Overwrite existing files.
        install_libs: bool, default: False
            Install the libraries specified in ds-requirements.txt.
        conflict_strategy: ConflictStrategy, default: IGNORE
           Determines how to handle version conflicts between the current environment and requirements of
           model artifact.
           Valid values: "IGNORE", "UPDATE" or ConflictStrategy.
           IGNORE: Use the installed version in  case of conflict
           UPDATE: Force update dependency to the version required by model artifact in case of conflict
        bucket_uri: (str, optional). Defaults to None.
            The OCI Object Storage URI where model artifacts will be copied to.
            The `bucket_uri` is only necessary for downloading large artifacts with
            size is greater than 2GB. Example: `oci://<bucket_name>@<namespace>/prefix/`.
        remove_existing_artifact: (bool, optional). Defaults to `True`.
            Whether artifacts uploaded to object storage bucket need to be removed or not.
        kwargs:
            compartment_id: (str, optional)
                Compartment OCID. If not specified, the value will be taken from the environment variables.
            timeout: (int, optional). Defaults to 10 seconds.
                The connection timeout in seconds for the client.

        Returns
        -------
        ModelArtifact
            An instance of ModelArtifact class.
        """
        from ads.catalog.model import ModelCatalog

        auth = auth or authutil.default_signer()
        artifact_dir = os.path.abspath(os.path.expanduser(artifact_dir))

        model_catalog = ModelCatalog(
            compartment_id=kwargs.pop("compartment_id", _COMPARTMENT_OCID),
            ds_client_auth=auth,
            identity_client_auth=auth,
            timeout=kwargs.pop("timeout", None),
        )

        model_catalog._download_artifact(
            model_id=model_id,
            target_dir=artifact_dir,
            force_overwrite=force_overwrite,
            bucket_uri=bucket_uri,
            remove_existing_artifact=remove_existing_artifact,
        )
        oci_model = model_catalog.get_model(model_id)

        result_artifact = cls(
            artifact_dir=artifact_dir,
            conflict_strategy=conflict_strategy,
            install_libs=install_libs,
            reload=False,
            model_file_name=model_file_name,
        )

        result_artifact.metadata_custom = oci_model.metadata_custom
        result_artifact.metadata_taxonomy = oci_model.metadata_taxonomy
        result_artifact.schema_input = oci_model.schema_input
        result_artifact.schema_output = oci_model.schema_output

        if not install_libs:
            logger.warning(
                "Libraries in `ds-requirements.txt` were not installed. "
                "Use `install_requirements()` to install the required dependencies."
            )

        return result_artifact


class VersionConflictWarning(object):
    def __init__(self, version_conflicts):
        self.version_conflicts = version_conflicts

    def __str__(self):
        msg = "WARNING: Version conflicts found:"
        if len(self.version_conflicts) > 0:
            for lib in self.version_conflicts:
                msg += "\nInstalled: %s, Required: %s" % (
                    lib,
                    self.version_conflicts[lib],
                )
        return msg


def pip_install(package, options="-U"):
    package = re.sub(r"<|>", "=", package.split(",")[0])
    for output in execute(["pip", "install", options, package]):
        print(output, end="")


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
