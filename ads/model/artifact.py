#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import fnmatch
import importlib
import os
import sys
import shutil
import tempfile
import uuid
import fsspec
from typing import Dict, Optional, Tuple
from ads.common import auth as authutil
from ads.common import logger, utils
from ads.common.object_storage_details import ObjectStorageDetails
from ads.config import CONDA_BUCKET_NAME, CONDA_BUCKET_NS
from ads.model.runtime.env_info import EnvInfo, InferenceEnvInfo, TrainingEnvInfo
from ads.model.runtime.runtime_info import RuntimeInfo
from jinja2 import Environment, PackageLoader
import warnings
from ads import __version__
from datetime import datetime

MODEL_ARTIFACT_VERSION = "3.0"
REQUIRED_ARTIFACT_FILES = ("runtime.yaml", "score.py")
SCORE_VERSION = "1.0"
ADS_VERSION = __version__


class ArtifactNestedFolderError(Exception):   # pragma: no cover
    def __init__(self, folder: str):
        self.folder = folder
        super().__init__("The required artifact files placed in a nested folder.")


class ArtifactRequiredFilesError(Exception):   # pragma: no cover
    def __init__(self, required_files: Tuple[str]):
        super().__init__(
            "Not all required files presented in artifact folder. "
            f"Required files for conda runtime: {required_files}. If you are using container runtime, set `ignore_conda_error=True`."
        )


class AritfactFolderStructureError(Exception):   # pragma: no cover
    def __init__(self, required_files: Tuple[str]):
        super().__init__(
            "The artifact folder has a wrong structure. "
            f"Required files: {required_files}"
        )


def _validate_artifact_dir(
    artifact_dir: str, required_files: Tuple[str] = REQUIRED_ARTIFACT_FILES
) -> bool:
    """The function helper to validate artifacts folder structure.

    Params
    ------
        artifact_dir: str
            The local artifact folder to store the files needed for deployment.
        required_files: (Tuple[str], optional). Defaults to ("runtime.yaml", "score.py").
            The list of required artifact files.

    Raises:
        ValueError
            If `required_files` not provided.
            If `artifact_dir` not exists.
        ArtifactNestedFolderError
            If artifact files located in a nested folder.
        ArtifactRequiredFilesError
            If not all required files found in artifact folder.
        AritfactFolderStructureError
            In case if artifact folder has a wrong structure.

    Returns:
        bool: True if artifact folder contains the list of the all required files.
    """
    if not required_files or len(required_files) == 0:
        raise ValueError("Required artifact files not provided.")

    artifact_dir = os.path.abspath(os.path.expanduser(artifact_dir))
    if not os.path.exists(artifact_dir):
        raise ValueError(f"The path `{artifact_dir}` not found.")

    result = {required_file.lower(): None for required_file in required_files}
    for dirpath, _, filenames in os.walk(artifact_dir):
        rel_path = os.path.abspath(dirpath)
        for required_file in required_files:
            for filename in fnmatch.filter(filenames, required_file):
                if filename.lower() in result and result[filename] == None:
                    result[filename] = rel_path

    # if not required artifact files found in provided artifact dir
    if None in result.values():
        raise ArtifactRequiredFilesError(required_files)
    # if required artifact files placed in different nested folders
    if len(set(result.values())) > 1:
        raise AritfactFolderStructureError(required_files)

    if all(path == artifact_dir for path in result.values()):
        return True

    # if required files are placed in a nested folder
    raise (ArtifactNestedFolderError(list(result.values())[0]))


class ModelArtifact:
    """The class that represents model artifacts.
    It is designed to help to generate and manage model artifacts.
    """

    def __init__(
        self,
        artifact_dir: str,
        model_file_name: str = None,
        reload: Optional[bool] = False,
        ignore_conda_error: Optional[bool] = False,
        local_copy_dir: str = None,
        auth: dict = None,
    ):
        """Initializes a ModelArtifact instance.

        Parameters
        ----------
        artifact_dir: str
            The artifact folder to store the files needed for deployment.
        model_file_name: (str, optional). Defaults to `None`.
            The file name of the serialized model.
        reload: (bool, optional). Defaults to False.
            Determine whether will reload the Model into the env.
        ignore_conda_error: (bool, optional). Defaults to False.
            Parameter to ignore error when collecting conda information.
        local_copy_dir: (str, optional). Defaults to None.
            The local back up directory of the model artifacts.
        auth :(Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        ModelArtifact
            A ModelArtifact instance.

        Raises
        ------
        ValueError
            If `artifact_dir` not provided.
        """
        if not artifact_dir:
            raise ValueError("The `artifact_dir` needs to be provided.")

        self.artifact_dir = (
            artifact_dir
            if ObjectStorageDetails.is_oci_path(artifact_dir)
            else os.path.abspath(os.path.expanduser(artifact_dir))
        )
        self.local_copy_dir = (
            local_copy_dir or tempfile.mkdtemp()
            if ObjectStorageDetails.is_oci_path(artifact_dir)
            else artifact_dir
        )

        self.score = None
        sys.path.insert(0, self.artifact_dir)
        self.model_file_name = model_file_name
        self._env = Environment(loader=PackageLoader("ads", "templates"))
        self.ignore_conda_error = ignore_conda_error
        self.model = None
        self.auth = auth or authutil.default_signer()
        if reload and not ignore_conda_error:
            self.reload()
            # Extracts the model_file_name from the score.py.
            if (
                not self.model_file_name
                and self.score
                and hasattr(self.score, "model_name")
                and self.score.model_name
            ):
                self.model_file_name = self.score.model_name

    def prepare_runtime_yaml(
        self,
        inference_conda_env: str,
        inference_python_version: str = None,
        training_conda_env: str = None,
        training_python_version: str = None,
        force_overwrite: bool = False,
        namespace: str = CONDA_BUCKET_NS,
        bucketname: str = CONDA_BUCKET_NAME,
        auth: dict = None,
        ignore_conda_error: bool = False,
    ) -> None:
        """Generate a runtime yaml file and save it to the artifact
        directory.

        Parameters
        ----------
        inference_conda_env: (str, optional). Defaults to None.
            The object storage path of conda pack which will be used in deployment.
            Can be either slug or object storage path of the conda pack.
            You can only pass in slugs if the conda pack is a service pack.
        inference_python_version: (str, optional). Defaults to None.
            The python version which will be used in deployment.
        training_conda_env: (str, optional). Defaults to None.
            The object storage path of conda pack used during training.
            Can be either slug or object storage path of the conda pack.
            You can only pass in slugs if the conda pack is a service pack.
        training_python_version: (str, optional). Defaults to None.
            The python version used during training.
        force_overwrite : (bool, optional). Defaults to False.
            Whether to overwrite existing files.
        namespace: (str, optional)
            The namespace of region. Defaults to environment variable CONDA_BUCKET_NS.
        bucketname: (str, optional)
            The bucketname of service pack. Defaults to environment variable CONDA_BUCKET_NAME.
        auth :(Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Raises
        ------
        ValueError
            If neither slug or conda_env_uri is provided.

        Returns
        -------
        RuntimeInfo
            A RuntimeInfo instance.
        """
        runtime_info = RuntimeInfo.from_env()
        runtime_info.model_artifact_version = MODEL_ARTIFACT_VERSION
        if ignore_conda_error:
            runtime_info.model_provenance.training_code.artifact_directory = (
                self.artifact_dir
            )
            runtime_info.save(storage_options=auth)
            return runtime_info
        inference_conda_env = ModelArtifact._populate_env_info(
            InferenceEnvInfo,
            conda_pack=inference_conda_env,
            bucketname=bucketname,
            namespace=namespace,
            auth=auth,
        )

        if training_conda_env:
            training_conda_env = ModelArtifact._populate_env_info(
                TrainingEnvInfo,
                conda_pack=training_conda_env,
                bucketname=bucketname,
                namespace=namespace,
                auth=auth,
            )
        else:
            training_conda_env = TrainingEnvInfo()
        if training_python_version:
            training_conda_env.training_python_version = training_python_version
        if inference_python_version:
            inference_conda_env.inference_python_version = inference_python_version
        runtime_info.model_deployment.inference_conda_env = inference_conda_env
        runtime_info.model_provenance.training_conda_env = training_conda_env
        runtime_info.model_provenance.training_code.artifact_directory = (
            self.artifact_dir
        )
        if (
            not runtime_info.model_deployment.inference_conda_env.inference_python_version
            or runtime_info.model_deployment.inference_conda_env.inference_python_version.strip()
            == ""
        ):
            warnings.warn(
                "Cannot automatically detect the inference python version. `inference_python_version` must be provided."
            )
        runtime_file_path = os.path.join(self.artifact_dir, "runtime.yaml")
        if os.path.exists(runtime_file_path) and not force_overwrite:
            raise ValueError(
                "runtime.yaml already exists. "
                "Set `force_overwrite` to True to overwrite all the files."
            )
        else:
            runtime_info.save(storage_options=auth)
        return runtime_info

    @staticmethod
    def _populate_env_info(
        clss: EnvInfo,
        conda_pack: str,
        bucketname: str = None,
        namespace: str = None,
        auth: dict = None,
    ) -> "EnvInfo":
        """Populates the Training/InferenceEnvInfo instance.

        Parameters
        ----------
        clss: EnvInfo
            A EnvInfo class.
        conda_pack: str
            The object storage path of conda pack.
            Can be either slug or object storage path of the conda pack.
            You can only pass in slugs if the conda pack is a service pack.
        namespace: (str, optional)
            The namespace of region.
        bucketname: (str, optional)
            The bucketname of service pack.
        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        EnvInfo
            An EnvInfo instance.
        """
        if conda_pack.startswith("oci://"):
            return clss.from_path(conda_pack)
        return clss.from_slug(
            env_slug=conda_pack, bucketname=bucketname, namespace=namespace, auth=auth
        )

    def prepare_score_py(
        self, jinja_template_filename: str, model_file_name: str = None, **kwargs
    ):
        """Prepares `score.py` file.

        Parameters
        ----------
        jinja_template_filename: str.
            The jinja template file name.
        model_file_name: (str, optional). Defaults to `None`.
            The file name of the serialized model.
        **kwargs: (dict)
            use_torch_script: bool
            data_deserializer: str

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `model_file_name` not provided.
        """
        self.model_file_name = model_file_name or self.model_file_name
        if not self.model_file_name:
            raise ValueError("The `model_file_name` must be provided.")

        if not os.path.exists(
            os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                "templates",
                f"{jinja_template_filename}.jinja2",
            )
        ):
            raise FileExistsError(f"{jinja_template_filename}.jinja2 does not exists.")
        scorefn_template = self._env.get_template(f"{jinja_template_filename}.jinja2")
        time_suffix = datetime.today().strftime("%Y%m%d_%H%M%S")

        context = {
            "model_file_name": self.model_file_name,
            "SCORE_VERSION": SCORE_VERSION,
            "ADS_VERSION": ADS_VERSION,
            "time_created": time_suffix,
        }
        storage_options = kwargs.pop("auth", {})
        storage_options = storage_options if storage_options else {}
        context.update(kwargs)
        with fsspec.open(
            os.path.join(self.artifact_dir, "score.py"), "w", **storage_options
        ) as f:
            f.write(scorefn_template.render(context))

    def reload(self):
        """Syncs the `score.py` to reload the model and predict function.

        Returns
        -------
        None
            Nothing

        """
        if ObjectStorageDetails.is_oci_path(self.artifact_dir):
            utils.copy_from_uri(
                uri=self.artifact_dir,
                to_path=self.local_copy_dir,
                force_overwrite=True,
                auth=self.auth,
            )

        spec = importlib.util.spec_from_file_location(
            "score%s" % uuid.uuid4(), os.path.join(self.local_copy_dir, "score.py")
        )
        self.score = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.score)
        self.model = self.score.load_model()  # load model in cache
        # remove the cache files.
        for dir in [
            os.path.join(self.local_copy_dir, "__pycache__"),
            os.path.join(self.local_copy_dir, ".ipynb_checkpoints"),
        ]:
            if os.path.exists(dir):
                shutil.rmtree(dir, ignore_errors=True)

    @classmethod
    def from_uri(
        cls,
        uri: str,
        artifact_dir: str,
        model_file_name: str = None,
        force_overwrite: Optional[bool] = False,
        auth: Optional[Dict] = None,
        ignore_conda_error: Optional[bool] = False,
    ):
        """Constructs a ModelArtifact object from the existing model artifacts.

        Parameters
        ----------
        uri: str
            The URI of source artifact folder or achive. Can be local path or
            OCI object storage URI.
        artifact_dir: str
            The local artifact folder to store the files needed for deployment.
        model_file_name: (str, optional). Defaults to `None`
            The file name of the serialized model.
        force_overwrite: (bool, optional). Defaults to False.
            Whether to overwrite existing files or not.
        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API.
            If you need to override the default, use the `ads.common.auth.api_keys`
            or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate
            IdentityClient object.

        Returns
        -------
        ModelArtifact
            A `ModelArtifact` instance

        Raises
        ------
        ValueError
            If `uri` is equal to `artifact_dir`, and it not exists.
            If `artifact_dir` is not provided.
        """
        if not artifact_dir:
            raise ValueError("The `artifact_dir` needs to be provided.")

        artifact_dir = (
            artifact_dir
            if ObjectStorageDetails.is_oci_path(artifact_dir)
            else os.path.join(os.path.abspath(os.path.expanduser(artifact_dir)), "")
        )

        if not ObjectStorageDetails.is_oci_path(uri):
            uri = os.path.join(os.path.abspath(os.path.expanduser(uri)).rstrip("/"), "")
        auth = auth or authutil.default_signer()

        to_path = (
            tempfile.mkdtemp()
            if ObjectStorageDetails.is_oci_path(artifact_dir)
            else artifact_dir
        )
        force_overwrite = (
            True if ObjectStorageDetails.is_oci_path(artifact_dir) else force_overwrite
        )
        if artifact_dir == uri and not ObjectStorageDetails.is_oci_path(artifact_dir):
            if not utils.is_path_exists(artifact_dir, auth=auth):
                raise ValueError("Provided `uri` doesn't exist.")
        else:
            utils.copy_from_uri(
                uri=uri,
                to_path=to_path,
                unpack=True,
                force_overwrite=force_overwrite,
                auth=auth,
            )

        if not ignore_conda_error:
            try:
                _validate_artifact_dir(to_path)
            except ArtifactNestedFolderError as exc:
                with tempfile.TemporaryDirectory() as temp_dir:
                    utils.copy_from_uri(
                        uri=exc.folder, to_path=temp_dir, force_overwrite=True
                    )
                    utils.copy_from_uri(
                        uri=temp_dir, to_path=to_path, force_overwrite=True
                    )

        if ObjectStorageDetails.is_oci_path(artifact_dir):
            for root, dirs, files in os.walk(to_path):
                prefix = (os.path.abspath(root).split(to_path)[-1]).lstrip("/")
                for file in files:
                    path = os.path.join(prefix, file)
                    utils.copy_file(
                        uri_src=os.path.join(root, file),
                        uri_dst=os.path.join(artifact_dir, path),
                        force_overwrite=True,
                        auth=auth,
                    )

        return cls(
            artifact_dir=artifact_dir,
            model_file_name=model_file_name,
            reload=True,
            ignore_conda_error=ignore_conda_error,
            local_copy_dir=to_path,
        )

    def __getattr__(self, item):
        """Makes the functions in `score.py` directly accessable by ModelArtifact class."""

        try:
            return getattr(self.score, item)
        except:
            if self.ignore_conda_error:
                logger.warn("`verify` is not guarenteed to work for byoc case.")
            else:
                raise
