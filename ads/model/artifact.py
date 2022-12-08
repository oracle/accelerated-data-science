#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import fnmatch
import importlib
import os
import sys
import shutil
import tempfile
import uuid
from typing import Dict, Optional, Tuple

from ads.common import auth as authutil
from ads.common import logger, utils
from ads.config import CONDA_BUCKET_NAME, CONDA_BUCKET_NS
from ads.model.runtime.env_info import EnvInfo, InferenceEnvInfo, TrainingEnvInfo
from ads.model.runtime.runtime_info import RuntimeInfo
from jinja2 import Environment, PackageLoader

MODEL_ARTIFACT_VERSION = "3.0"
REQUIRED_ARTIFACT_FILES = ("runtime.yaml", "score.py")


class ArtifactNestedFolderError(Exception):
    def __init__(self, folder: str):
        self.folder = folder
        super().__init__("The required artifact files placed in a nested folder.")


class ArtifactRequiredFilesError(Exception):
    def __init__(self, required_files: Tuple[str]):
        super().__init__(
            "Not all required files presented in artifact folder. "
            f"Required files: {required_files}"
        )


class AritfactFolderStructureError(Exception):
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
    ):
        """Initializes a ModelArtifact instance.

        Parameters
        ----------
        artifact_dir: str
            The local artifact folder to store the files needed for deployment.
        model_file_name: (str, optional). Defaults to `None`.
            The file name of the serialized model.
        reload: (bool, optional). Defaults to False.
            Determine whether will reload the Model into the env.

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

        self.score = None
        self.artifact_dir = os.path.abspath(os.path.expanduser(artifact_dir))
        sys.path.insert(0, self.artifact_dir)
        self.model_file_name = model_file_name
        self._env = Environment(loader=PackageLoader("ads", "templates"))
        if reload:
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
            The namespace of region.
        bucketname: (str, optional)
            The bucketname of service pack.

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
        inference_conda_env = ModelArtifact._populate_env_info(
            InferenceEnvInfo,
            conda_pack=inference_conda_env,
            bucketname=bucketname,
            namespace=namespace,
        )

        if training_conda_env:
            training_conda_env = ModelArtifact._populate_env_info(
                TrainingEnvInfo,
                conda_pack=training_conda_env,
                bucketname=bucketname,
                namespace=namespace,
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
            raise ValueError(
                "Cannot automatically detect the inference python version. `inference_python_version` must be provided."
            )
        runtime_file_path = os.path.join(self.artifact_dir, "runtime.yaml")
        if os.path.exists(runtime_file_path) and not force_overwrite:
            raise ValueError(
                "runtime.yaml already exists. "
                "Set `force_overwrite` to True to overwrite all the files."
            )
        else:
            runtime_info.save()
        return runtime_info

    @staticmethod
    def _populate_env_info(
        clss: EnvInfo, conda_pack: str, bucketname: str = None, namespace: str = None
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

        Returns
        -------
        EnvInfo
            An EnvInfo instance.
        """
        if conda_pack.startswith("oci://"):
            return clss.from_path(conda_pack)
        return clss.from_slug(
            env_slug=conda_pack, bucketname=bucketname, namespace=namespace
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
        context = {
            "model_file_name": self.model_file_name,
            "use_torch_script": kwargs.get("use_torch_script", False),
        }
        with open(os.path.join(self.artifact_dir, "score.py"), "w") as sfl:
            sfl.write(scorefn_template.render(context))

    def reload(self):
        """Syncs the `score.py` to reload the model and predict function.

        Returns
        -------
        None
            Nothing
        """
        spec = importlib.util.spec_from_file_location(
            "score%s" % uuid.uuid4(), os.path.join(self.artifact_dir, "score.py")
        )
        self.score = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.score)
        self.model = self.score.load_model()  # load model in cache
        # remove the cache files.
        for dir in [
            os.path.join(self.artifact_dir, "__pycache__"),
            os.path.join(self.artifact_dir, ".ipynb_checkpoints"),
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
        """
        artifact_dir = os.path.join(
            os.path.abspath(os.path.expanduser(artifact_dir)), ""
        )

        if artifact_dir == os.path.join(
            os.path.abspath(os.path.expanduser(uri)).rstrip("/"), ""
        ):
            if not os.path.exists(artifact_dir):
                raise ValueError("Provided `uri` doesn't exist.")
        else:
            auth = auth or authutil.default_signer()
            utils.copy_from_uri(
                uri=uri,
                to_path=artifact_dir,
                unpack=True,
                force_overwrite=force_overwrite,
                auth=auth,
            )
        try:
            _validate_artifact_dir(artifact_dir)
        except ArtifactNestedFolderError as exc:
            with tempfile.TemporaryDirectory() as temp_dir:
                utils.copy_from_uri(
                    uri=exc.folder, to_path=temp_dir, force_overwrite=True
                )
                utils.copy_from_uri(
                    uri=temp_dir, to_path=artifact_dir, force_overwrite=True
                )

        return cls(artifact_dir, model_file_name, reload=True)

    def __getattr__(self, item):
        """Makes the functions in `score.py` directly accessable by ModelArtifact class."""
        return getattr(self.score, item)
