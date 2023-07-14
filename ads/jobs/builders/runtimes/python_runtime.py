#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from __future__ import annotations

import os
from typing import Dict

from ads.common.auth import default_signer
from ads.jobs.builders.runtimes.base import Runtime
from ads.opctl.config.utils import convert_notebook


class CondaRuntime(Runtime):
    """Represents a job runtime with conda pack
    This is the base class for Runtime using conda environment.
    The ``CondaRuntime`` is not designed to be used directly when creating a job.
    """

    CONST_CONDA = "conda"
    CONST_CONDA_TYPE = "type"
    CONST_CONDA_TYPE_SERVICE = "service"
    CONST_CONDA_TYPE_CUSTOM = "published"
    CONST_CONDA_SLUG = "slug"
    CONST_CONDA_URI = "uri"
    CONST_CONDA_REGION = "region"

    attribute_map = {CONST_CONDA: CONST_CONDA}
    attribute_map.update(Runtime.attribute_map)

    @property
    def conda(self) -> dict:
        """The conda environment specification.

        For service conda environment, the specification contains:

        * ``type``, the type of the conda environment.
          This is always ``service`` for service conda environment.
        * ``slug``, the slug of the conda environment.

        For custom conda environment, the specification contains:

        * ``type``, the type of the conda environment.
          This is always ``published`` for custom conda environment.
        * ``uri``, the uri of the conda environment, e.g. oci://bucket@namespace/prefix/to/conda
        * ``region``, the region of the bucket in which the conda environment is stored.
          By default, ADS will determine the region based on the authenticated API key or resource principal.
          This is only needed if your conda environment is stored in a different region.

        Returns
        -------
        dict
            A dictionary containing the conda environment specifications.

        """
        return self.get_spec(self.CONST_CONDA)

    def with_service_conda(self, slug: str):
        """Specifies the service conda pack for running the job

        Parameters
        ----------
        slug : str
            The slug name of the service conda pack

        Returns
        -------
        self
            The runtime instance.
        """
        return self.set_spec(
            self.CONST_CONDA,
            {
                self.CONST_CONDA_TYPE: self.CONST_CONDA_TYPE_SERVICE,
                self.CONST_CONDA_SLUG: slug,
            },
        )

    def with_custom_conda(self, uri: str, region: str = None):
        """Specifies the custom conda pack for running the job
        Make sure you have configured the IAM policy for the job run to access the conda environment.

        Parameters
        ----------
        uri : str
            The OCI object storage URI for the conda pack,
            e.g. "oci://your_bucket@namespace/object_name."
            In the Environment Explorer of an OCI notebook session,
            this is shown as the "source" of the conda pack.
        region: str, optional
            The region of the bucket storing the custom conda pack, by default None.
            If region is not specified, ADS will use the region from your authentication credentials:

            * For API Key, config["region"] is used.
            * For Resource Principal, signer.region is used.

            This is required if the conda pack is stored in a different region.

        Returns
        -------
        self
            The runtime instance.

        See Also
        --------
        https://docs.oracle.com/en-us/iaas/data-science/using/conda_publishs_object.htm

        """
        conda_spec = {
            self.CONST_CONDA_TYPE: self.CONST_CONDA_TYPE_CUSTOM,
            self.CONST_CONDA_URI: uri,
        }
        if region:
            conda_spec[self.CONST_CONDA_REGION] = region
        return self.set_spec(self.CONST_CONDA, conda_spec)

    def init(self) -> "CondaRuntime":
        """Initializes a starter specification for the runtime.

        Returns
        -------
        CondaRuntime
            The runtime instance.
        """
        super().init()
        return self.with_custom_conda(
            "{Path to the custom conda environment. "
            "Example: oci://your_bucket@namespace/object_name"
        )


class ScriptRuntime(CondaRuntime):
    """Represents job runtime with scripts and conda pack.

    This runtime is designed to define job artifacts and configurations supported by OCI Data Science Jobs natively.
    It can be used with any script types that is supported by the OCI Data Science Jobs,
    including shell scripts and python scripts.

    To run a script with all dependencies contained in a local folder::

        runtime = (
            ScriptRuntime()
            # Specify the service conda environment by slug name.
            .with_service_conda("pytorch110_p38_cpu_v1")
            # The job artifact can be a single Python script, a directory or a zip file.
            .with_source("local/path/to/code_dir")
            # Environment variable
            .with_environment_variable(NAME="Welcome to OCI Data Science.")
            # Command line argument
            .with_argument("100 linux 'hi there'")
            # The entrypoint is applicable only to directory or zip file as source
            # The entrypoint should be a path relative to the working dir.
            # Here my_script.sh is a file in the code_dir/my_package directory
            .with_entrypoint("my_package/my_script.sh")
        )


    References
    ----------
    https://docs.oracle.com/en-us/iaas/data-science/using/jobs-artifact.htm

    """

    CONST_ENTRYPOINT = "entrypoint"
    CONST_SCRIPT_PATH = "scriptPathURI"

    attribute_map = {
        CONST_ENTRYPOINT: CONST_ENTRYPOINT,
        CONST_SCRIPT_PATH: "script_path_uri",
    }
    attribute_map.update(CondaRuntime.attribute_map)

    @property
    def script_uri(self) -> str:
        """The URI of the source code"""
        return self.get_spec(self.CONST_SCRIPT_PATH)

    def with_script(self, uri: str):
        """Specifies the source code script for the job

        Parameters
        ----------
        uri : str
            URI to the source code script, which can be any URI supported by fsspec,
            including http://, https:// and OCI object storage.
            For example: oci://your_bucket@your_namespace/path/to/script.py

        Returns
        -------
        self
            The runtime instance.
        """
        return self.set_spec(self.CONST_SCRIPT_PATH, uri)

    @property
    def source_uri(self) -> str:
        """The URI of the source code"""
        return self.get_spec(self.CONST_SCRIPT_PATH)

    def with_source(self, uri: str, entrypoint: str = None):
        """Specifies the source code for the job

        Parameters
        ----------
        uri : str
            URI to the source code,
            which can be a (.py/.sh) script, a zip/tar file or directory containing the scripts/modules
            If the source code is a single file, URI can be any URI supported by fsspec,
            including http://, https:// and OCI object storage.
            For example: oci://your_bucket@your_namespace/path/to/script.py
            URI can also be a folder or a zip file containing the source code.
            In that case, entrypoint is required.

        entrypoint : str, optional
            The relative path of the script to be set as entrypoint when source is a zip/tar/directory.
            By default None. This is not needed when the source is a single script.

        Returns
        -------
        self
            The runtime instance.
        """
        if entrypoint:
            self.set_spec(self.CONST_ENTRYPOINT, entrypoint)
        return self.with_script(uri)

    @property
    def entrypoint(self) -> str:
        """The relative path of the script to be set as entrypoint when source is a zip/tar/directory."""
        return self.get_spec(self.CONST_ENTRYPOINT)

    def with_entrypoint(self, entrypoint: str):
        """Specify the entrypoint for the job

        Parameters
        ----------
        entrypoint : str
            The relative path of the script to be set as entrypoint when source is a zip/tar/directory.

        Returns
        -------
        self
            The runtime instance.
        """
        return self.set_spec(self.CONST_ENTRYPOINT, entrypoint)

    def init(self) -> "ScriptRuntime":
        """Initializes a starter specification for the runtime.

        Returns
        -------
        ScriptRuntime
            The runtime instance.
        """
        super().init()
        return (
            self.with_entrypoint(
                "{Entrypoint script. For MLflow, it will be replaced with the CMD}"
            )
            .with_script(
                "{Path to the script. For MLflow, it will be replaced with the path to the project}"
            )
            .with_argument(key1="val1")
        )


class _PythonRuntimeMixin(Runtime):
    CONST_OUTPUT_DIR = "outputDir"
    CONST_OUTPUT_URI = "outputUri"
    CONST_PYTHON_PATH = "pythonPath"
    CONST_ENTRYPOINT = "entrypoint"
    CONST_ENTRY_FUNCTION = "entryFunction"
    CONST_WORKING_DIR = "workingDir"

    attribute_map = {
        CONST_OUTPUT_DIR: "output_dir",
        CONST_OUTPUT_URI: "output_uri",
        CONST_PYTHON_PATH: "python_path",
        CONST_ENTRYPOINT: CONST_ENTRYPOINT,
        CONST_ENTRY_FUNCTION: "entry_function",
        CONST_WORKING_DIR: "working_dir",
    }
    attribute_map.update(Runtime.attribute_map)

    def with_output(self, output_dir: str, output_uri: str):
        """Specifies the outputs of the job.
        The output files in output_dir will be copied to remote output_uri when the job is finished.

        Parameters
        ----------
        output_dir : str
            Path to the output directory in the job run.
            This path should be a relative path from the working directory.
            The source code should write all outputs into this directory.
        output_uri : str
            The OCI object storage URI prefix for saving the output files.
            For example, oci://bucket_name@namespace/path/to/directory

        Returns
        -------
        Self
            The runtime instance.
        """
        self.set_spec(self.CONST_OUTPUT_DIR, output_dir)
        self.set_spec(self.CONST_OUTPUT_URI, output_uri)
        return self

    def with_python_path(self, *python_paths):
        """Specifies additional python paths for running the source code.

        Parameters
        ----------
        *python_paths :
            Additional python path(s) for running the source code.
            Each path should be a relative path from the working directory.

        Returns
        -------
        self
            The runtime instance.
        """
        python_paths = list(python_paths)
        for path in python_paths:
            if os.path.isabs(path):
                raise ValueError(
                    f"{path} is an absolute path."
                    "Please specify relative path from the working directory as python path."
                )
        return self.set_spec(self.CONST_PYTHON_PATH, python_paths)

    def with_entrypoint(self, path: str, func: str = None):
        """Specifies the entrypoint for the job.
        The entrypoint can be a script or a function in a script.

        Parameters
        ----------
        script : str
            The relative path for the script/module starting the job.
        func : str, optional
            The function name in the script for starting the job, by default None.
            If this is not specified, the script will be run with python command in a subprocess.

        Returns
        -------
        self
            The runtime instance.
        """
        self.set_spec(self.CONST_ENTRYPOINT, path)
        self.set_spec(self.CONST_ENTRY_FUNCTION, func)
        return self

    def with_working_dir(self, working_dir: str):
        """Specifies the working directory in the job run.
        By default, the working directory will the directory containing the user code (job artifact directory).
        This can be changed by specifying a relative path to the job artifact directory.

        Parameters
        ----------
        working_dir : str
            The path of the working directory.
            This can be a relative path from the job artifact directory.

        Returns
        -------
        self
            The runtime instance.
        """
        return self.set_spec(self.CONST_WORKING_DIR, working_dir)

    @property
    def working_dir(self) -> str:
        """The working directory for the job run."""
        return self.get_spec(self.CONST_WORKING_DIR, ".")

    @property
    def output_dir(self) -> str:
        """Directory in the Job run container for saving output files generated in the job"""
        return self.get_spec(self.CONST_OUTPUT_DIR)

    @property
    def output_uri(self) -> str:
        """OCI object storage URI prefix for saving output files generated in the job"""
        return self.get_spec(self.CONST_OUTPUT_DIR)

    @property
    def python_path(self):
        """Additional python paths for running the source code."""
        return self.get_spec(self.CONST_PYTHON_PATH)

    @property
    def entry_script(self) -> str:
        """The path of the entry script"""
        return self.get_spec(self.CONST_ENTRYPOINT)

    @property
    def entry_function(self) -> str:
        """The name of the entry function in the entry script"""
        return self.get_spec(self.CONST_ENTRY_FUNCTION)


class PythonRuntime(ScriptRuntime, _PythonRuntimeMixin):
    """Represents a job runtime using ADS driver script to run Python code

    Example::

        runtime = (
            PythonRuntime()
            # Specify the service conda environment by slug name.
            .with_service_conda("pytorch110_p38_cpu_v1")
            # The job artifact can be a single Python script, a directory or a zip file.
            .with_source("local/path/to/code_dir")
            # Environment variable
            .with_environment_variable(NAME="Welcome to OCI Data Science.")
            # Command line argument, arg1 --key arg2
            .with_argument("arg1", key="arg2")
            # Set the working directory
            # When using a directory as source, the default working dir is the parent of code_dir.
            # Working dir should be a relative path beginning from the source directory (code_dir)
            .with_working_dir("code_dir")
            # The entrypoint is applicable only to directory or zip file as source
            # The entrypoint should be a path relative to the working dir.
            # Here my_script.py is a file in the code_dir/my_package directory
            .with_entrypoint("my_package/my_script.py")
            # Add an additional Python path, relative to the working dir (code_dir/other_packages).
            .with_python_path("other_packages")
            # Copy files in "code_dir/output" to object storage after job finishes.
            .with_output("output", "oci://bucket_name@namespace/path/to/dir")
        )

    """

    attribute_map = {}
    attribute_map.update(ScriptRuntime.attribute_map)
    attribute_map.update(_PythonRuntimeMixin.attribute_map)

    def init(self) -> "PythonRuntime":
        """Initializes a starter specification for the runtime.

        Returns
        -------
        PythonRuntime
            The runtime instance.
        """
        super().init()
        return (
            self.with_working_dir("{For MLflow the project folder will be used.}")
            .with_entrypoint(
                "{Entrypoint script. For MLflow, it will be replaced with the CMD}"
            )
            .with_script(
                "{Path to the script. For MLflow, it will be replaced with the path to the project}"
            )
        )


class NotebookRuntime(CondaRuntime):
    """Represents a job runtime with Jupyter notebook

    To run a job with a single Jupyter notebook,
    you can define the run time as::

        runtime = (
            NotebookRuntime()
            .with_notebook(
                path="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
                encoding='utf-8'
            )
            .with_service_conda("tensorflow28_p38_cpu_v1")
            .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
            .with_exclude_tag(["ignore", "remove"])
            .with_output("oci://bucket_name@namespace/path/to/dir")
        )

    Note that the notebook path can be local or remote path supported by fsspec,
    including OCI object storage path like ``oci://bucket@namespace/path/to/notebook``

    """

    CONST_NOTEBOOK_PATH = "notebookPathURI"
    CONST_NOTEBOOK_ENCODING = "notebookEncoding"
    CONST_OUTPUT_URI = "outputUri"
    CONST_OUTPUT_URI_ALT = "outputURI"
    CONST_EXCLUDE_TAG = "excludeTags"
    CONST_SOURCE = "source"
    CONST_ENTRYPOINT = "entrypoint"

    attribute_map = {
        CONST_NOTEBOOK_PATH: "notebook_path_uri",
        CONST_NOTEBOOK_ENCODING: "notebook_encoding",
        CONST_OUTPUT_URI: "output_uri",
        CONST_EXCLUDE_TAG: "exclude_tags",
        CONST_SOURCE: "source",
        CONST_ENTRYPOINT: "entrypoint",
    }
    attribute_map.update(CondaRuntime.attribute_map)

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        if spec and self.CONST_OUTPUT_URI_ALT in spec:
            val = spec.pop(self.CONST_OUTPUT_URI_ALT)
            spec[self.CONST_OUTPUT_URI] = val
        if self.CONST_OUTPUT_URI_ALT in kwargs:
            val = kwargs.pop(self.CONST_OUTPUT_URI_ALT)
            kwargs[self.CONST_OUTPUT_URI] = val
        super().__init__(spec, **kwargs)

    @property
    def notebook_uri(self) -> str:
        """The URI of the notebook"""
        return self.get_spec(self.CONST_NOTEBOOK_PATH)

    @property
    def notebook_encoding(self) -> str:
        """The encoding of the notebook"""
        return self.get_spec(self.CONST_NOTEBOOK_ENCODING)

    def with_notebook(self, path: str, encoding="utf-8") -> NotebookRuntime:
        """Specifies the notebook to be run as a job.
        Use this method if you would like to run a single notebook.
        Use ``with_source()`` method if you would like to run a notebook with additional dependency files.

        Parameters
        ----------
        path : str
            The path of the Jupyter notebook
        encoding : str
            The encoding for opening the notebook. Defaults to utf-8.

        Returns
        -------
        self
            The runtime instance.
        """
        self.set_spec(self.CONST_NOTEBOOK_ENCODING, encoding)
        return self.set_spec(self.CONST_NOTEBOOK_PATH, path)

    @property
    def exclude_tag(self) -> list:
        """A list of cell tags indicating cells to be excluded from the job"""
        return self.get_spec(self.CONST_EXCLUDE_TAG, [])

    def with_exclude_tag(self, *tags) -> NotebookRuntime:
        """Specifies the cell tags in the notebook to exclude cells from the job script.

        Parameters
        ----------
        *tags : list
            A list of tags (strings).

        Returns
        -------
        self
            The runtime instance.
        """
        exclude_tag_list = []
        for tag in tags:
            if isinstance(tag, list):
                exclude_tag_list.extend(tag)
            else:
                exclude_tag_list.append(tag)
        return self.set_spec(self.CONST_EXCLUDE_TAG, exclude_tag_list)

    @property
    def output_uri(self) -> list:
        """URI for storing the output notebook and files"""
        return self.get_spec(self.CONST_OUTPUT_URI)

    def with_output(self, output_uri: str) -> NotebookRuntime:
        """Specifies the output URI for storing the output notebook and files.
        All files in the directory containing the notebook will be saved.

        Parameters
        ----------
        output_uri : str
            URI for a directory storing the output notebook and files.
            For example, oci://bucket@namespace/path/to/dir

        Returns
        -------
        self
            The runtime instance.
        """
        return self.set_spec(self.CONST_OUTPUT_URI, output_uri)

    def with_source(self, uri: str, notebook: str, encoding="utf-8"):
        """Specify source code directory containing the notebook and dependencies for the job.
        Use this method if you would like to run a notebook with additional dependency files.
        Use the `with_notebook()` method if you would like to run a single notebook.

        In the following example, local folder "path/to/source" contains the notebook and dependencies,
        The local path of the notebook is "path/to/source/relative/path/to/notebook.ipynb"::

            runtime.with_source(uri="path/to/source", notebook="relative/path/to/notebook.ipynb")

        Parameters
        ----------
        uri : str
            URI of the source code directory. This can be local or on OCI object storage.
        notebook : str
            The relative path of the notebook from the source URI.
        encoding : str
            The encoding for opening the notebook. Defaults to utf-8.

        Returns
        -------
        Self
            The runtime instance.

        """
        self.set_spec(self.CONST_SOURCE, uri)
        self.set_spec(self.CONST_ENTRYPOINT, notebook)
        self.set_spec(self.CONST_NOTEBOOK_ENCODING, encoding)
        return self

    @property
    def source(self) -> str:
        """The source code location."""
        return self.get_spec(self.CONST_SOURCE)

    @property
    def notebook(self) -> str:
        """The path of the notebook relative to the source."""
        return self.get_spec(self.CONST_ENTRYPOINT)

    def init(self) -> "NotebookRuntime":
        """Initializes a starter specification for the runtime.

        Returns
        -------
        NotebookRuntime
            The runtime instance.
        """
        super().init()
        return self.with_source(
            uri="{Path to the source code directory. For MLflow, it will be replaced with the path to the project}",
            notebook="{Entrypoint notebook. For MLflow, it will be replaced with the CMD}",
        ).with_exclude_tag("tag1")


class GitPythonRuntime(CondaRuntime, _PythonRuntimeMixin):
    """Represents a job runtime with source code from git repository

    Example::

        runtime = (
            GitPythonRuntime()
            .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
            # Specify the service conda environment by slug name.
            .with_service_conda("pytorch19_p37_gpu_v1")
            # Specify the git repository
            # Optionally, you can specify the branch or commit
            .with_source("https://github.com/pytorch/tutorials.git")
            # Entrypoint is a relative path from the root of the git repo.
            .with_entrypoint("beginner_source/examples_nn/polynomial_nn.py")
            # Copy files in "beginner_source/examples_nn" to object storage after job finishes.
            .with_output(
              output_dir="beginner_source/examples_nn",
              output_uri="oci://bucket_name@namespace/path/to/dir"
            )
        )

    """

    CONST_GIT_URL = "url"
    CONST_BRANCH = "branch"
    CONST_COMMIT = "commit"
    CONST_GIT_SSH_SECRET_ID = "gitSecretId"
    CONST_SKIP_METADATA = "skipMetadataUpdate"
    attribute_map = {
        CONST_GIT_URL: CONST_GIT_URL,
        CONST_BRANCH: CONST_BRANCH,
        CONST_COMMIT: CONST_COMMIT,
        CONST_GIT_SSH_SECRET_ID: "git_secret_id",
        CONST_SKIP_METADATA: "skip_metadata_update",
    }
    attribute_map.update(CondaRuntime.attribute_map)
    attribute_map.update(_PythonRuntimeMixin.attribute_map)

    @property
    def skip_metadata_update(self):
        """Indicate if the metadata update should be skipped after the job run

        By default, the job run metadata will be updated with the following freeform tags:
        * repo: The URL of the Git repository
        * commit: The Git commit ID
        * module: The entry script/module
        * method: The entry function/method
        * outputs. The prefix of the output files in object storage.

        This update step also requires resource principals to have the permission to update the job run.

        Returns
        -------
        bool
            True if the metadata update will be skipped. Otherwise False.
        """
        return self.get_spec(self.CONST_SKIP_METADATA, False)

    def with_source(
        self, url: str, branch: str = None, commit: str = None, secret_ocid: str = None
    ):
        """Specifies the Git repository and branch/commit for the job source code.

        Parameters
        ----------
        url : str
            URL of the Git repository.
        branch : str, optional
            Git branch name, by default None, the default branch will be used.
        commit : str, optional
            Git commit ID (SHA1 hash), by default None, the most recent commit will be used.
        secret_ocid : str
            The secret OCID storing the SSH key content for checking out the Git repository.

        Returns
        -------
        self
            The runtime instance.
        """
        self.set_spec(self.CONST_GIT_URL, url)
        self.set_spec(self.CONST_BRANCH, branch)
        self.set_spec(self.CONST_COMMIT, commit)
        self.set_spec(self.CONST_GIT_SSH_SECRET_ID, secret_ocid)
        return self

    @property
    def url(self) -> str:
        """URL of the Git repository."""
        return self.get_spec(self.CONST_GIT_URL)

    @property
    def branch(self) -> str:
        """Git branch name."""
        return self.get_spec(self.CONST_BRANCH)

    @property
    def commit(self) -> str:
        """Git commit ID (SHA1 hash)"""
        return self.get_spec(self.CONST_COMMIT)

    @property
    def ssh_secret_ocid(self) -> str:
        """The OCID of the OCI Vault secret storing the Git SSH key."""
        return self.get_spec(self.CONST_GIT_SSH_SECRET_ID)

    def init(self) -> "GitPythonRuntime":
        """Initializes a starter specification for the runtime.

        Returns
        -------
        GitPythonRuntime
            The runtime instance.
        """
        super().init()
        return self.with_source(
            "{Git URI. For MLflow, it will be replaced with the Project URI}"
        ).with_entrypoint(
            "{Entrypoint script. For MLflow, it will be replaced with the CMD}"
        )


class DataFlowRuntime(CondaRuntime):
    CONST_SCRIPT_BUCKET = "scriptBucket"
    CONST_ARCHIVE_BUCKET = "archiveBucket"
    CONST_ARCHIVE_URI = "archiveUri"
    CONST_SCRIPT_PATH = "scriptPathURI"
    CONST_CONFIGURATION = "configuration"
    CONST_CONDA_AUTH_TYPE = "condaAuthType"
    CONST_OVERWRITE = "overwrite"
    attribute_map = {
        CONST_SCRIPT_BUCKET: "script_bucket",
        CONST_ARCHIVE_URI: "archive_bucket",
        CONST_ARCHIVE_URI: "archive_uri",
        CONST_SCRIPT_PATH: "script_path_uri",
        CONST_CONFIGURATION: CONST_CONFIGURATION,
        CONST_CONDA_AUTH_TYPE: "conda_auth_type",
        CONST_OVERWRITE: CONST_OVERWRITE,
    }
    attribute_map.update(Runtime.attribute_map)

    def with_conda(self, conda_spec: dict = None):
        if conda_spec.get(self.CONST_CONDA_TYPE) == self.CONST_CONDA_TYPE_SERVICE:
            raise NotImplementedError(
                "Service Packs not supported. Please download and re-upload as a custom pack."
            )
        elif conda_spec.get(self.CONST_CONDA_TYPE) == self.CONST_CONDA_TYPE_CUSTOM:
            return self.with_custom_conda(
                uri=conda_spec.get(self.CONST_CONDA_URI),
                region=conda_spec.get(self.CONST_CONDA_REGION),
            )
        else:
            raise ValueError(
                f"Unknown conda type: {conda_spec.get(self.CONST_CONDA_TYPE)}."
            )

    def with_service_conda(self, slug: str):
        raise NotImplementedError(
            "Publish this conda pack first, and provide the published conda pack uri."
        )

    def with_custom_conda(self, uri: str, region: str = None, auth_type: str = None):
        """Specifies the custom conda pack for running the job

        Parameters
        ----------
        uri : str
            The OCI object storage URI for the conda pack,
            e.g. "oci://your_bucket@namespace/object_name."
            In the Environment Explorer of an OCI notebook session,
            this is shown as the "source" of the conda pack.
        region: str, optional
            The region of the bucket storing the custom conda pack, by default None.
            If region is not specified, ADS will use the region from your authentication credentials,
            * For API Key, config["region"] is used.
            * For Resource Principal, signer.region is used.
            This is required if the conda pack is stored in a different region.
        auth_type: str, (="resource_principal")
            One of "resource_principal", "api_keys", "instance_principal", etc.
            Auth mechanism used to read the conda back uri provided.

        Returns
        -------
        self
            The runtime instance.

        See Also
        --------
        https://docs.oracle.com/en-us/iaas/data-science/using/conda_publishs_object.htm

        """
        if not auth_type:
            auth_type = "resource_principal"
        self.set_spec(self.CONST_CONDA_AUTH_TYPE, auth_type)
        return super().with_custom_conda(uri=uri, region=region)

    def with_archive_uri(self, uri: str) -> "DataFlowRuntime":
        """
        Set archive uri (which is a zip file containing dependencies).

        Parameters
        ----------
        uri: str
            uri to the archive zip

        Returns
        -------
        DataFlowRuntime
            runtime instance itself
        """
        return self.set_spec(self.CONST_ARCHIVE_URI, uri)

    @property
    def archive_uri(self):
        """The Uri of archive zip"""
        return self.get_spec(self.CONST_ARCHIVE_URI)

    @property
    def script_uri(self) -> str:
        """The URI of the source code"""
        return self.get_spec(self.CONST_SCRIPT_PATH)

    def with_script_uri(self, path: str) -> "DataFlowRuntime":
        """
        Set script uri.

        Parameters
        ----------
        path: str
            uri to the script

        Returns
        -------
        DataFlowRuntime
            runtime instance itself
        """
        return self.set_spec(self.CONST_SCRIPT_PATH, path)

    def with_script_bucket(self, bucket) -> "DataFlowRuntime":
        """
        Set object storage bucket to save the script, in case script uri given is local.

        Parameters
        ----------
        bucket: str
            name of the bucket

        Returns
        -------
        DataFlowRuntime
            runtime instance itself
        """
        return self.set_spec(self.CONST_SCRIPT_BUCKET, bucket)

    @property
    def script_bucket(self) -> str:
        """Bucket to save script"""
        return self.get_spec(self.CONST_SCRIPT_BUCKET)

    def with_archive_bucket(self, bucket) -> "DataFlowRuntime":
        """
        Set object storage bucket to save the archive zip, in case archive uri given is local.

        Parameters
        ----------
        bucket: str
            name of the bucket

        Returns
        -------
        DataFlowRuntime
            runtime instance itself
        """
        return self.set_spec(self.CONST_ARCHIVE_BUCKET, bucket)

    @property
    def archive_bucket(self) -> str:
        """Bucket to save archive zip"""
        return self.get_spec(self.CONST_ARCHIVE_BUCKET)

    def with_configuration(self, config: dict) -> "DataFlowRuntime":
        """
        Set Configuration for Spark.

        Parameters
        ----------
        config: dict
            dictionary of configuration details
            https://spark.apache.org/docs/latest/configuration.html#available-properties.
            Example: { “spark.app.name” : “My App Name”, “spark.shuffle.io.maxRetries” : “4” }

        Returns
        -------
        DataFlowRuntime
            runtime instance itself
        """
        return self.set_spec(self.CONST_CONFIGURATION, config)

    @property
    def configuration(self) -> dict:
        """Configuration for Spark"""
        return self.get_spec(self.CONST_CONFIGURATION)

    def with_overwrite(self, overwrite: bool) -> "DataFlowRuntime":
        """
        Whether to overwrite the existing script in object storage (script bucket).
        If the Object Storage bucket already contains a script with the same name,
        then it will be overwritten with the new one if the `overwrite` flag equal to `True`.

        Parameters
        ----------
        overwrite: bool
            Whether to overwrite the existing script in object storage (script bucket).

        Returns
        -------
        DataFlowRuntime
             The DataFlowRuntime instance (self).
        """
        return self.set_spec(self.CONST_OVERWRITE, overwrite)

    @property
    def overwrite(self) -> str:
        """Whether to overwrite the existing script in object storage (script bucket)."""
        return self.get_spec(self.CONST_OVERWRITE)

    def convert(self, **kwargs):
        pass

    def init(self) -> "DataFlowRuntime":
        """Initializes a starter specification for the runtime.

        Returns
        -------
        DataFlowRuntime
            The runtime instance.
        """
        super().init()
        self._spec.pop(self.CONST_ENV_VAR, None)
        return (
            self.with_script_uri(
                "{Path to the executable script. For MLflow, it will be replaced with the CMD}"
            )
            .with_script_bucket(
                "{The object storage bucket to save a script. "
                "Example: oci://<bucket_name>@<tenancy>/<prefix>}"
            )
            .with_overwrite(True)
            .with_configuration({"spark.driverEnv.env_key": "env_value"})
        )


class DataFlowNotebookRuntime(DataFlowRuntime, NotebookRuntime):
    def convert(self, overwrite=False):
        if self.output_uri:
            path = os.path.join(
                self.output_uri,
                str(os.path.basename(self.notebook_uri)).replace(".ipynb", ".py"),
            )
        else:
            path = os.path.splitext(self.notebook_uri)[0] + ".py"
        exclude_tags = self.exclude_tag or {}
        convert_notebook(
            self.notebook_uri, default_signer(), exclude_tags, path, overwrite=overwrite
        )
        self.set_spec(self.CONST_SCRIPT_PATH, path)
