#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import contextlib
import logging
import os
import shutil
import tempfile
import zipfile
from io import DEFAULT_BUFFER_SIZE
from urllib import request
from urllib.parse import urlparse

import fsspec
from ads.common.auth import default_signer

logger = logging.getLogger(__name__)


class Artifact:
    """Represents a OCI Data Science Job artifact.
    The Artifact class is designed to add an additional processing step on runtime/source code.
    before uploading it as data science job artifact.

    A sub-class should implement the build() method to do the additional processing.
    A sub-class is designed to be used with context manager so that the temporary files are cleaned up properly.

    For example, the NotebookArtifact implements the build() method to convert the notebook to python script.
    with NotebookArtifact(runtime) as artifact:
        # The build() method will be called when entering the context manager
        # The final artifact for the job will be stored in artifact.path
        upload_artifact(artifact.path)
        # Files are cleaned up when exit or if there is an exception.

    """

    def __init__(self, source, runtime=None) -> None:
        # Get the full path of source file if it is local file.
        if not urlparse(source).scheme:
            self.source = os.path.abspath(os.path.expanduser(source))
        else:
            self.source = source
        self.path = None
        self.temp_dir = None
        self.runtime = runtime

    def __str__(self) -> str:
        if self.path:
            return self.path
        return self.source

    def __enter__(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.build()
        return self

    def __exit__(self, *exc):
        if self.temp_dir:
            self.temp_dir.cleanup()

    @staticmethod
    def _write_file(fp, to_path):
        """Reads a file from a file-like object and write it to specific local path.

        Parameters
        ----------
        fp : file-like object
            The source of the file.
        to_path : path-like object
            Local destination path.
        """
        with open(to_path, "wb") as out_file:
            block_size = DEFAULT_BUFFER_SIZE * 8
            while True:
                block = fp.read(block_size)
                if not block:
                    break
                out_file.write(block)

    @staticmethod
    def _download_from_web(url, to_path):
        """Downloads a single file from http/https/ftp.

        Parameters
        ----------
        url : str
            The URL of the source file.
        to_path : path-like object
            Local destination path.
        """
        url_response = request.urlopen(url)
        with contextlib.closing(url_response) as fp:
            logger.debug("Downloading from %s", url)
            Artifact._write_file(fp, to_path)

    @staticmethod
    def copy_from_uri(uri, to_path, unpack=False, **storage_options):
        """Copy file(s) to local path

        Parameters
        ----------
        uri : URI of the file
            The URI of the source file or directory, which can be local path of OCI object storage URI.
        to_path : path-like object
            The local destination path.
            If this is a directory, the source file/directory will be placed under it.
        unpack : bool
            Indicate if zip or tar.gz file specified by the uri should be unpacked.
            This option has no effect on other files.
        storage_options :
            Storage options for fsspec.
            For OCI object storage, the default_signer from ads.common.auth will be used
            if storage option is not specified.

        Returns
        -------
        str or path-like object
            The actual path of file/directory at destination.
            For copying a single file and to_path is a filename, this will be the same as to_path.
            For copying a single file and to_path is a directory, this will be to_path + filename.
            For copying a directory, this will be to_path + directory name.
        """
        scheme = urlparse(uri).scheme
        # temp_dir is used only if the uri is zip/tar file
        with tempfile.TemporaryDirectory() as temp_dir:
            if unpack and (str(uri).endswith("zip") or str(uri).endswith("tar.gz")):
                unpack_path = to_path
                to_path = temp_dir
            else:
                unpack_path = None
            if scheme in ["http", "https", "ftp"]:
                if os.path.isdir(to_path):
                    to_path = os.path.join(to_path, os.path.basename(uri))
                Artifact._download_from_web(uri, to_path)
            else:
                if scheme == "oci" and not storage_options:
                    storage_options = default_signer()
                fs = fsspec.filesystem(scheme, **storage_options)
                if os.path.isdir(to_path):
                    to_path = os.path.join(
                        to_path, os.path.basename(str(uri).rstrip("/"))
                    )
                fs.get(uri, to_path, recursive=True)
            if unpack_path:
                shutil.unpack_archive(to_path, unpack_path)
                to_path = unpack_path
        return to_path

    def build(self):
        """Builds the runtime artifact in the temporary directory.
        Subclass should implement this method to:
        1. Process the runtime
        2. Set the self.path to the final artifact path

        Raises
        ------
        NotImplementedError
            When this method is not implemented in the subclass.
        """
        raise NotImplementedError()


class NotebookArtifact(Artifact):
    """Represents a NotebookRuntime job artifact"""

    CONST_DRIVER_SCRIPT = "driver_notebook.py"

    def __init__(self, source, runtime) -> None:
        super().__init__(source, runtime=runtime)

    def build(self):
        """Prepares job artifact for notebook runtime"""
        driver_script = os.path.join(
            os.path.dirname(__file__), "../../templates", self.CONST_DRIVER_SCRIPT
        )
        notebook_path = os.path.join(self.temp_dir.name, os.path.basename(self.source))
        output_path = (
            os.path.join(
                self.temp_dir.name,
                str(os.path.basename(self.source)).split(".", maxsplit=1)[0],
            )
            + ".zip"
        )
        self.copy_from_uri(self.source, notebook_path)
        with zipfile.ZipFile(output_path, "w") as zip_file:
            zip_file.write(notebook_path, os.path.basename(notebook_path))
            zip_file.write(driver_script, os.path.basename(driver_script))
        self.path = output_path


class ScriptArtifact(Artifact):
    """Represents a ScriptRuntime job artifact"""

    def build(self):
        """Prepares job artifact for script runtime.
        If the source is a file, it will be returned as is.
        If the source is a directory, it will be compressed as a zip file.
        """
        source = self.copy_from_uri(self.source, self.temp_dir.name)
        # Zip the artifact if it is a directory
        if os.path.isdir(source):
            basename = os.path.basename(str(source).rstrip("/"))
            source = str(source).rstrip("/")
            # Runtime must have entrypoint if the source is a directory
            if self.runtime and not self.runtime.entrypoint:
                raise ValueError(
                    "Please specify entrypoint when script source is a directory."
                )
            output = os.path.join(self.temp_dir.name, basename)
            shutil.make_archive(output, "zip", os.path.dirname(source), base_dir=basename)
            self.path = output + ".zip"
            return
        # Otherwise, use the artifact directly
        self.path = source


class PythonArtifact(Artifact):
    """Represents a PythonRuntime job artifact"""

    CONST_DRIVER_SCRIPT = "driver_python.py"
    # Temp archive dir storing the zip/tar file from the user
    ARCHIVE_DIR = "archive"
    # The directory to store user code
    # This directory must match the USER_CODE_DIR in driver_python.py
    USER_CODE_DIR = "code"

    def build(self):
        """Prepares job artifact for script runtime.
        If the source is a file, it will be returned as is.
        If the source is a directory, it will be compressed as a zip file.
        """
        driver_script = os.path.join(
            os.path.dirname(__file__), "../../templates", self.CONST_DRIVER_SCRIPT
        )
        basename = os.path.basename(str(self.source).rstrip("/")).split(".", 1)[0]
        artifact_dir = os.path.join(self.temp_dir.name, basename)
        code_dir = os.path.join(artifact_dir, self.USER_CODE_DIR)
        os.makedirs(artifact_dir, exist_ok=True)

        # Copy the driver script
        shutil.copy(driver_script, os.path.join(artifact_dir, self.CONST_DRIVER_SCRIPT))

        # Copy user code
        os.makedirs(code_dir, exist_ok=True)
        Artifact.copy_from_uri(self.source, code_dir, unpack=True)

        # Check if entrypoint is valid
        # If the user code is a directory,
        # user should specify the entrypoint with the name of the top level directory.
        # For example, if the user code is in "path/to/dir"
        # The entrypoint should be something like dir/
        if self.runtime and self.runtime.entrypoint:
            if not os.path.exists(
                os.path.join(
                    code_dir, self.runtime.working_dir, self.runtime.entrypoint
                )
            ):
                # The specific entrypoint does not exist.
                # Check if user forgot to specify the top level directory.
                possible_entrypoint = os.path.join(
                    code_dir,
                    self.runtime.working_dir,
                    basename,
                    self.runtime.entrypoint,
                )
                err_message = (
                    f"Invalid entrypoint. {self.runtime.entrypoint} does not exist."
                )
                if os.path.exists(possible_entrypoint):
                    err_message += f" Do you mean {os.path.join(self.runtime.working_dir, basename, self.runtime.entrypoint)}?"
                raise ValueError(err_message)

        # Zip the job artifact
        output = os.path.join(self.temp_dir.name, basename)
        shutil.make_archive(output, "zip", artifact_dir, base_dir="./")
        self.path = output + ".zip"
