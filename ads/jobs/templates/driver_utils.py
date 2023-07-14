#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import contextlib
import importlib
import json
import logging
import os
import runpy
import shlex
import stat
import subprocess
import sys
import time
import traceback
from io import DEFAULT_BUFFER_SIZE
from typing import List, Optional
from urllib import request
from urllib.parse import urlparse


import oci


CONST_ENV_LOG_LEVEL = "OCI_LOG_LEVEL"
CONST_ENV_WORKING_DIR = "WORKING_DIR"
CONST_ENV_CODE_DIR = "CODE_DIR"
CONST_ENV_JOB_RUN_OCID = "JOB_RUN_OCID"
CONST_ENV_PYTHON_PATH = "PYTHON_PATH"
CONST_ENV_ENTRYPOINT = "CODE_ENTRYPOINT"
CONST_ENV_ENTRY_FUNC = "ENTRY_FUNCTION"
CONST_ENV_OUTPUT_DIR = "OUTPUT_DIR"
CONST_ENV_OUTPUT_URI = "OUTPUT_URI"
CONST_ENV_OCI_RP = "OCI_RESOURCE_PRINCIPAL_VERSION"
CONST_ENV_ADS_IAM = "OCI_IAM_TYPE"
CONST_ENV_INPUT_MAPPINGS = "OCI__INPUT_MAPPINGS"
CONST_ENV_PIP_REQ = "OCI__PIP_REQUIREMENTS"
CONST_ENV_PIP_PKG = "OCI__PIP_PKG"
CONST_API_KEY = "api_key"


DEFAULT_CODE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.environ.get(CONST_ENV_CODE_DIR, "code"),
)


def set_log_level(the_logger: logging.Logger):
    """Sets the log level of a logger based on the environment variable.
    This will also set the log level of logging.lastResort.

    Parameters
    ----------
    the_logger : logging.Logger
        A logger object.
    """
    # Do nothing if env var is not set or is empty/None
    if not os.environ.get(CONST_ENV_LOG_LEVEL):
        return the_logger
    log_level = os.environ.get(CONST_ENV_LOG_LEVEL)
    try:
        the_logger.setLevel(log_level)
        logging.lastResort.setLevel(log_level)
        the_logger.info(f"Log level set to {log_level}")
    except Exception:
        # Catching all exceptions here
        # Setting log level should not interrupt the job run even if there is an exception.
        the_logger.warning("Failed to set log level.")
        the_logger.debug(traceback.format_exc())
    return the_logger


logger = logging.getLogger(__name__)
set_log_level(logger)


class OCIHelper:
    """Contains helper functions to call OCI APIs"""

    @staticmethod
    def init_oci_client(client_class):
        """Initializes OCI client with API key or Resource Principal.

        Parameters
        ----------
        client_class :
            The class of OCI client to be initialized.
        """
        if (
            os.environ.get(CONST_ENV_ADS_IAM, "").lower() == CONST_API_KEY
            or CONST_ENV_OCI_RP not in os.environ
        ):
            logger.info("Initializing %s with API Key...", {client_class.__name__})
            client = client_class(
                oci.config.from_file(
                    file_location=os.environ.get(
                        "OCI_CONFIG_LOCATION", oci.config.DEFAULT_LOCATION
                    ),
                    profile_name=os.environ.get(
                        "OCI_CONFIG_PROFILE", oci.config.DEFAULT_PROFILE
                    ),
                )
            )
        else:
            logger.info(
                "Initializing %s with Resource Principal...", client_class.__name__
            )
            client = client_class(
                {}, signer=oci.auth.signers.get_resource_principals_signer()
            )
        return client

    @staticmethod
    def copy_to_oci_object_storage(
        output_dir: str, namespace: str, bucket: str, prefix: str
    ) -> List[str]:
        """Copies the output files to OCI object storage

        Parameters
        ----------
        output_dir : str
            Path of the output directory containing files to be copied.
        namespace : str
            Namespace of the object storage location.
        bucket : str
            Bucket name of the object storage location.
        prefix : str
            Prefix (path) of the object storage location.

        Returns
        -------
        list
            A list of URIs for files that are copied to output_uri
        """
        client = OCIHelper.init_oci_client(oci.object_storage.ObjectStorageClient)

        if not prefix:
            prefix = ""
        prefix = prefix.strip("/")

        copied = []
        for path, _, files in os.walk(output_dir):
            for name in files:
                file_path = os.path.join(path, name)
                # Get the relative path of the file to keep the directory structure
                relative_path = os.path.relpath(file_path, output_dir)
                if prefix:
                    file_prefix = os.path.join(prefix, relative_path)
                else:
                    # Save file to bucket root if prefix is empty.
                    file_prefix = relative_path

                logger.debug(
                    "Saving %s to %s@%s/%s",
                    relative_path,
                    bucket,
                    namespace,
                    file_prefix,
                )

                with open(file_path, "rb") as pkf:
                    client.put_object(
                        namespace,
                        bucket,
                        file_prefix,
                        pkf,
                    )
                copied.append(f"{bucket}@{namespace}/{file_prefix}")
        return copied

    @staticmethod
    def substitute_output_uri(output_uri):
        """Expand shell variables of form $var and ${var}.
        Unknown variables are left unchanged.
        """
        try:
            return os.path.expandvars(output_uri)
        except Exception:
            logger.warning("Failed to expand output URI with environment variables.")
            logger.debug(traceback.format_exc())
            # Do nothing if there is an error.
            return output_uri

    @staticmethod
    def copy_outputs(
        output_dir: str = os.environ.get("OUTPUT_DIR"),
        output_uri: str = os.environ.get("OUTPUT_URI"),
    ) -> List[str]:
        """Copies the output files to remote URI.

        No file will be copied if either output_dir or output_uri is empty or None.

        Parameters
        ----------
        output_dir : str
            Path of the output directory containing files to be copied.
            By default, os.environ.get("OUTPUT_DIR")
        output_uri : str
            URI of the object storage path to store the output files.
            By default, os.environ.get("OUTPUT_URI")

        Returns
        -------
        list
            A list of URIs for files that are copied to output_uri
        """
        if not output_dir:
            logger.info("OUTPUT_DIR is not defined. No file is copied.")
            return
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        if not os.path.exists(output_dir):
            logger.error("Output directory %s not found.", output_dir)
            return

        if not output_uri:
            logger.info("OUTPUT_URI is not defined. No file is copied.")
            return
        output_uri = OCIHelper.substitute_output_uri(output_uri)

        logger.debug("Copying files in %s to %s...", output_dir, output_uri)
        parsed = urlparse(output_uri)
        # Only OCI object storage is supported at the moment

        bucket = parsed.username
        namespace = parsed.hostname
        if not bucket or not namespace:
            logger.error(
                "Invalid bucket name or namespace in output URI: %s", output_uri
            )
            logger.error(
                "Output URI should have the format of oci://bucket@namespace/path/to/dir"
            )
            return

        prefix = parsed.path
        return OCIHelper.copy_to_oci_object_storage(
            output_dir, namespace, bucket, prefix
        )

    @staticmethod
    def _download_with_urlopen(src, dest):
        url_response = request.urlopen(src)
        with contextlib.closing(url_response) as f:
            logger.debug("Downloading from %s", src)
            with open(dest, "wb") as out_file:
                block_size = DEFAULT_BUFFER_SIZE * 8
                while True:
                    block = f.read(block_size)
                    if not block:
                        break
                    out_file.write(block)

    @staticmethod
    def _download_with_fsspec(src, dest):
        import fsspec

        scheme = urlparse(src).scheme
        fs = fsspec.filesystem(scheme)
        fs.get(
            src,
            dest,
            recursive=True,
            callback=fsspec.callbacks.TqdmCallback(),
        )

    @staticmethod
    def copy_inputs(mappings: dict = None):
        if not mappings and CONST_ENV_INPUT_MAPPINGS in os.environ:
            mappings = json.loads(os.environ[CONST_ENV_INPUT_MAPPINGS])
            logger.debug("Inputs specified from ENV: %s", mappings)

        if not mappings:
            logger.debug("No inputs specified.")
            return

        for src, dest in mappings.items():
            logger.debug("Copying %s to %s", src, dest)
            # Create the dest dir if one does not exist.
            if str(dest).endswith("/"):
                dest_dir = dest
            else:
                dest_dir = os.path.dirname(dest)

            # Do not make dirs if user is using cwd.
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)

            # Use native Python to download http/ftp.
            scheme = urlparse(src).scheme
            if scheme in ["http", "https", "ftp"]:
                OCIHelper._download_with_urlopen(src, dest)
            else:
                OCIHelper._download_with_fsspec(src, dest)
        return


class ArgumentParser:
    """Contains methods for parsing arguments for entry function."""

    def __init__(self, argument_list: list) -> None:
        """Initialize the parser with a list of arguments

        Parameters
        ----------
        argument_list : list
            A list of arguments.
        """
        self.argument_list = argument_list

    def parse(self):
        """Parses the arguments into args and kwargs.
        args is a list of positional arguments.
        kwargs is a dictionary of keyword arguments.
        The "--" will be removed from the keywords.

        Returns
        -------
        (list, dict)
            A tuple of positional arguments (list) and keyword arguments (dict).
        """
        args = []
        kwargs = {}
        key = None
        for arg in self.argument_list:
            if not isinstance(arg, str):
                args.append(arg)
            elif len(arg) > 2 and arg.startswith("--"):
                key = arg[2:]
                kwargs[key] = None
            elif key:
                kwargs[key] = arg
                key = None
            else:
                args.append(arg)

        return args, kwargs


class JobRunner:
    def __init__(self, code_dir: str = DEFAULT_CODE_DIR) -> None:
        """Initialize the job runner

        Parameters
        ----------
        code_dir : str
            The path to the directory containing the user code.
        """
        logger.info("Job Run ID is: %s", os.environ.get(CONST_ENV_JOB_RUN_OCID))
        self.code_dir = code_dir
        self.conda_prefix = sys.executable.split("/bin/python", 1)[0]

    @staticmethod
    def run_command(
        command: str, conda_prefix: str = None, level: Optional[int] = None, check=False
    ) -> int:
        """Runs a shell command and logs the outputs with specific log level.

        Parameters
        ----------
        command : str
            The shell command
        conda_prefix : str, optional
            Prefix of the conda environment for running the command.
            Defaults to None.
        level : int, optional
            Logging level for the command outputs, by default None.
            If this is set to a log level from logging, e.g. logging.DEBUG,
            the command outputs will be logged with the level.
            If this is None, the command outputs will be printed.

        Returns
        -------
        int
            The return code of the command.
        """
        # Add a small time delay so the existing outputs will not intersect with the command outputs.
        time.sleep(0.05)
        logger.info(">>> %s", command)
        if conda_prefix:
            # Conda activate
            # https://docs.conda.io/projects/conda/en/latest/release-notes.html#id241
            cmd = (
                "CONDA_BASE=$(conda info --base) && "
                + "source $CONDA_BASE/etc/profile.d/conda.sh && "
                + f"conda activate {conda_prefix}; "
                + command
            )
        else:
            cmd = command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            shell=True,
        )
        # Stream the outputs
        while True:
            output = process.stdout.readline()
            if process.poll() is not None and output == b"":
                break
            if output:
                msg = output.decode()
                if level is None:
                    # output already contains the line break
                    print(msg, flush=True, end="")
                else:
                    # logging will flush outputs by default
                    # logging will add line break
                    msg = msg.rstrip("\n")
                    logger.log(level=level, msg=msg)
            # Add a small delay so that
            # outputs from the subsequent code will have different timestamp for oci logging
            time.sleep(0.05)
        if check and process.returncode != 0:
            # If there is an error, exit the main process with the same return code.
            sys.exit(process.returncode)
        return process.returncode

    def conda_unpack(self):
        if self.run_command("conda-unpack", level=logging.DEBUG):
            logger.info("conda-unpack exits with non-zero return code.")
        return self

    def install_pip_requirements(self, req_path: str = None):
        if not req_path:
            req_path = os.environ.get(CONST_ENV_PIP_REQ)
        if not req_path:
            return self
        self.run_command(
            f"pip install -r {req_path}", conda_prefix=self.conda_prefix, check=True
        )
        return self

    def install_pip_packages(self, packages: str = None):
        if not packages:
            packages = os.environ.get(CONST_ENV_PIP_PKG)
        if not packages:
            return self
        self.run_command(
            f"pip install {packages}", conda_prefix=self.conda_prefix, check=True
        )
        return self

    def install_dependencies(self):
        return self.install_pip_requirements().install_pip_packages()

    def set_working_dir(self, working_dir: str = os.environ.get(CONST_ENV_WORKING_DIR)):
        """Sets the working directory for the job run.

        Parameters
        ----------
        working_dir : str, optional
            Working directory, by default os.environ.get("WORKING_DIR")
            If working_dir is not set, the working dir will be set to the code dir.
            If working_dir is a relative path, it will be joined with the code dir.

        """
        if working_dir:
            working_dir = os.path.join(self.code_dir, working_dir)
        else:
            working_dir = self.code_dir
        os.chdir(working_dir)
        logger.debug("Working directory set to %s", working_dir)
        # Add working dir to sys.path
        if working_dir not in sys.path:
            sys.path.append(working_dir)
            logger.debug("Added %s to sys.path", working_dir)
        return self

    def setup_python_path(
        self, python_path: str = os.environ.get(CONST_ENV_PYTHON_PATH, "")
    ):
        """Adds additional python paths.
        Relative paths are expanded based on the current working directory.
        This method should be called after setting the working directory (if needed).

        Parameters
        ----------
        python_path : str
            Additional python paths to be added to sys.path,
            by default, os.environ.get("PYTHON_PATH", "")
            Multiple paths can be separated by os.pathsep, which is colon(:) for Linux and Mac.

        """
        if not python_path:
            return self
        path_list = python_path.split(os.pathsep)
        path_list.append(self.code_dir)
        for path in path_list:
            abs_path = os.path.abspath(os.path.expanduser(path))
            if abs_path not in sys.path:
                sys.path.append(abs_path)
                logger.debug("Added %s to sys.path", abs_path)
        logger.debug("Python Path: %s", sys.path)
        return self

    def _run_function(self, module_path: str, entry_function: str, argv: list):
        """Runs the entry function in module specified by module path.

        Parameters
        ----------
        module_path : str
            The path to the module containing the entry function.
        entry_function : str
            The name of the entry function.
        argv : list
            Argument list from command line.
        This list will be parsed into positional arguments and keyword arguments.
        """
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        method = getattr(module, entry_function)

        args, kwargs = ArgumentParser(argv).parse()

        logger.debug(
            "Invoking method: %s with args: %s, kwargs: %s",
            entry_function,
            args,
            kwargs,
        )
        method(*args, **kwargs)

    def _run_script(self, entrypoint: str):
        if (
            os.path.isdir(entrypoint)
            or entrypoint.endswith(".py")
            or entrypoint.endswith(".zip")
        ):
            logger.info("Running script: %s", entrypoint)
            # The file path may refer directly to a Python script
            # or else it may refer to a zipfile or directory containing a top level __main__.py script.
            # See https://docs.python.org/3/library/runpy.html#runpy.run_path
            # Arguments from sys.argv will be passed into the script
            runpy.run_path(entrypoint, run_name="__main__")
        else:
            if os.path.exists(entrypoint):
                # User should make the file executable before committing it to Git
                # e.g. git update-index --chmod=+x my_script.sh
                # Here we make the entrypoint executable just in case
                try:
                    st = os.stat(entrypoint)
                    os.chmod(entrypoint, st.st_mode | stat.S_IEXEC)
                except Exception:
                    # Ignore any error here and continue to try to run the script.
                    # Show the exception for debugging
                    logger.debug(traceback.format_exc())
            # Run the entrypoint as shell command with conda activated
            cmd = shlex.join([entrypoint] + sys.argv[1:])
            return_code = self.run_command(cmd, conda_prefix=self.conda_prefix)
            # Exit the job run with the same return code if it is non-zero.
            if return_code:
                logger.error("CMD exited with return code %s.", return_code)
                sys.exit(return_code)

    def run(
        self,
        entrypoint: str = os.environ.get(CONST_ENV_ENTRYPOINT),
        entry_function: str = os.environ.get(CONST_ENV_ENTRY_FUNC),
    ):
        """Runs the user code

        Parameters
        ----------
        entrypoint : str
            Path to the file serve as the entrypoint,
            by default, os.environ.get("CODE_ENTRYPOINT")

        entry_function : str, optional
            Name of the function in the entrypoint,
            by default, os.environ.get("ENTRY_FUNCTION").
            If this is not set, the entrypoint will be run as a python script.

        """
        if not entrypoint:
            raise ValueError(f"Invalid entrypoint: {str(entrypoint)}")
        entrypoint_abs_path = os.path.abspath(os.path.expanduser(entrypoint))
        if not os.path.exists(entrypoint_abs_path):
            raise ValueError(f"Entrypoint {entrypoint_abs_path} not found.")

        if entry_function:
            logger.info("Running function: %s in %s", entry_function, entrypoint)
            self._run_function(entrypoint, entry_function, sys.argv[1:])
        elif entrypoint.endswith(".ipynb"):
            from driver_notebook import run_notebook

            logger.info("Running notebook: %s", entrypoint)
            # Exclude tags
            tags = os.environ.get("NOTEBOOK_EXCLUDE_TAGS")
            if tags:
                tags = json.loads(tags)
                logger.info("Excluding cells with any of the following tags: %s", tags)
            # Pass in the absolute path to make sure the working dir is notebook directory
            run_notebook(
                os.path.abspath(os.path.expanduser(entrypoint)), exclude_tags=tags
            )
        else:
            self._run_script(entrypoint_abs_path)
        logger.info("Job run completed.")
        return self
