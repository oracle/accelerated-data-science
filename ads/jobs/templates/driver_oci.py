#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""This module runs user code from git repository.
The following environment variables are used:
GIT_URL:
    URL to the Git repository.
GIT_BRANCH:
    Optional, the Git branch to checkout.
GIT_COMMIT:
    Optional, the Git commit to checkout. By default, the most recent git commit will be checked out.
CODE_DIR:
    Optional, the directory for saving the user code from Git repository. Defaults to "~/Code"
GIT_ENTRYPOINT:
    Relative path to the entry script/module file in the Git repository.
ENTRY_FUNCTION:
    Optional, function name in the entry script/module to be invoked.
    If this is not specified, the entry script will be run as Python script.
PYTHON_PATH:
    Optional, additional paths to be added to sys.path for looking up modules and packages.
    The root of the Git repository will be added by default.
    Multiple paths can be separated by os.pathsep, which is colon(:) for Linux and Mac, semicolon(;) for Windows.
OUTPUT_DIR:
    Optional, output directory to be copied to object storage.
OUTPUT_URI:
    Optional, object storage URI for saving files from the output directory.
GIT_SECRET_OCID:
    The OCID of the OCI vault secret storing the SSH key for Git commands.
SKIP_METADATA_UPDATE:
    If this variable exists, the update metadata step will be skipped.
OCI_IAM_TYPE:
    Authentication method for OCI services.
    OCI API key will be used if this is set to api_key.
    Otherwise resource principal will be used.
OCI_CONFIG_LOCATION:
    The location of OCI API key when OCI_IAM_TYPE is set to api_key.
    If this is not set, oci.config.DEFAULT_LOCATION will be used.
OCI_CONFIG_PROFILE:
    The profile name to be used for API key authentication.
    If this is not set, oci.config.DEFAULT_PROFILE will be used.
OCI__GIT_SSH_KEY_PATH:
    The location to save the SSH Key for accessing the git repository.

JOB_RUN_OCID:
    The OCID of the job run. This is set by the job run.

This module requires the following packages:
oci
requests
GitPython
git
openssh

"""
import base64
import importlib
import json
import os
import random
import shutil
import string
import subprocess
import sys
import time
import traceback
import uuid
from typing import Optional
from urllib.parse import urlparse
from urllib.request import getproxies

import git
import oci
import requests
from oci.data_science import DataScienceClient
from oci.object_storage import ObjectStorageClient


SSH_DIR = os.path.join("home", "datascience", "ssh_" + str(uuid.uuid4()))
SSH_KEY_FILE_PATH = os.path.join(SSH_DIR, "id_rsa")
DEFAULT_CODE_DIR = "~/Code/"


class GitManager:
    """Contains methods for fetching code from Git repository"""

    def __init__(self, repo_url: str, code_dir: str = None):
        """Initialize the GitManager

        Parameters
        ----------
        repo_url : str
            The URL of the repository.
        code_dir : str
            The local directory for storing the code from Git repository.

        Raises
        ------
        ValueError
            URL is not specified.
        """
        if not repo_url:
            raise ValueError("Specify the URL of Git repository.")
        self.repo_url = repo_url
        if not code_dir:
            code_dir = os.environ.get("CODE_DIR")
        # Use default directory if CODE_DIR is set to None or empty string.
        if not code_dir:
            code_dir = os.path.join(DEFAULT_CODE_DIR, os.path.basename(repo_url).split(".", 1)[0])
        self.code_dir = os.path.abspath(os.path.expanduser(code_dir))

        print(f"Initializing code directory at {self.code_dir} ...")
        # Rename the existing directory if one already exists
        if os.path.exists(self.code_dir) and os.listdir(self.code_dir):
            shutil.move(
                self.code_dir,
                self.code_dir
                + "_"
                + "".join(random.choice(string.ascii_lowercase) for i in range(5)),
            )
        os.makedirs(self.code_dir, exist_ok=True)

        self.repo = None
        self.commit = None

    def fetch_repo(self):
        """Clones the Git repository."""
        print(f"Cloning Repo from {self.repo_url} ...")
        repo_url = self.repo_url
        if repo_url.startswith("git@"):
            repo_url = "ssh://" + repo_url
        host = urlparse(repo_url).hostname
        ssh_config_path = os.path.join(SSH_DIR, "config")
        if urlparse(repo_url).scheme == "ssh":
            os.makedirs(os.path.dirname(ssh_config_path), exist_ok=True)
            proxies = getproxies()
            if ("http" in proxies or "https" in proxies) and "ssh" not in proxies:
                print(
                    "You have http/https proxy configured. "
                    "Please set ssh_proxy in environment variable install the 'socat' package "
                    "if you would like to use the proxy for Git clone via SSH."
                )
            if "ssh" in proxies:
                proxy = urlparse(proxies["ssh"])
                try:
                    subprocess.check_output("command -v socat", shell=True)
                    ssh_config = f"ProxyCommand socat - PROXY:{proxy.hostname}:%h:%p,proxyport={proxy.port}"
                    print(f"Adding proxy for SSH: {ssh_config}")
                    with open(ssh_config_path, "a", encoding="utf-8") as f:
                        f.write(ssh_config)
                    print(f"SSH config saved to {ssh_config_path}")
                except subprocess.CalledProcessError:
                    print(
                        "You have ssh_proxy configured. "
                        "Please install the 'socat' package into your environment "
                        "if you would like to use proxy for Git clone via SSH."
                    )

            try:
                cmd = f"ssh-keyscan -H {host} >> {SSH_DIR}/known_hosts"
                subprocess.check_output(cmd, shell=True)
                print(f"Added {host} to known hosts.")
            except subprocess.CalledProcessError as ex:
                print(f"'{cmd}' exited with code {ex.returncode}.")
                print(ex.output)
                print(
                    "You may need to configure your subnet security list to allow traffic on port 22."
                )

        ssh_key_path = os.path.expanduser(SSH_KEY_FILE_PATH)
        if os.path.exists(ssh_key_path):
            # Test the connection, for debugging purpose
            os.system(f"ssh -vT git@{host}")
            # Ignore the fingerprint checking
            ssh_cmd = f'ssh -i "{ssh_key_path}" '
            if os.path.exists(ssh_config_path):
                ssh_cmd += f'-F "{ssh_config_path}" '
            ssh_cmd += (
                ' -o "StrictHostKeyChecking=no" -o "UserKnownHostsFile=/dev/null"'
            )
            env = {"GIT_SSH_COMMAND": ssh_cmd}
        else:
            env = None
        self.repo = git.Repo.clone_from(self.repo_url, self.code_dir, env=env)
        print(f"Cloned repo to: {self.code_dir}")
        return self

    def setup_code(self, branch: Optional[str] = None, commit: Optional[str] = None):
        """Checkouts the branch or commit of the Git repository.

        If neither branch nor commit is specified, the tip of the default branch will be used.
        If both branch and commit are specified, the commit will be used.

        Parameters
        ----------
        branch : str, optional
            The name of the branch, by default None
        commit : str, optional
            The commit ID (SHA1 hash), by default None

        """
        if commit:
            print(f"Checking out commit: {commit}")
            self.repo.git.checkout(commit)
        elif branch:
            print(f"Checking out branch: {branch}")
            self.repo.git.checkout(branch)
        else:
            print(f"Checking out the latest commit {self.repo.head.commit.hexsha}...")
        self.commit = self.repo.head.commit.hexsha
        return self


class CredentialManager:
    @staticmethod
    def init_oci_client(client_class):
        """Initializes OCI client with API key or Resource Principal.

        Parameters
        ----------
        client_class :
            The class of OCI client to be initialized.
        """
        if os.environ.get("OCI_IAM_TYPE", "").lower() == "api_key":
            print(f"Initializing {client_class.__name__} with API Key...")
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
            print(f"Initializing {client_class.__name__} with Resource Principal...")
            client = client_class(
                {}, signer=oci.auth.signers.get_resource_principals_signer()
            )
        return client

    @staticmethod
    def read_secret(secret_id):
        """Reads and decode the value of of a secret from OCI vault.

        Parameters
        ----------
        secret_id : str
            OCID of the secret

        Returns
        -------
        str
            The value of the secret decoded with ASCII.
        """
        secret_client = CredentialManager.init_oci_client(oci.secrets.SecretsClient)
        secret_bundle = secret_client.get_secret_bundle(secret_id)
        base64_secret_bytes = secret_bundle.data.secret_bundle_content.content.encode(
            "ascii"
        )
        base64_message_bytes = base64.b64decode(base64_secret_bytes)
        secret_content = base64_message_bytes.decode("ascii")
        return secret_content


class GitSSHKey:
    def __init__(self, secret_id) -> None:
        self.secret_id = secret_id
        self.key_file_path = os.path.expanduser(
            os.environ.get("OCI__GIT_SSH_KEY_PATH", SSH_KEY_FILE_PATH)
        )
        self.existing_git_ssh_cmd = None
        self.backup_file_path = None

    def _set_ssh_key(self):
        """Setup SSH key for Git command"""
        content = CredentialManager().read_secret(self.secret_id)
        # Add a new line to the SSH key in case the user forget to do so
        # SSH key without a new line at the end is considered "invalid format".
        if not content.endswith("\n"):
            content += "\n"
        os.makedirs(os.path.dirname(self.key_file_path), exist_ok=True)
        with open(self.key_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        # Set the correct permission for the SSH key.
        os.chmod(self.key_file_path, 0o600)
        print(f"SSH key saved to {self.key_file_path}")

    def _backup_ssh_key(self):
        """Backup the existing SSH key if one exists at the same file location."""
        if not os.path.exists(self.key_file_path):
            return
        suffix = (
            "."
            + "".join(random.choice(string.ascii_lowercase) for i in range(10))
            + ".bak"
        )
        self.backup_file_path = self.key_file_path + suffix
        os.rename(self.key_file_path, self.backup_file_path)

    def _restore_backup(self):
        """Restore the SSH key from backup."""
        if self.backup_file_path and os.path.exists(self.backup_file_path):
            os.rename(self.backup_file_path, self.key_file_path)

    def __enter__(self):
        """Setup the SSH key for Git command"""
        if self.secret_id:
            self._backup_ssh_key()
            self._set_ssh_key()

    def __exit__(self, tp, value, tb):
        """Remove the SSH key and reset the git command option"""
        if not self.key_file_path:
            return
        # Remove the ssh key file only if there is secret
        if self.secret_id:
            try:
                os.remove(self.key_file_path)
                self._restore_backup()
            except OSError:
                # Since this will not affect any job outputs,
                # ignore the error if the file operation failed for some reason.
                pass


class JobRunner:
    """Contains methods for running the job."""

    def __init__(self, job_run_ocid: str = ""):
        """Initialize the job runner

        Parameters
        ----------
        job_run_ocid : str, optional
            Job run OCID, by default "".
        For local testing purpose, job run OCID can be set to empty string
        """
        self.job_run_ocid = job_run_ocid
        self.git_manager = None
        self.artifacts = []
        self.source_info = {}

    @staticmethod
    def setup_python_path(python_path="", code_dir=""):
        """Adds additional python paths."""
        if not python_path:
            python_path = os.environ.get("PYTHON_PATH", "")
        python_paths = python_path.split(os.pathsep)
        python_paths.append(code_dir)
        for path in python_paths:
            python_path = os.path.join(code_dir, path)
            if python_path not in sys.path:
                sys.path.append(python_path)
        print(f"Python Path: {sys.path}")

    @staticmethod
    def run_command(command):
        # Conda activate
        # https://docs.conda.io/projects/conda/en/latest/release-notes.html#id241
        conda_prefix = sys.executable.split("/bin/python", 1)[0]
        cmd = f"CONDA_BASE=$(conda info --base) && source $CONDA_BASE/etc/profile.d/conda.sh && conda activate {conda_prefix}; "
        cmd += command
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=os.environ.copy(), shell=True
        )
        # Steam the outputs
        while True:
            output = process.stdout.readline()
            if process.poll() is not None and output == b"":
                break
            if output:
                print(output.decode().strip(), flush=True)
            time.sleep(0.5)
        return process.returncode

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

        print(f"Invoking method: {entry_function} with args: {args}, kwargs: {kwargs}")
        method(*args, **kwargs)

    def _run_script(self, module_path: str, arguments: Optional[list] = None):
        """Runs the script specified by the user

        Parameters
        ----------
        module_path : str
            The path to the entry script.
        arguments : list, optional
            A list of command line arguments, by default None.
        """
        # Use -u option to run python so that the outputs will not be buffered.
        # This will allow user to see the logs sooner in long running job.
        commands = [sys.executable, "-u", module_path]
        if arguments:
            commands.extend(arguments)
        return_code = self.run_command(" ".join(commands))
        if return_code != 0:
            # If there is an error, exit the main process with the same return code.
            sys.exit(return_code)

    def fetch_code(self):
        """Gets the source code from Git repository."""
        print("Beginning Git Clone...")
        self.git_manager = (
            GitManager(os.environ.get("GIT_URL"))
            .fetch_repo()
            .setup_code(
                branch=os.environ.get("GIT_BRANCH", None),
                commit=os.environ.get("GIT_COMMIT", None),
            )
        )
        print("Code Fetch completed.")
        return self

    def run(self, argv=None):
        """Runs the job

        Parameters
        ----------
        argv : list, optional
            A list of arguments for the entry script/function, by default None

        """
        if not argv:
            argv = []

        entry_function = os.environ.get("ENTRY_FUNCTION")
        entry_script = os.environ.get("GIT_ENTRYPOINT")
        self.source_info = {
            "repo": self.git_manager.repo_url,
            "commit": self.git_manager.commit,
            "module": entry_script,
            "method": entry_function if entry_function else "",
        }

        module_path = os.path.join(self.git_manager.code_dir, entry_script)

        if entry_function:
            print(f"Running function: {entry_function} in {module_path}...")
            self._run_function(module_path, entry_function, argv)
        else:
            print(f"Running script: {module_path}")
            self._run_script(module_path, argv)
        return self

    def copy_artifacts(self, output_dir: str, output_uri: dict) -> None:
        """Copies the output files to object storage bucket.

        Parameters
        ----------
        bucket_info : dict
            Containing information for the object storage bucket.
            The dictionary should have the following keys:
            name: the name of the bucket.
            namespace: the namespace of the bucket.
            prefix: the prefix for saving the files.
            prefix is optional, defaults to Job run ID.

        output_dir : str
            Path of the output directory containing files to be copied.
        """
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        if not os.path.exists(output_dir):
            print(f"Output directory f{output_dir} not found.")
            return
        if not output_uri:
            print("OUTPUT_URI is not defined. No file is copied.")
            return
        print(f"Copying files in {output_dir} to {output_uri}...")
        parsed = urlparse(output_uri)
        bucket_name = parsed.username
        namespace = parsed.hostname
        prefix = parsed.path
        oci_os_client = CredentialManager.init_oci_client(ObjectStorageClient)

        if not prefix:
            prefix = ""
        prefix = prefix.strip("/")

        for path, _, files in os.walk(output_dir):
            for name in files:
                file_path = os.path.join(path, name)

                with open(file_path, "rb") as pkf:
                    # Get the relative path of the file to keep the directory structure
                    relative_path = os.path.relpath(file_path, output_dir)
                    if prefix:
                        file_prefix = os.path.join(prefix, relative_path)
                    else:
                        # Save file to bucket root if prefix is empty.
                        file_prefix = relative_path

                    print(
                        f"Saving {relative_path} to {bucket_name}@{namespace}/{file_prefix}"
                    )

                    oci_os_client.put_object(
                        namespace,
                        bucket_name,
                        file_prefix,
                        pkf,
                    )
                    self.artifacts.append(f"{bucket_name}@{namespace}/{file_prefix}")

    @staticmethod
    def _raise_for_error(response: requests.Response):
        if response.status_code >= 400:
            try:
                message = response.json().get("message", "")
            except:
                message = response.content
            error = f"{response.status_code} {response.reason} Error for {response.url}\n{message}"
            raise requests.exceptions.HTTPError(error, response=response)

    def update_job_run_metadata_with_rest_api(
        self, client: DataScienceClient, metadata: dict
    ) -> None:
        """Updates the metadata of the job run by call OCI REST API.

        Parameters
        ----------
        client : DataScienceClient
            OCI DataScienceClient
        metadata : dict
            Metadata to be saved as freeform tags.
        """
        endpoint = f"{client.base_client.endpoint}/jobRuns/{self.job_run_ocid}"
        print(f"Request endpoint: {endpoint}")
        headers = {"accept": "application/json", "content-type": "application/json"}
        signer = client.base_client.signer
        # Get the existing tags
        response = requests.get(endpoint, headers=headers, auth=signer)
        self._raise_for_error(response)

        tags = response.json().get("freeformTags", {})
        tags.update(metadata)
        body = {
            "definedTags": None,
            "displayName": None,
            "freeformTags": tags,
        }
        # Additional Headers for PUT request
        headers.update(
            {
                "expect": "100-continue",
                "opc-request-id": client.base_client.build_request_id(),
            }
        )
        response = requests.put(endpoint, headers=headers, json=body, auth=signer)
        self._raise_for_error(response)
        # print(response.json())

    def save_metadata(self) -> None:
        """Saves the metadata to job run"""
        print("Saving Metdata to job run...")
        tags = {}
        # Source info
        for key, val in self.source_info.items():
            print(f"{key} = {val}")
            if val:
                tags[key] = str(val)
        # Output files
        if self.artifacts:
            prefix = "oci://" + os.path.commonprefix(self.artifacts)
            print(f"Job Output: {prefix}")
            tags["outputs"] = prefix

        client = CredentialManager.init_oci_client(DataScienceClient)
        oci.retry.DEFAULT_RETRY_STRATEGY.make_retrying_call(
            self.update_job_run_metadata_with_rest_api, client, tags
        )
        print("Updated Job Run metadata.")


def check_internet():
    """Checks the internet connection by sending GET request to oracle.com"""
    print("Checking internet connection...")
    try:
        response = requests.get("https://oracle.com", timeout=3)
        print(f"Request Status Code: {response.status_code}")
        response.raise_for_status()
    except Exception as ex:
        print(str(ex))
        print("Internet is not available!")


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

    @staticmethod
    def decode_arg(val):
        """Decodes the value of the argument if it is a JSON payload.

        Parameters
        ----------
        val : str
            The argument value in a string.

        Returns
        -------
        Any
            None, if the val is None.
            String value, if the val is a string but not a JSON payload.
            Otherwise, the object after JSON decoded.
        """
        if val is None:
            return None
        try:
            return json.loads(val)
        except json.decoder.JSONDecodeError:
            return val

    @staticmethod
    def join_values(value_list):
        """Joins the values of a keyword argument.

        Parameters
        ----------
        value_list : list
            Values in a list of strings.

        Returns
        -------
        str or None
            The value of the argument as a string.
        """
        if value_list:
            return " ".join(value_list)
        return None

    def parse(self):
        """Parses the arguments

        Returns
        -------
        (list, dict)
            A tuple of positional arguments (list) and keyword arguments (dict).
        """
        args = []
        kwargs = {}
        parsing_kwargs = False
        key = None
        val = []
        for arg in self.argument_list:
            arg = str(arg)
            if len(arg) > 2 and arg.startswith("--"):
                if key:
                    # Save previous key and val
                    kwargs[key] = self.join_values(val)
                parsing_kwargs = True
                key = arg[2:]
                # Reset val
                val = []
            elif parsing_kwargs:
                val.append(arg)
            else:
                args.append(arg)
        # Save the last key and val
        if key:
            kwargs[key] = self.join_values(val)

        args = [self.decode_arg(arg) for arg in args]
        kwargs = {k: self.decode_arg(v) for k, v in kwargs.items()}
        return args, kwargs


def main():
    """The main function for running the job."""
    job_run_ocid = os.environ.get("JOB_RUN_OCID")
    print(f"Job Run ID is: {job_run_ocid}")

    check_internet()
    runner = JobRunner(job_run_ocid)
    with GitSSHKey(os.environ.get("GIT_SECRET_OCID")):
        runner.fetch_code().setup_python_path(code_dir=runner.git_manager.code_dir)
        runner.run(sys.argv[1:])

    # Copy outputs
    if "OUTPUT_DIR" in os.environ:
        print(f"Found OUTPUT_DIR is configured as: {os.environ['OUTPUT_DIR']}")
        runner.copy_artifacts(
            output_dir=os.environ["OUTPUT_DIR"],
            output_uri=os.environ.get("OUTPUT_URI"),
        )
    else:
        print("OUTPUT_DIR is not configured. Skipping copy artifacts")

    # Save metadata only if job run OCID is available
    if job_run_ocid and "SKIP_METADATA_UPDATE" not in os.environ:
        try:
            runner.save_metadata()
        except:
            # Allow the job run to finish successfully even if the driver script failed to save the metadata.
            traceback.print_exc()

    print("Job completed.")


if __name__ == "__main__":
    main()
