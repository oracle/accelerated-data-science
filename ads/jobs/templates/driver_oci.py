#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
This is a driver script from Oracle ADS to run Python script in OCI Data Science Jobs.
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
import logging
import os
import random
import shutil
import string
import traceback
import uuid
from time import sleep, time
from typing import Optional
from urllib.parse import urlparse
from urllib.request import getproxies

import git
import oci
import requests
from oci.data_science import DataScienceClient

try:
    # This is used in a job run.
    import driver_utils
except ImportError:
    # This is used when importing by other ADS module and testing.
    from . import driver_utils


CONST_ENV_ENTRYPOINT = "GIT_ENTRYPOINT"
CONST_ENV_GIT_URL = "GIT_URL"
CONST_ENV_GIT_BRANCH = "GIT_BRANCH"
CONST_ENV_GIT_COMMIT = "GIT_COMMIT"
CONST_ENV_SECRET_OCID = "GIT_SECRET_OCID"

DEFAULT_CODE_DIR = "~/Code/"
SSH_DIR = os.path.join("/home", "datascience", "ssh_" + str(uuid.uuid4()))
SSH_KEY_FILE_PATH = os.path.join(SSH_DIR, "id_rsa")
SSH_CONFIG_FILE_PATH = os.path.join(SSH_DIR, "config")


logger = logging.getLogger(__name__)
logger = driver_utils.set_log_level(logger)


class GitManager:
    """Contains methods for fetching code from Git repository"""

    def __init__(
        self,
        repo_url: str,
        code_dir: str = os.environ.get(driver_utils.CONST_ENV_CODE_DIR),
    ):
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
        # Use default directory if code_dir is set to None or empty string.
        if not code_dir:
            code_dir = os.path.join(
                DEFAULT_CODE_DIR, os.path.basename(repo_url).split(".", 1)[0]
            )
        self.code_dir = os.path.abspath(os.path.expanduser(code_dir))

        logger.info("Initializing code directory at %s", self.code_dir)
        # Rename the existing directory if one already exists
        if os.path.exists(self.code_dir) and os.listdir(self.code_dir):
            logger.warning(
                "Directory %s already exists and is not empty.", self.code_dir
            )
            new_name = (
                self.code_dir
                + "_"
                + "".join(random.choice(string.ascii_lowercase) for i in range(5))
            )
            shutil.move(
                self.code_dir,
                new_name,
            )
            logger.warning("Renamed %s to %s", self.code_dir, new_name)
        os.makedirs(self.code_dir, exist_ok=True)

        self.repo = None
        self.commit = None

    def _config_ssh_proxy(self, proxy):
        return_code = driver_utils.JobRunner.run_command(
            "command -v socat", level=logging.DEBUG
        )
        if return_code:
            logger.warning(
                "You have ssh_proxy configured. "
                "Please install the 'socat' package into your environment "
                "if you would like to use proxy for Git clone via SSH."
            )
        else:
            ssh_config = f"ProxyCommand socat - PROXY:{proxy.hostname}:%h:%p,proxyport={proxy.port}"
            logger.debug("Adding proxy for SSH: %s", ssh_config)
            with open(SSH_CONFIG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(ssh_config)
            logger.debug("SSH config saved to %s", SSH_CONFIG_FILE_PATH)

    def _config_known_hosts(self, host: str):
        if driver_utils.JobRunner.run_command(
            f"ssh-keyscan -H {host} >> {SSH_DIR}/known_hosts", level=logging.DEBUG
        ):
            logger.debug("Added %s to known hosts.", host)
        else:
            logger.warning(
                "Failed to add %s to known hosts."
                "You may need to configure your subnet security list to allow traffic on port 22.",
                host,
            )
        # Test the connection, for debugging purpose
        if logger.level == logging.DEBUG:
            os.system(f"ssh -vT git@{host}")

    def _config_ssh_key(self):
        if os.path.exists(SSH_KEY_FILE_PATH):
            # Ignore the fingerprint checking
            ssh_cmd = f'ssh -i "{SSH_KEY_FILE_PATH}" '
            if os.path.exists(SSH_CONFIG_FILE_PATH):
                ssh_cmd += f'-F "{SSH_CONFIG_FILE_PATH}" '
            ssh_cmd += (
                '-o "StrictHostKeyChecking=no" -o "UserKnownHostsFile=/dev/null" '
            )
            if logger.level == logging.DEBUG:
                ssh_cmd += "-v "
            logger.debug("SSH command: %s", ssh_cmd)
            return {"GIT_SSH_COMMAND": ssh_cmd}
        return None

    def fetch_repo(self):
        """Clones the Git repository."""
        logger.debug("Cloning from %s", self.repo_url)
        env = None
        if self.repo_url.startswith(("git@", "ssh://")):
            proxies = getproxies()
            if ("http" in proxies or "https" in proxies) and "ssh" not in proxies:
                logger.warning(
                    "You have http/https proxy configured. "
                    "Please set ssh_proxy in environment variable and install the 'socat' package "
                    "if you would like to use the proxy for Git clone via SSH. "
                    "conda install -c conda-forge socat"
                )
            if "ssh" in proxies:
                proxy = urlparse(proxies["ssh"])
                self._config_ssh_proxy(proxy=proxy)
            env = self._config_ssh_key()

        if logger.level == logging.DEBUG:
            os.system("git --version")

        self.repo = git.Repo.clone_from(self.repo_url, self.code_dir, env=env)
        logger.info("Cloned repo to: %s", self.code_dir)
        return self

    def checkout_code(self, branch: Optional[str] = None, commit: Optional[str] = None):
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
            logger.info("Checking out commit: %s", commit)
            self.repo.git.checkout(commit)
        elif branch:
            logger.info("Checking out branch: %s", branch)
            self.repo.git.checkout(branch)
        else:
            logger.info(
                "Checking out the latest commit %s", self.repo.head.commit.hexsha
            )
            self.commit = self.repo.head.commit.hexsha
        return self


class CredentialManager:
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
        secret_client = driver_utils.OCIHelper.init_oci_client(
            oci.secrets.SecretsClient
        )
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
        logger.info("SSH key saved to %s", self.key_file_path)

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


class GitJobRunner(driver_utils.JobRunner):
    """Contains methods for running the job."""

    def __init__(
        self,
        git_manager: GitManager,
        job_run_ocid: str = os.environ.get(driver_utils.CONST_ENV_JOB_RUN_OCID),
    ):
        """Initialize the job runner

        Parameters
        ----------
        job_run_ocid : str, optional
            Job run OCID, by default "".
        For local testing purpose, job run OCID can be set to empty string
        """
        self.job_run_ocid = job_run_ocid
        self.git_manager = git_manager
        self.artifacts = []
        self.source_info = {}
        super().__init__(code_dir=git_manager.code_dir)

    def run(
        self,
        entrypoint: str = os.environ.get(CONST_ENV_ENTRYPOINT),
        entry_function: str = os.environ.get(driver_utils.CONST_ENV_ENTRY_FUNC),
    ):
        """Runs the job

        Parameters
        ----------
        argv : list, optional
            A list of arguments for the entry script/function, by default None

        """
        self.source_info = {
            "repo": self.git_manager.repo_url,
            "commit": self.git_manager.commit,
            "module": entrypoint,
            "method": entry_function if entry_function else "",
        }

        # For git, the entrypoint is relative to the root of the repository
        entrypoint = os.path.join(self.code_dir, entrypoint)

        return super().run(entrypoint=entrypoint, entry_function=entry_function)

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
        logger.debug("Request endpoint: %s", endpoint)
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
        logger.debug(response.json())

    def save_metadata(self) -> None:
        """Saves the metadata to job run"""
        logger.info("Saving metadata to job run...")
        tags = {}
        # Source info
        for key, val in self.source_info.items():
            logger.debug("%s = %s", key, val)
            if val:
                tags[key] = str(val)
        # Output files
        if self.artifacts:
            prefix = "oci://" + os.path.commonprefix(self.artifacts)
            logger.debug("Job output: %s", prefix)
            tags["outputs"] = prefix

        client = driver_utils.OCIHelper.init_oci_client(DataScienceClient)
        oci.retry.DEFAULT_RETRY_STRATEGY.make_retrying_call(
            self.update_job_run_metadata_with_rest_api, client, tags
        )
        logger.debug("Updated Job Run metadata.")


def main():
    """The main function for running the job."""
    second_started = time()

    with GitSSHKey(os.environ.get(CONST_ENV_SECRET_OCID)):
        git_manager = (
            GitManager(os.environ.get(CONST_ENV_GIT_URL))
            .fetch_repo()
            .checkout_code(
                branch=os.environ.get(CONST_ENV_GIT_BRANCH, None),
                commit=os.environ.get(CONST_ENV_GIT_COMMIT, None),
            )
        )

    runner = GitJobRunner(git_manager)
    runner.set_working_dir().setup_python_path().run()

    # Copy outputs
    runner.artifacts = driver_utils.OCIHelper.copy_outputs()

    # Save metadata only if job run OCID is available
    if (
        os.environ.get(driver_utils.CONST_ENV_JOB_RUN_OCID)
        and "SKIP_METADATA_UPDATE" not in os.environ
    ):
        try:
            # Wait before updating the metadata.
            # Job run might still be in the ACCEPTED state shortly after it started,
            # Cannot update job run while in ACCEPTED, CANCELLING, DELETED or NEEDS_ATTENTION state
            second_elapsed = time() - second_started
            if second_elapsed < 90:
                sleep(90 - second_elapsed)
            runner.save_metadata()
        except Exception:
            logger.error("An error occurred when saving the metadata.")
            # Allow the job run to finish successfully even if the driver script failed to save the metadata.
            logger.debug(traceback.format_exc())

    logger.info("Job completed.")


if __name__ == "__main__":
    main()
