#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import glob
import importlib
import inspect
import os
import re
import shutil
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib.parse import urlparse

from ads.common import auth as authutil
from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.common.utils import copy_from_uri
from ads.opctl import logger
from ads.opctl.constants import OPERATOR_MODULE_PATH
from ads.opctl.operator import __operators__

from .const import ARCH_TYPE, PACK_TYPE
from .errors import OperatorNotFoundError

LOCAL_SCHEME = "local"
MAIN_BRANCH = "main"


@dataclass
class OperatorInfo:
    """Class representing brief information about the operator.

    Attributes
    ----------
    name (str)
        The name of the operator.
    gpu (bool)
        Whether the operator supports GPU.
    short_description (str)
        A short description of the operator.
    description (str)
        A detailed description of the operator.
    version (str)
        The version of the operator.
    conda (str)
        The conda environment required to run the operator.
    conda_type (str)
        The type of conda pack (e.g., PACK_TYPE.CUSTOM).
    path (str)
        The location of the operator.
    keywords (List[str])
        Keywords associated with the operator.
    backends (List[str])
        List of supported backends.

    Properties
    ----------
    conda_prefix (str)
        Generates the conda prefix for the custom conda pack.
    """

    name: str
    gpu: bool
    short_description: str
    description: str
    version: str
    conda: str
    conda_type: str
    path: str
    keywords: List[str]
    backends: List[str]

    @property
    def conda_prefix(self) -> str:
        """
        Generates conda prefix for the custom conda pack.

        Example
        -------
        conda = "forecast_v1"
        conda_prefix == "cpu/forecast/1/forecast_v1"

        Returns
        -------
        str
            The conda prefix for the custom conda pack.
        """
        return os.path.join(
            f"{ARCH_TYPE.GPU if self.gpu else ARCH_TYPE.CPU}",
            self.name,
            re.sub("[^0-9.]", "", self.version),
            f"{self.name}_{self.version}",
        )

    @classmethod
    def from_init(cls, **kwargs: Dict) -> "OperatorInfo":
        """
        Instantiates the class from the initial operator details config.

        Parameters
        ----------
        **kwargs (Dict)
            Keyword arguments containing operator details.

        Returns
        -------
        OperatorInfo
            An instance of OperatorInfo.
        """
        path = kwargs.get("__operator_path__")
        operator_readme = None
        if path:
            readme_file_path = os.path.join(path, "readme.md")
            if os.path.exists(readme_file_path):
                with open(readme_file_path, "r") as readme_file:
                    operator_readme = readme_file.read()

        return OperatorInfo(
            name=kwargs.get("__type__"),
            gpu=kwargs.get("__gpu__", "").lower() == "yes",
            description=operator_readme or kwargs.get("__short_description__"),
            short_description=kwargs.get("__short_description__"),
            version=kwargs.get("__version__"),
            conda=kwargs.get("__conda__"),
            conda_type=kwargs.get("__conda_type__", PACK_TYPE.CUSTOM),
            path=path,
            keywords=kwargs.get("__keywords__", []),
            backends=kwargs.get("__backends__", []),
        )


class Loader(ABC):
    """Operator Loader Interface.

    Attributes
    ----------
    uri (str)
        The operator's location (e.g., local path, HTTP path, OCI path, GIT path).
    uri_dst (str)
        The local folder where the operator can be downloaded from the remote location.
        A temporary folder will be generated if not provided.
    auth (Dict, optional)
        Default authentication settings.

    Methods
    -------
    load (**kwargs)
        Downloads the operator's source code to the local folder.
    cleanup (**kwargs)
        Cleans up all temporary files and folders created during operator loading.
    """

    def __init__(self, uri: str, uri_dst: str = None, auth: Dict = None) -> None:
        """
        Instantiates Loader.

        Parameters
        ----------
        uri (str)
            The operator's location.
        uri_dst (str)
            The local folder where the operator can be downloaded from the remote location.
            A temporary folder will be generated if not provided.
        auth (Dict, optional)
            Default authentication settings.
        """
        self.uri = uri
        self.uri_dst = uri_dst
        self.auth = auth

    @abstractmethod
    def _load(self, **kwargs: Dict) -> OperatorInfo:
        """
        Downloads the operator's source code to the local folder.
        This method needs to be implemented on the child level.

        Parameters
        ------------
        **kwargs (Dict)
            Additional optional attributes.

        Returns
        -------
        OperatorInfo
            Information about the operator.
        """
        pass

    def load(self, **kwargs: Dict) -> OperatorInfo:
        """
        Downloads the operator's source code to the local folder.

        Parameters
        ------------
        **kwargs (Dict)
            Additional optional attributes.

        Returns
        -------
        OperatorInfo
            Information about the operator.
        """
        operator_info = self._load(**kwargs)
        # Adds the operators path to the system path.
        # This will allow to execute the operator via runpy.run_module()
        sys.path.insert(0, "/".join(operator_info.path.split("/")[0:-1]))
        return operator_info

    def cleanup(self, **kwargs: Dict) -> None:
        """
        Cleans up all temporary files and folders created during the loading of the operator.

        Parameters
        ------------
        **kwargs (Dict)
            Additional optional attributes.
        """
        pass

    @classmethod
    @abstractmethod
    def compatible(cls, uri: str, **kwargs: Dict) -> bool:
        """
        Checks if the loader is compatible with the given URI.

        Parameters
        ------------
        uri (str)
            The operator's location.
        **kwargs (Dict)
            Additional optional attributes.
        Returns
        -------
        bool
            Whether the loader is compatible with the given URI.
        """
        pass


class OperatorLoader:
    """
    The operator loader class.
    Helps to download the operator's source code to the local folder.

    Attributes
    ----------
    loader (Loader)
        The specific operator's loader.
    """

    def __init__(self, loader: Loader):
        """
        Initializes OperatorLoader.

        Parameters
        ----------
        loader (Loader)
            The particular operator loader.
        """
        self.loader = loader

    def load(self, **kwargs: Dict) -> OperatorInfo:
        """
        Downloads the operator's source code to the local folder.

        Parameters
        ------------
        **kwargs (Dict)
            Additional optional attributes.

        Returns
        -------
        OperatorInfo
            Detailed information about the operator.
        """
        return self.loader.load(**kwargs)

    @classmethod
    def from_uri(
        cls, uri: str, uri_dst: str = None, auth: Dict = None
    ) -> "OperatorLoader":
        """
        Constructs the operator's loader instance.

        Parameters
        ----------
        uri (str)
            The operator's location.
        uri_dst (str)
            The local folder where the operator can be downloaded from the remote location.
            A temporary folder will be generated if not provided.
        auth (Dict, optional)
            Default authentication settings.

        Returns
        -------
        OperatorLoader
            An instance of OperatorLoader.
        """
        if not uri:
            raise ValueError("The `uri` attribute must be provided.")

        uri = os.path.expanduser(uri)

        auth = auth or authutil.default_signer()

        for loader in (
            ServiceOperatorLoader,
            LocalOperatorLoader,
            GitOperatorLoader,
            RemoteOperatorLoader,
        ):
            if loader.compatible(uri=uri, auth=auth):
                return cls(loader=loader(uri=uri, uri_dst=uri_dst, auth=auth))

        raise ValueError(f"The operator cannot be loaded from the given source: {uri}.")


class ServiceOperatorLoader(Loader):
    """
    Class to load a service operator.

    Attributes
    ----------
    uri (str)
        The operator's location (e.g., local path, HTTP path, OCI path, GIT path).
    uri_dst (str)
        The local folder where the operator can be downloaded from the remote location.
        A temporary folder will be generated if not provided.
    auth (Dict, optional)
        Default authentication settings.
    """

    def _load(self, **kwargs: Dict) -> OperatorInfo:
        """
        Loads the service operator info.

        Parameters
        ----------
        **kwargs (Dict)
            Additional optional attributes.

        Returns
        -------
        OperatorInfo
            Detailed information about the operator.
        """
        return _operator_info(name=self.uri)

    @classmethod
    def compatible(cls, uri: str, **kwargs: Dict) -> bool:
        """
        Checks if the loader is compatible with the given URI.

        Parameters
        ----------
        uri (str)
            The operator's location.
        **kwargs (Dict)
            Additional optional attributes.

        Returns
        -------
        bool
            Whether the loader is compatible with the given URI.
        """
        return uri.lower() in __operators__


class LocalOperatorLoader(Loader):
    """
    Class to load a local operator.

    Attributes
    ----------
    uri (str)
        The operator's location (e.g., local path, HTTP path, OCI path, GIT path).
    uri_dst (str)
        The local folder where the operator can be downloaded from the remote location.
        A temporary folder will be generated if not provided.
    auth (Dict, optional)
        Default authentication settings.
    """

    def _load(self, **kwargs: Dict) -> OperatorInfo:
        """
        Loads the local operator info.

        Parameters
        ----------
        **kwargs (Dict)
            Additional optional attributes.

        Returns
        -------
        OperatorInfo
            Detailed information about the operator.
        """
        return _operator_info(path=self.uri)

    @classmethod
    def compatible(cls, uri: str, **kwargs: Dict) -> bool:
        """Checks if the loader is compatible with the given URI.

        Parameters
        ----------
        uri (str)
            The operator's location.
        **kwargs (Dict)
            Additional optional attributes.

        Returns
        -------
        bool
            Whether the loader is compatible with the given URI.
        """
        return not urlparse(uri).scheme


class RemoteOperatorLoader(Loader):
    """
    Class to load an operator from a remote location (OCI Object Storage).

    Attributes
    ----------
    uri (str)
        The operator's location (e.g., local path, HTTP path, OCI path, GIT path).
    uri_dst (str)
        The local folder where the operator can be downloaded from the remote location.
        A temporary folder will be generated if not provided.
    auth (Dict, optional)
        Default authentication settings.
    """

    def _load(self, **kwargs: Dict) -> OperatorInfo:
        """Downloads the operator's source code to the local folder.

        Parameters
        ----------
        **kwargs (Dict)
            Additional optional attributes.

        Returns
        -------
        OperatorInfo
            Detailed information about the operator.
        """
        self.tmp_dir = tempfile.mkdtemp() if not self.uri_dst else None
        uri_dst = os.path.join(
            (self.uri_dst or self.tmp_dir).rstrip(),
            os.path.splitext(os.path.basename(self.uri.rstrip()))[0],
        )

        logger.info(f"Downloading operator from `{self.uri}` to `{uri_dst}`.")
        copy_from_uri(
            self.uri, uri_dst, force_overwrite=True, auth=self.auth, unpack=True
        )

        return _operator_info(path=uri_dst)

    def cleanup(self, **kwargs: Dict) -> None:
        """Cleans up all temporary files and folders created during operator loading.

        Parameters
        ----------
        **kwargs (Dict)
            Additional optional attributes.
        """
        super().cleanup(**kwargs)
        try:
            shutil.rmtree(self.tmp_dir)
        except Exception as ex:
            logger.debug(ex)

    @classmethod
    def compatible(cls, uri: str, **kwargs: Dict) -> bool:
        """Checks if the loader is compatible with the given URI.

        Parameters
        ----------
        uri (str)
            The operator's location.
        **kwargs (Dict)
            Additional optional attributes.
        Returns
        -------
        bool
            Whether the loader is compatible with the given URI.
        """
        return urlparse(uri).scheme.lower() == "oci"


class GitOperatorLoader(Loader):
    """
    Class to load an operator from a GIT repository.
    Supported URI format: https://github.com/<repository>@<branch-nane>#<path/to/the/operator>
    Examples:
        - https://github.com/my-operator-repository.git@feature-branch#forecasting
        - https://github.com/my-operator-repository#forecasting
        - https://github.com/my-operator-repository

    Attributes
    ----------
    uri (str)
        The operator's location (e.g., local path, HTTP path, OCI path, GIT path).
    uri_dst (str)
        The local folder where the operator can be downloaded from the remote location.
        A temporary folder will be generated if not provided.
    auth (Dict, optional)
        Default authentication settings.
    """

    @runtime_dependency(
        module="git",
        err_msg=(
            "The `git` library is required. "
            "Use `pip install git` to install the `git` library."
        ),
    )
    def _load(self, **kwargs: Dict) -> OperatorInfo:
        """
        Downloads the operator's source code to the local folder.

        Parameters
        ----------
        **kwargs (Dict)
            Additional optional attributes.

        Returns
        -------
        OperatorInfo
            Detailed information about the operator.
        """
        import git

        self.tmp_dir = tempfile.mkdtemp() if not self.uri_dst else None
        uri_dst = self.uri_dst or self.tmp_dir

        uri_dst = os.path.join(
            (self.uri_dst or self.tmp_dir).rstrip(),
            os.path.splitext(os.path.basename(self.uri.rstrip()))[0],
        )

        logger.info(f"Fetching operator from `{self.uri}` to `{uri_dst}`.")

        # Parse the GitHub URL
        parsed_url = urlparse(self.uri)
        logger.debug(parsed_url)

        branch = "main"  # Default branch
        repo_name = parsed_url.path

        if "@" in parsed_url.path:
            # Extract the branch if provided in the URL
            branch = parsed_url.path.split("@")[1]
            repo_name = parsed_url.path.split("@")[0]

        # Construct the repository URL
        repo_url = f"https://{parsed_url.netloc}{repo_name}"
        logger.debug(repo_url)

        # Clone the GitHub repository to a temporary directory
        with tempfile.TemporaryDirectory() as tmp_git_dir:
            repo = git.Repo.clone_from(repo_url, tmp_git_dir, branch=branch)

            # Find the folder to download
            if parsed_url.fragment:
                folder_to_download = parsed_url.fragment
                folder_path = os.path.join(tmp_git_dir, folder_to_download)

                if not os.path.exists(folder_path):
                    raise ValueError(
                        f"Folder '{folder_to_download}' not found in the repository."
                    )

                # Move the folder to the desired local path
                for item in glob.glob(os.path.join(folder_path, "**"), recursive=True):
                    destination_item = os.path.join(
                        uri_dst, os.path.relpath(item, folder_path)
                    )
                    if os.path.isdir(item):
                        # If it's a directory, create it in the destination directory
                        if not os.path.exists(destination_item):
                            os.makedirs(destination_item)
                    else:
                        # If it's a file, move it to the destination directory
                        shutil.move(item, destination_item)

            # Clean up the temporary directory
            repo.close()

    def cleanup(self, **kwargs: Dict) -> None:
        """Cleans up all temporary files and folders created during operator loading.

        Parameters
        ----------
        **kwargs (Dict)
            Additional optional attributes.
        """
        super().cleanup(**kwargs)
        try:
            shutil.rmtree(self.tmp_dir)
        except Exception as ex:
            logger.debug(ex)

    @classmethod
    def compatible(cls, uri: str, **kwargs: Dict) -> bool:
        """Checks if the loader is compatible with the given URI.

        Parameters
        ----------
        uri (str)
            The operator's location.
        **kwargs (Dict)
            Additional optional attributes.

        Returns
        -------
        bool
            Whether the loader is compatible with the given URI.
        """
        return any(element in uri.lower() for element in ("github", ".git"))


def _module_from_file(module_name: str, module_path: str) -> Any:
    """
    Loads module by it's location.

    Parameters
    ----------
    module_name (str)
        The name of the module to be imported.
    module_path (str)
        The physical path of the module.

    Returns
    -------
    Loaded module.
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_constant_values(module_name: str, module_path: str) -> Dict[str, Any]:
    """Returns the list of constant variables from a given module.

    Parameters
    ----------
    module_name (str)
        The name of the module to be imported.
    module_path (str)
        The physical path of the module.

    Returns
    -------
    Dict[str, Any]
        Map of variable names and their values.
    """
    module = _module_from_file(module_name, module_path)
    return {name: value for name, value in vars(module).items()}


def _operator_info(path: str = None, name: str = None) -> OperatorInfo:
    """
    Extracts operator's details by given path.
    The expectation is that the operator has an init file where all details are placed.

    Parameters
    ------------
    path (str, optional)
        The path to the operator.
    name (str, optional)
        The name of the service operator.

    Returns
    -------
    OperatorInfo
        The operator details.
    """
    try:
        if name:
            path = os.path.dirname(
                inspect.getfile(
                    importlib.import_module(f"{OPERATOR_MODULE_PATH}.{name}")
                )
            )

        module_name = os.path.basename(path.rstrip("/"))
        module_path = f"{path.rstrip('/')}/__init__.py"
        return OperatorInfo.from_init(
            **_module_constant_values(module_name, module_path)
        )
    except (ModuleNotFoundError, FileNotFoundError) as ex:
        logger.debug(ex)
        raise OperatorNotFoundError(name or path)


def _operator_info_list() -> List[OperatorInfo]:
    """Returns the list of registered operators.

    Returns
    -------
    List[OperatorInfo]
        The list of registered operators.
    """
    return (_operator_info(name=operator_name) for operator_name in __operators__)
