#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Utilities used by the model deployment package
"""

# Standard lib

import json
import logging
import os
import shutil
import tempfile
import time
import requests
from typing import Dict
from enum import Enum, auto

import oci
import fsspec
from oci.data_science.models import CreateModelDetails
from ads.common import auth, oci_client
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.config import COMPARTMENT_OCID, PROJECT_OCID


logger = logging.getLogger(__name__)
DEFAULT_CONTENT_TYPE_JSON = "application/json"
DEFAULT_CONTENT_TYPE_BYTES = "application/octet-stream"


def get_logger():
    return logger


def set_log_level(level="INFO"):
    """set_log_level sets the logger level

    Args:
        level (str, optional): The logger level. Defaults to "INFO"

    Returns:
        Nothing
    """

    level = logging.getLevelName(level)
    logger.setLevel(level)


def seconds_since(t):
    """seconds_since returns the seconds since `t`. `t` is assumed to be a time
    in epoch seconds since time.time() returns the current time in epoch seconds.

    Args:
        t (int) - a time in epoch seconds

    Returns
        int: the number of seconds since `t`
    """

    return time.time() - t


@runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
def is_notebook():
    """is_notebook returns True if the environment is a Jupyter notebook and
    False otherwise

    Args:
        None

    Returns:
        bool: True if Jupyter notebook; False otherwise

    Raises:
        NameError: If retrieving the shell name from get_ipython() throws an error

    """

    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # pragma: no cover
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except ImportError:
        # IPython is not installed
        return False
    except NameError:
        return False  # Probably standard Python interpreter


def send_request(
    data,
    endpoint: str,
    dry_run: bool = False,
    is_json_payload: bool = False,
    header: dict = {},
    **kwargs,
):
    """Sends the data to the predict endpoint.

    Args:
        data (bytes or Json serializable): data need to be sent to the endpoint.
        endpoint (str): The model HTTP endpoint.
        dry_run (bool, optional): Defaults to False.
        is_json_payload (bool, optional): Indicate whether to send data with a `application/json` MIME TYPE. Defaults to False.
        header (dict, optional): A dictionary of HTTP headers to send to the specified url. Defaults to {}.

    Returns:
        A JSON representive of a requests.Response object.
    """
    if is_json_payload:
        header["Content-Type"] =  header.pop("content_type", DEFAULT_CONTENT_TYPE_JSON) or DEFAULT_CONTENT_TYPE_JSON
        request_kwargs = {"json": data}
    else:
        header["Content-Type"] = header.pop("content_type", DEFAULT_CONTENT_TYPE_BYTES) or DEFAULT_CONTENT_TYPE_BYTES
        request_kwargs = {"data": data}  # should pass bytes when using data
    
    request_kwargs["headers"] = header

    if dry_run:
        request_kwargs["headers"]["Accept"] = "*/*"
        req = requests.Request("POST", endpoint, **request_kwargs).prepare()
        if is_json_payload:
            return json.loads(req.body)
        return req.body
    else:
        request_kwargs["auth"] = header.pop("signer")
        return requests.post(endpoint, **request_kwargs).json()


# State Constants
class State(Enum):
    ACTIVE = auto()
    CREATING = auto()
    DELETED = auto()
    DELETING = auto()
    FAILED = auto()
    INACTIVE = auto()
    UPDATING = auto()
    UNKNOWN = auto()

    @staticmethod
    def _from_str(state):
        if state == None:
            return State.UNKNOWN
        elif state.upper() == "ACTIVE":
            return State.ACTIVE
        elif state.upper() == "CREATING":
            return State.CREATING
        elif state.upper() == "DELETED":
            return State.DELETED
        elif state.upper() == "DELETING":
            return State.DELETING
        elif state.upper() == "FAILED":
            return State.FAILED
        elif state.upper() == "INACTIVE":
            return State.INACTIVE
        elif state.upper() == "UPDATING":
            return State.UPDATING
        else:
            return State.UNKNOWN

    def __call__(self):
        # This will provide backward compatibility.
        # In previous release, ModelDeployment has state() as method instead of property
        return self


class OCIClientManager:
    """OCIClientManager is a helper class used for accessing DataScienceClient and
    DataScienceCompositeClient objects

    Attributes
    ----------
    ds_client - class attribute for data science client
    ds_composite_client - class attribute for data science composite client

    """

    def __init__(self, config=None) -> None:
        if not config:
            config = auth.default_signer()
        self.config = config
        self.ds_client = oci_client.OCIClientFactory(**config).data_science
        self.ds_composite_client = (
            oci.data_science.DataScienceClientCompositeOperations(self.ds_client)
        )

    def default_compartment_id(self):
        """Determines the default compartment OCID
        This method finds the compartment OCID from (in priority order):
        an environment variable, an API key config or a resource principal signer.

        Parameters
        ----------
        config : dict, optional
            The model deployment config, which contains the following keys:
            auth: Authentication method, must be either "resource_principal" or "api_key".
            If auth is not specified:
                1. api_key will be used if available.
                2. If api_key is not available, resource_principal will be used.
            oci_config_file: OCI API key config file location. Defaults to "~/.oci/config"
            oci_config_profile: OCI API key config profile name. Defaults to "DEFAULT"

        Returns
        -------
        str or None
            The compartment OCID if found. Otherwise None.
        """
        # Try to get compartment ID from environment variable.')
        if os.environ.get("NB_SESSION_COMPARTMENT_OCID"):
            return os.environ.get("NB_SESSION_COMPARTMENT_OCID")
        # Try to get compartment ID from OCI config, then RP signer
        # Note: we assume compartment_ids can never be: 0, False, etc.
        oci_config = self.config.get("config")
        signer = self.config.get("signer")
        return (
            oci_config.get("compartment_id")
            or oci_config.get("tenancy")
            or getattr(signer, "tenancy_id", None)
        )

    def prepare_artifact(self, model_uri: str, properties: Dict) -> str:
        """
        Prepare model artifact. Returns model ocid.

        Args:
            model_uri (str): uri to model files, can be local or in cloud storage
            properties (dict): dictionary of properties that are needed for creating a model.
            ds_client (DataScienceClient): OCI DataScienceClient

        Returns:
            str: model ocid
        """
        properties_dict = {}
        if properties:
            properties_dict = (
                properties
                if isinstance(properties, dict)
                else json.loads(repr(properties))
            )
        with tempfile.TemporaryDirectory() as d:
            fhandlers = fsspec.open_files(
                model_uri,
                config=self.config.get("config", {}),
                mode="rb",
            )
            if len(fhandlers) == 0:
                raise FileNotFoundError("No files found under this path.")
            for fh in fhandlers:
                with fh as fin:
                    with open(os.path.join(d, os.path.basename(fh.path)), "wb") as fout:
                        fout.write(fin.read())
            shutil.make_archive(
                os.path.join(os.path.dirname(d), "model_files"),
                "zip",
                os.path.dirname(d),
                os.path.basename(d),
            )
            return self._upload_artifact(
                f"{os.path.join(os.path.dirname(d), 'model_files')}.zip",
                properties_dict,
            )

    def _upload_artifact(self, model_zip: str, properties: dict) -> str:
        """Uploads the model artifact to cloud storage.

        Args:
            ds_client (DataScienceClient): OCI DataScienceClient
            model_zip (str): path to model artifact zip file
            properties (dict): dictionary of properties

        Returns:
            str: model ocid
        """
        project_id = properties.get("project_id", PROJECT_OCID)
        compartment_id = properties.get("compartment_id", COMPARTMENT_OCID)

        if not project_id or not compartment_id:
            raise ValueError(
                "Both `project_id` and `compartment_id` need to be provided. You can pass them through kwargs or `ModelDeploymentProperties` object."
            )

        create_model_details = CreateModelDetails(
            display_name=properties.get("display_name", None),
            project_id=project_id,
            compartment_id=compartment_id,
        )

        model = self.ds_client.create_model(create_model_details).data
        with open(model_zip, "rb") as data:
            self.ds_client.create_model_artifact(
                model.id,
                data,
                content_disposition=f'attachment; filename="{model.id}.zip"',
            )
        return model.id
