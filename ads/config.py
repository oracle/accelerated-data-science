#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import contextlib
import inspect
import os
from typing import Dict, Optional

from ads.common.config import DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_PROFILE, Config, Mode

OCI_ODSC_SERVICE_ENDPOINT = os.environ.get("OCI_ODSC_SERVICE_ENDPOINT")
OCI_IDENTITY_SERVICE_ENDPOINT = os.environ.get("OCI_IDENTITY_SERVICE_ENDPOINT")
NB_SESSION_COMPARTMENT_OCID = os.environ.get("NB_SESSION_COMPARTMENT_OCID")
PROJECT_OCID = os.environ.get("PROJECT_OCID") or os.environ.get("PIPELINE_PROJECT_OCID")
NB_SESSION_OCID = os.environ.get("NB_SESSION_OCID")
USER_OCID = os.environ.get("USER_OCID")
OCI_RESOURCE_PRINCIPAL_VERSION = os.environ.get("OCI_RESOURCE_PRINCIPAL_VERSION")
OCI_RESOURCE_PRINCIPAL_RPT_PATH = os.environ.get("OCI_RESOURCE_PRINCIPAL_RPT_PATH")
OCI_RESOURCE_PRINCIPAL_RPT_ID = os.environ.get("OCI_RESOURCE_PRINCIPAL_RPT_ID")
TENANCY_OCID = os.environ.get("TENANCY_OCID")
OCI_REGION_METADATA = os.environ.get("OCI_REGION_METADATA")
JOB_RUN_OCID = os.environ.get("JOB_RUN_OCID")
JOB_RUN_COMPARTMENT_OCID = os.environ.get("JOB_RUN_COMPARTMENT_OCID")
PIPELINE_RUN_OCID = os.environ.get("PIPELINE_RUN_OCID")
PIPELINE_RUN_COMPARTMENT_OCID = os.environ.get("PIPELINE_RUN_COMPARTMENT_OCID")
PIPELINE_COMPARTMENT_OCID = os.environ.get("PIPELINE_COMPARTMENT_OCID")

CONDA_BUCKET_NAME = os.environ.get("CONDA_BUCKET_NAME", "service-conda-packs")
CONDA_BUCKET_NS = os.environ.get("CONDA_BUCKET_NS", "id19sfcrra6z")
OCI_RESOURCE_PRINCIPAL_RPT_ENDPOINT = os.environ.get(
    "OCI_RESOURCE_PRINCIPAL_RPT_ENDPOINT"
)
COMPARTMENT_OCID = (
    NB_SESSION_COMPARTMENT_OCID
    or JOB_RUN_COMPARTMENT_OCID
    or PIPELINE_RUN_COMPARTMENT_OCID
)
MD_OCID = os.environ.get("MD_OCID")
DATAFLOW_RUN_OCID = os.environ.get("DATAFLOW_RUN_ID")

RESOURCE_OCID = (
    NB_SESSION_OCID or JOB_RUN_OCID or MD_OCID or PIPELINE_RUN_OCID or DATAFLOW_RUN_OCID
)
NO_CONTAINER = os.environ.get("NO_CONTAINER")


def export(
    uri: Optional[str] = DEFAULT_CONFIG_PATH,
    auth: Dict = None,
    force_overwrite: Optional[bool] = False,
) -> Config:
    """Exports the ADS config.

    Parameters
    ----------
    uri: (str, optional). Defaults to `~/.ads/config`.
        The path to the config file. Can be local or Object Storage file.
    auth: (Dict, optional). Defaults to None.
        The default authentication is set using `ads.set_auth` API. If you need to override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
        authentication signer and kwargs required to instantiate IdentityClient object.
    force_overwrite: (bool, optional). Defaults to `False`.
        Overwrites the config if exists.

    Returns
    -------
    ads.Config
        The ADS config object.
    """
    return Config().load().save(uri=uri, auth=auth, force_overwrite=force_overwrite)


def load(
    uri: Optional[str] = DEFAULT_CONFIG_PATH,
    auth: Dict = None,
    force_overwrite: Optional[bool] = False,
) -> Config:
    """Imports the ADS config.

    The config will be imported from the URI and saved the `~/.ads/config`.

    Parameters
    ----------
    uri: (str, optional). Defaults to `~/.ads/config`.
        The path where the config file needs to be saved. Can be local or Object Storage file.
    auth: (Dict, optional). Defaults to None.
        The default authentication is set using `ads.set_auth` API. If you need to override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
        authentication signer and kwargs required to instantiate IdentityClient object.
    force_overwrite: (bool, optional). Defaults to `False`.
            Overwrites the config if exists.

    Returns
    -------
    ads.Config
        The ADS config object.
    """
    return Config().load(uri=uri, auth=auth).save(force_overwrite=force_overwrite)


@contextlib.contextmanager
def open(
    uri: Optional[str] = DEFAULT_CONFIG_PATH,
    profile: Optional[str] = DEFAULT_CONFIG_PROFILE,
    mode: Optional[str] = Mode.READ,
    auth: Dict = None,
):
    """Context manager helping to read and write config files.

    Parameters
    ----------
    uri: (str, optional). Defaults to `~/.ads/config`.
        The path to the config file. Can be local or Object Storage file.
    profile: (str, optional). Defaults to `DEFAULT`
        The name of the profile to be loaded.
    mode: (str, optional). Defaults to `r`.
        The config mode. Supported values: ['r', 'w']
    auth: (Dict, optional). Defaults to None.
        The default authentication is set using `ads.set_auth` API. If you need to override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
        authentication signer and kwargs required to instantiate IdentityClient object.

    Yields
    ------
    ConfigSection
        The config section object.
    """
    section = profile.upper()
    config = Config(uri=uri, auth=auth)

    try:
        config.load()
    except:
        if mode == Mode.READ:
            raise

    if not config.section_exists(section):
        config[section] = {}

    defined_globals = {}
    section_obj = config[section]
    section_keys = section_obj.keys()
    frame = inspect.currentframe().f_back.f_back

    # Saves original globals and adding new ones
    for key in section_keys:
        if key in frame.f_globals:
            defined_globals[key] = frame.f_globals[key]
        frame.f_globals[key] = section_obj[key]

    frame.f_globals["config"] = section_obj

    try:
        yield section_obj
    finally:
        # Removes config attributes from the globals
        for key in section_keys:
            frame.f_globals.pop(key, None)

        frame.f_globals.pop("config", None)

        # Restores original globals
        for key in defined_globals.keys():
            frame.f_globals[key] = defined_globals[key]

        # Saving config if it necessary
        if mode == Mode.WRITE:
            config.save(force_overwrite=True)
