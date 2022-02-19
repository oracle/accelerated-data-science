#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import oci
import os
from ads.common import utils


def api_keys(
    oci_config: str = os.path.join(os.path.expanduser("~"), ".oci", "config"),
    profile: str = "DEFAULT",
    client_kwargs: dict = None,
) -> dict:
    r"""Prepares authentication and extra arguments necessary for creating clients for different OCI services using API
    Keys.

    Parameters
    ----------
    oci_config : str
        OCI authentication config file location. Default is $HOME/.oci/config.
    profile : str
        Profile name to select from the config file. The defautl is DEFAULT
    client_kwargs : dict
        kwargs that are required to instantiate the Client if we need to override the defaults.

    Returns
    -------
    dict
        Contains keys - config, signer and client_kwargs.

        - The config contains the config loaded from the configuration loaded from `oci_config`.
        - The signer contains the signer object created from the api keys.
        - client_kwargs contains the `client_kwargs` that was passed in as input parameter.

    Examples
    --------
    >>> from ads.common import auth as authutil
    >>> from ads.common import oci_client as oc
    >>> auth = authutil.api_keys(oci_config="/home/datascience/.oci/config", profile="TEST", client_kwargs={"timeout": 6000})
    >>> oc.OCIClientFactory(**auth).object_storage # Creates Object storage client with timeout set to 6000 using API Key authentication
    """
    configuration = oci.config.from_file(oci_config, profile)
    return {
        "config": configuration,
        "signer": oci.signer.Signer(
            configuration["tenancy"],
            configuration["user"],
            configuration["fingerprint"],
            configuration["key_file"],
            configuration.get("pass_phrase"),
        ),
        "client_kwargs": client_kwargs,
    }


def resource_principal(client_kwargs=None):
    r"""Prepares authentication and extra arguments necessary for creating clients for different OCI services using
    Resource Principals.

    Parameters
    ----------
    client_kwargs : dict
        kwargs that are required to instantiate the Client if we need to override the defaults.

    Returns
    -------
    dict
        Contains keys - config, signer and client_kwargs.

        - The config contains and empty dictionary.
        - The signer contains the signer object created from the resource principal.
        - client_kwargs contains the `client_kwargs` that was passed in as input parameter.

    Examples
    --------
    >>> from ads.common import auth as authutil
    >>> from ads.common import oci_client as oc
    >>> auth = authutil.resource_principal({"timeout": 6000})
    >>> oc.OCIClientFactory(**auth).object_storage # Creates Object Storage client with timeout set to 6000 seconds using resource principal authentication
    """

    return {
        "config": {},
        "signer": oci.auth.signers.get_resource_principals_signer(),
        "client_kwargs": client_kwargs,
    }


def default_signer(client_kwargs=None):
    r"""Prepares authentication and extra arguments necessary for creating clients for different OCI services based on
    the default authentication setting for the session. Refer ads.set_auth API for further reference.

    Parameters
    ----------
    client_kwargs : dict
        kwargs that are required to instantiate the Client if we need to override the defaults.

    Returns
    -------
    dict
        Contains keys - config, signer and client_kwargs.

        - The config contains the config loaded from the configuration loaded from the default location if the default
          auth mode is API keys, otherwise it is empty dictionary.
        - The signer contains the signer object created from default auth mode.
        - client_kwargs contains the `client_kwargs` that was passed in as input parameter.

    Examples
    --------
    >>> from ads.common import auth as authutil
    >>> from ads.common import oci_client as oc
    >>> auth = authutil.default_signer()
    >>> oc.OCIClientFactory(**auth).object_storage # Creates Object storage client
    """

    if utils.is_resource_principal_mode():
        return resource_principal(client_kwargs)
    else:
        return api_keys(client_kwargs=client_kwargs, profile=utils.oci_key_profile())


def get_signer(oci_config=None, oci_profile=None, **client_kwargs):
    if oci_config and oci_profile:
        return api_keys(oci_config, oci_profile, client_kwargs)
    else:
        return resource_principal(client_kwargs)
