#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any

import oci
from oci.config import (
    DEFAULT_LOCATION,  # "~/.oci/config"
    DEFAULT_PROFILE,  # "DEFAULT"
)

import ads.telemetry
from ads.common import logger
from ads.common.extended_enum import ExtendedEnumMeta


class AuthType(str, metaclass=ExtendedEnumMeta):
    API_KEY = "api_key"
    RESOURCE_PRINCIPAL = "resource_principal"
    INSTANCE_PRINCIPAL = "instance_principal"


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass()
class AuthState(metaclass=SingletonMeta):
    """
    Class stores state of variables specified for auth method, configuration,
    configuration file location, profile name, signer or signer_callable, which
    set by use at any given time and can be provided by this class in any ADS module.
    """

    oci_iam_type: str = None
    oci_cli_auth: str = None
    oci_config_path: str = None
    oci_key_profile: str = None
    oci_config: str = None
    oci_signer: str = None
    oci_signer_callable: str = None
    oci_signer_kwargs: str = None

    def __post_init__(self):
        self.oci_iam_type = self.oci_iam_type or os.environ.get(
            "OCI_IAM_TYPE", AuthType.API_KEY
        )
        self.oci_cli_auth = self.oci_cli_auth or os.environ.get(
            "OCI_CLI_AUTH", AuthType.RESOURCE_PRINCIPAL
        )
        self.oci_config_path = self.oci_config_path or os.environ.get(
            "OCI_CONFIG_LOCATION", DEFAULT_LOCATION
        )
        self.oci_key_profile = self.oci_key_profile or os.environ.get(
            "OCI_CONFIG_PROFILE", DEFAULT_PROFILE
        )
        self.oci_config = self.oci_config or {}
        self.oci_signer = self.oci_signer
        self.oci_signer_callable = self.oci_signer_callable
        self.oci_signer_kwargs = self.oci_signer_kwargs or {}


def set_auth(
    auth: Optional[str] = AuthType.API_KEY,
    oci_config_location: Optional[str] = DEFAULT_LOCATION,
    profile: Optional[str] = DEFAULT_PROFILE,
    config: Optional[Dict] = {},
    signer: Optional[Any] = None,
    signer_callable: Optional[Callable] = None,
    signer_kwargs: Optional[Dict] = {},
) -> None:
    """
    Save type of authentication, profile, config location, config (keypair identity) or signer, which will be used
    when actual creation of config or signer happens.

    Parameters
    ----------
    auth: Optional[str], default 'api_key'
        'api_key', 'resource_principal' or 'instance_principal'. Enable/disable resource principal identity,
         instance principal or keypair identity in a notebook session
    oci_config_location: Optional[str], default oci.config.DEFAULT_LOCATION, which is '~/.oci/config'
        config file location
    profile: Optional[str], default is DEFAULT_PROFILE, which is 'DEFAULT'
         profile name for api keys config file
    config: Optional[Dict], default {}
        created config dictionary
    signer: Optional[Any], default None
        created signer, can be resource principals signer, instance principal signer or other.
        Check documentation for more signers: https://docs.oracle.com/en-us/iaas/tools/python/latest/api/signing.html
    signer_callable: Optional[Callable], default None
        a callable object that returns signer
    signer_kwargs: Optional[Dict], default None
        parameters accepted by the signer.
        Check documentation: https://docs.oracle.com/en-us/iaas/tools/python/latest/api/signing.html

    Examples
    --------
    >>> ads.set_auth("api_key") # default signer is set to api keys

    >>> ads.set_auth("api_key", profile = "TEST") # default signer is set to api keys and to use TEST profile

    >>> ads.set_auth("api_key", oci_config_location = "other_config_location") # use non-default oci_config_location

    >>> other_config = oci.config.from_file("other_config_location", "OTHER_PROFILE") # Create non-default config
    >>> ads.set_auth(config=other_config) # Set api keys type of authentication based on provided config

    >>> ads.set_auth("resource_principal")  # Set resource principal authentication

    >>> ads.set_auth("instance_principal")  # Set instance principal authentication

    >>> singer = oci.auth.signers.get_resource_principals_signer()
    >>> ads.auth.create_signer(config={}, singer=signer) # resource principals authentication dictionary created

    >>> signer_callable = oci.auth.signers.ResourcePrincipalsFederationSigner
    >>> ads.set_auth(signer_callable=signer_callable) # Set resource principal federation singer callable

    >>> signer_callable = oci.auth.signers.InstancePrincipalsSecurityTokenSigner
    >>> signer_kwargs = dict(log_requests=True) # will log the request url and response data when retrieving
    >>> # instance principals authentication dictionary created based on callable with kwargs parameters:
    >>> ads.set_auth(signer_callable=signer_callable, signer_kwargs=signer_kwargs)
    """
    auth_state = AuthState()

    valid_auth_keys = AuthFactory.classes.keys()
    if auth in valid_auth_keys:
        auth_state.oci_iam_type = auth
    else:
        raise ValueError(
            f"Allowed values are: {valid_auth_keys}. If you wish to use other authentication form, "
            f"pass a valid signer or use signer_callable and signer_kwargs"
        )

    if config and (
        oci_config_location != DEFAULT_LOCATION or profile != DEFAULT_PROFILE
    ):
        raise ValueError(
            f"'config' and 'oci_config_location', 'profile' pair are mutually exclusive."
            f"Please specify 'config' OR 'oci_config_location', 'profile' pair."
        )

    auth_state.oci_config = config
    auth_state.oci_key_profile = profile
    if os.path.exists(os.path.expanduser(oci_config_location)):
        auth_state.oci_config_path = oci_config_location
    else:
        raise ValueError(
            f"{oci_config_location} path does not exist, please provide existing path to config file."
        )

    auth_state.oci_signer = signer
    auth_state.oci_signer_callable = signer_callable
    auth_state.oci_signer_kwargs = signer_kwargs


def api_keys(
    oci_config: str = os.path.join(os.path.expanduser("~"), ".oci", "config"),
    profile: str = DEFAULT_PROFILE,
    client_kwargs: Dict = None,
) -> Dict:
    """
    Prepares authentication and extra arguments necessary for creating clients for different OCI services using API
    Keys.

    Parameters
    ----------
    oci_config: Optional[str], default is $HOME/.oci/config
        OCI authentication config file location.
    profile: Optional[str], is DEFAULT_PROFILE, which is 'DEFAULT'
        Profile name to select from the config file.
    client_kwargs: Optional[Dict], default None
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
    >>> from ads.common import oci_client as oc
    >>> auth = ads.auth.api_keys(oci_config="/home/datascience/.oci/config", profile="TEST", client_kwargs={"timeout": 6000})
    >>> oc.OCIClientFactory(**auth).object_storage # Creates Object storage client with timeout set to 6000 using API Key authentication
    """
    signer_args = dict(
        oci_config_location=oci_config,
        oci_key_profile=profile,
        client_kwargs=client_kwargs,
    )
    signer_generator = AuthFactory().signerGenerator(AuthType.API_KEY)
    return signer_generator(signer_args).create_signer()


def resource_principal(
    client_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Prepares authentication and extra arguments necessary for creating clients for different OCI services using
    Resource Principals.

    Parameters
    ----------
    client_kwargs: Dict, default None
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
    >>> from ads.common import oci_client as oc
    >>> auth = ads.auth.resource_principal({"timeout": 6000})
    >>> oc.OCIClientFactory(**auth).object_storage # Creates Object Storage client with timeout set to 6000 seconds using resource principal authentication
    """
    signer_args = dict(client_kwargs=client_kwargs)
    signer_generator = AuthFactory().signerGenerator(AuthType.RESOURCE_PRINCIPAL)
    return signer_generator(signer_args).create_signer()


def create_signer(
    auth_type: Optional[str] = AuthType.API_KEY,
    oci_config_location: Optional[str] = DEFAULT_LOCATION,
    profile: Optional[str] = DEFAULT_PROFILE,
    config: Optional[Dict] = {},
    signer: Optional[Any] = None,
    signer_callable: Optional[Callable] = None,
    signer_kwargs: Optional[Dict] = {},
    client_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Prepares authentication and extra arguments necessary for creating clients for different OCI services based on
    provided parameters. If `signer` or `signer`_callable` provided, authentication with that signer will be created.
    If `config` provided, api_key type of authentication will be created. Accepted values for `auth_type`:
    `api_key` (default), 'instance_principal', 'resource_principal'.

    Parameters
    ----------
    auth: Optional[str], default 'api_key'
        'api_key', 'resource_principal' or 'instance_principal'. Enable/disable resource principal identity,
         instance principal or keypair identity in a notebook session
    oci_config_location: Optional[str], default oci.config.DEFAULT_LOCATION, which is '~/.oci/config'
        config file location
    profile: Optional[str], default is DEFAULT_PROFILE, which is 'DEFAULT'
         profile name for api keys config file
    config: Optional[Dict], default {}
        created config dictionary
    signer: Optional[Any], default None
        created signer, can be resource principals signer, instance principal signer or other.
        Check documentation for more signers: https://docs.oracle.com/en-us/iaas/tools/python/latest/api/signing.html
    signer_callable: Optional[Callable], default None
        a callable object that returns signer
    signer_kwargs: Optional[Dict], default None
        parameters accepted by the signer.
        Check documentation: https://docs.oracle.com/en-us/iaas/tools/python/latest/api/signing.html
    client_kwargs : dict
        kwargs that are required to instantiate the Client if we need to override the defaults

    Examples
    --------
    >>> import ads
    >>> auth = ads.auth.create_signer() # api_key type of authentication dictionary created with default config location and default profile

    >>> config = oci.config.from_file("other_config_location", "OTHER_PROFILE")
    >>> auth = ads.auth.create_signer(config=config) # api_key type of authentication dictionary created based on provided config

    >>> singer = oci.auth.signers.get_resource_principals_signer()
    >>> auth = ads.auth.create_signer(config={}, singer=signer) # resource principals authentication dictionary created

    >>> auth = ads.auth.create_signer(auth_type='instance_principal') # instance principals authentication dictionary created

    >>> signer_callable = oci.auth.signers.InstancePrincipalsSecurityTokenSigner
    >>> signer_kwargs = dict(log_requests=True) # will log the request url and response data when retrieving
    >>> auth = ads.auth.create_signer(signer_callable=signer_callable, signer_kwargs=signer_kwargs) # instance principals authentication dictionary created based on callable with kwargs parameters
    """
    if signer or signer_callable:
        configuration = ads.telemetry.update_oci_client_config()
        if signer_callable:
            signer = signer_callable(**signer_kwargs)
        signer_dict = {
            "config": configuration,
            "signer": signer,
            "client_kwargs": client_kwargs,
        }
        logger.info(f"Using authentication signer type {type(signer)}.")
        return signer_dict
    else:
        signer_args = dict(
            oci_config_location=oci_config_location,
            oci_key_profile=profile,
            oci_config=config,
            client_kwargs=client_kwargs,
        )
        if config:
            auth_type = AuthType.API_KEY
        signer_generator = AuthFactory().signerGenerator(auth_type)
        return signer_generator(signer_args).create_signer()


def default_signer(client_kwargs: Optional[Dict] = None) -> Dict:
    """
    Prepares authentication and extra arguments necessary for creating clients for different OCI services based on
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
    >>> import ads
    >>> from ads.common import oci_client as oc

    >>> auth = ads.auth.default_signer()
    >>> oc.OCIClientFactory(**auth).object_storage # Creates Object storage client

    >>> ads.set_auth("resource_principal")
    >>> auth = ads.auth.default_signer()
    >>> oc.OCIClientFactory(**auth).object_storage # Creates Object storage client using resource principal authentication

    >>> signer_callable = oci.auth.signers.InstancePrincipalsSecurityTokenSigner
    >>> ads.set_auth(signer_callable=signer_callable) # Set instance principal callable
    >>> auth = ads.auth.default_signer() # signer_callable instantiated
    >>> oc.OCIClientFactory(**auth).object_storage # Creates Object storage client using instance principal authentication
    """
    auth_state = AuthState()
    if auth_state.oci_signer or auth_state.oci_signer_callable:
        configuration = ads.telemetry.update_oci_client_config()
        signer = auth_state.oci_signer
        if auth_state.oci_signer_callable:
            signer_kwargs = auth_state.oci_signer_kwargs or {}
            signer = auth_state.oci_signer_callable(**signer_kwargs)
        signer_dict = {
            "config": configuration,
            "signer": signer,
            "client_kwargs": client_kwargs,
        }
        logger.info(f"Using authentication signer type {type(signer)}.")
        return signer_dict
    else:
        signer_args = dict(
            oci_config_location=auth_state.oci_config_path,
            oci_key_profile=auth_state.oci_key_profile,
            oci_config=auth_state.oci_config,
            client_kwargs=client_kwargs,
        )
        signer_generator = AuthFactory().signerGenerator(auth_state.oci_iam_type)
        return signer_generator(signer_args).create_signer()


def get_signer(
    oci_config: Optional[str] = None, oci_profile: Optional[str] = None, **client_kwargs
) -> Dict:
    """
    Provides config and signer based given parameters. If oci_config (api key config file location) and
    oci_profile specified new signer will ge generated. Else singer of a type specified in OCI_CLI_AUTH
    environment variable will be used to generate signer and return. If OCI_CLI_AUTH not set,
    resource principal signer will be provided. Accepted values for OCI_CLI_AUTH: 'api_key',
    'instance_principal', 'resource_principal'.

    Parameters
    ----------
    oci_config: Optional[str], default None
        Path to the config file
    oci_profile: Optional[str], default None
        the profile to load from the config file
    client_kwargs:
        kwargs that are required to instantiate the Client if we need to override the defaults
    """
    if oci_config and oci_profile:
        signer_args = dict(
            oci_config_location=oci_config,
            oci_key_profile=oci_profile,
            client_kwargs=client_kwargs,
        )
        signer_generator = AuthFactory().signerGenerator(AuthType.API_KEY)
    else:
        oci_cli_auth = (
            AuthState().oci_cli_auth
        )  # "resource_principal", if env variable OCI_CLI_AUTH not set
        valid_auth_keys = AuthFactory.classes.keys()
        if oci_cli_auth not in valid_auth_keys:
            oci_cli_auth = AuthType.RESOURCE_PRINCIPAL
        signer_args = dict(
            client_kwargs=client_kwargs,
        )
        signer_generator = AuthFactory().signerGenerator(oci_cli_auth)
    return signer_generator(signer_args).create_signer()


class AuthSignerGenerator:  # pragma: no cover
    """
    Abstract class for auth configuration and signer creation.
    """

    def create_signer(self):
        pass


class APIKey(AuthSignerGenerator):
    """
    Creates api keys auth instance. This signer is intended to be used when signing requests for
    a given user - it requires that user’s ID, their private key and certificate fingerprint.
    It prepares extra arguments necessary for creating clients for variety of OCI services.
    """

    def __init__(self, args: Optional[Dict] = None):
        """
        Signer created based on args provided. If not provided current values of according arguments
        will be used from current global state from AuthState class.

        Parameters
        ----------
        args: dict
            args that are required to create api key config and signer. Contains keys: oci_config,
            oci_config_location, oci_key_profile, client_kwargs.

            - oci_config is a configuration dict that can be used to create clients
            - oci_config_location - path to config file
            - oci_key_profile - the profile to load from config file
            - client_kwargs - optional parameters for OCI client creation in next steps
        """
        self.oci_config = args.get("oci_config")
        self.oci_config_location = args.get("oci_config_location")
        self.oci_key_profile = args.get("oci_key_profile")
        self.client_kwargs = args.get("client_kwargs")

    def create_signer(self) -> Dict:
        """
        Creates api keys configuration and signer with extra arguments necessary for creating clients.
        Signer constructed from the `oci_config` provided. If not 'oci_config', configuration will be
        constructed from 'oci_config_location' and 'oci_key_profile' in place.

        Resturns
        --------
        dict
            Contains keys - config, signer and client_kwargs.

            - config contains the configuration information
            - signer contains the signer object created. It is instantiated from signer_callable, or
            signer provided in args used, or instantiated in place
            - client_kwargs contains the `client_kwargs` that was passed in as input parameter

        Examples
        --------
        >>> signer_args = dict(
        >>>     client_kwargs=client_kwargs
        >>> )
        >>> signer_generator = AuthFactory().signerGenerator(AuthType.API_KEY)
        >>> signer_generator(signer_args).create_signer()
        """
        if self.oci_config:
            configuration = ads.telemetry.update_oci_client_config(self.oci_config)
        else:
            configuration = ads.telemetry.update_oci_client_config(
                oci.config.from_file(self.oci_config_location, self.oci_key_profile)
            )
        logger.info(f"Using 'api_key' authentication.")
        return {
            "config": configuration,
            "signer": oci.signer.Signer(
                configuration["tenancy"],
                configuration["user"],
                configuration["fingerprint"],
                configuration["key_file"],
                configuration.get("pass_phrase"),
            ),
            "client_kwargs": self.client_kwargs,
        }


class ResourcePrincipal(AuthSignerGenerator):
    """
    Creates Resource Principal signer - a security token for a resource principal.
    It prepares extra arguments necessary for creating clients for variety of OCI services.
    """

    def __init__(self, args: Optional[Dict] = None):
        """
        Signer created based on args provided. If not provided current values of according arguments
        will be used from current global state from AuthState class.

        Parameters
        ----------
        args: dict
            args that are required to create Resource Principal signer. Contains keys: client_kwargs.

            - client_kwargs - optional parameters for OCI client creation in next steps
        """
        self.client_kwargs = args.get("client_kwargs")

    def create_signer(self) -> Dict:
        """
        Creates Resource Principal signer with extra arguments necessary for creating clients.

        Resturns
        --------
        dict
            Contains keys - config, signer and client_kwargs.

            - config contains the configuration information
            - signer contains the signer object created. It is instantiated from signer_callable, or
            signer provided in args used, or instantiated in place
            - client_kwargs contains the `client_kwargs` that was passed in as input parameter

        Examples
        --------
        >>> signer_args = dict(
        >>>     signer=oci.auth.signers.get_resource_principals_signer()
        >>> )
        >>> signer_generator = AuthFactory().signerGenerator(AuthType.RESOURCE_PRINCIPAL)
        >>> signer_generator(signer_args).create_signer()
        """
        configuration = ads.telemetry.update_oci_client_config()
        signer_dict = {
            "config": configuration,
            "signer": oci.auth.signers.get_resource_principals_signer(),
            "client_kwargs": self.client_kwargs,
        }
        logger.info(f"Using 'resource_principal' authentication.")
        return signer_dict


class InstancePrincipal(AuthSignerGenerator):
    """
    Creates Instance Principal signer - a SecurityTokenSigner which uses a security token for an instance
    principal. It prepares extra arguments necessary for creating clients for variety of OCI services.
    """

    def __init__(self, args: Optional[Dict] = None):
        """
        Signer created based on args provided. If not provided current values of according arguments
        will be used from current global state from AuthState class.

        Parameters
        ----------
        args: dict
            args that are required to create Instance Principal signer. Contains keys: signer_kwargs, client_kwargs.

            - signer_kwargs - optional parameters required to instantiate instance principal signer
            - client_kwargs - optional parameters for OCI client creation in next steps
        """
        self.signer_kwargs = args.get("signer_kwargs", dict())
        self.client_kwargs = args.get("client_kwargs")

    def create_signer(self) -> Dict:
        """
        Creates Instance Principal signer with extra arguments necessary for creating clients.
        Signer instantiated from the `signer_callable` or if the `signer` provided is will be return by this method.
        If `signer_callable` or `signer` not provided new signer will be created in place.

        Resturns
        --------
        dict
            Contains keys - config, signer and client_kwargs.

            - config contains the configuration information
            - signer contains the signer object created. It is instantiated from signer_callable, or
            signer provided in args used, or instantiated in place
            - client_kwargs contains the `client_kwargs` that was passed in as input parameter

        Examples
        --------
        >>> signer_args = dict(signer_kwargs={"log_requests": True})
        >>> signer_generator = AuthFactory().signerGenerator(AuthType.INSTANCE_PRINCIPAL)
        >>> signer_generator(signer_args).create_signer()
        """
        configuration = ads.telemetry.update_oci_client_config()
        signer_dict = {
            "config": configuration,
            "signer": oci.auth.signers.InstancePrincipalsSecurityTokenSigner(
                **self.signer_kwargs
            ),
            "client_kwargs": self.client_kwargs,
        }
        logger.info(f"Using 'instance_principal' authentication.")
        return signer_dict


class AuthFactory:
    """
    AuthFactory class which contains list of registered signers and alllows to register new signers.
    Check documentation for more signers: https://docs.oracle.com/en-us/iaas/tools/python/latest/api/signing.html.

    Current signers:
        * APIKey
        * ResourcePrincipal
        * InstancePrincipal
    """

    classes = {
        AuthType.API_KEY: APIKey,
        AuthType.RESOURCE_PRINCIPAL: ResourcePrincipal,
        AuthType.INSTANCE_PRINCIPAL: InstancePrincipal,
    }

    @classmethod
    def register(cls, signer_type: str, signer: Any) -> None:
        """Registers a new signer.

        Parameters
        ----------
        signer_type: str
            Singer type to be registers
        signer: RecordParser
            A new Singer class to be registered.

        Returns
        -------
        None
            Nothing.
        """
        cls.classes[signer_type] = signer

    def signerGenerator(self, iam_type: Optional[str] = "api_key"):
        """
        Generates signer classes based of iam_type, which specify one of auth methods:
        'api_key', 'resource_principal' or 'instance_principal'.

        Parameters
        ----------
        iam_type: str, default 'api_key'
            type of auth provided in IAM_TYPE environment variable or set in parameters in
            ads.set_auth() method.

        Returns
        -------
        :class:`APIKey` or :class:`ResourcePrincipal` or :class:`InstancePrincipal`
            returns one of classes, which implements creation of signer of specified type

        Raises
        ------
        ValueError
            If iam_type is not supported.
        """

        valid_auth_keys = AuthFactory.classes.keys()
        if iam_type in valid_auth_keys:
            return AuthFactory.classes[iam_type]
        else:
            raise ValueError(
                f"Allowed values are: {valid_auth_keys}. If you wish to use other authentication form, "
                f"pass a valid signer or use signer_callable and signer_kwargs"
            )


class OCIAuthContext:
    """
    OCIAuthContext used in 'with' statement for properly managing global authentication type
    and global configuration profile parameters.

    Examples
    --------
    >>> from ads.jobs import DataFlowRun
    >>> with OCIAuthContext(profile='TEST'):
    >>>     df_run = DataFlowRun.from_ocid(run_id)
    """

    def __init__(self, profile: str = None):
        """
        Initialize class OCIAuthContext and saves global state of authentication type and configuration profile.

        Parameters
        ----------
        profile: str, default is None
            profile name for api keys config file
        """
        self.profile = profile
        self.prev_mode = AuthState().oci_iam_type
        self.prev_profile = AuthState().oci_key_profile
        self.oci_cli_auth = AuthState().oci_cli_auth

    def __enter__(self):
        """
        When called by the 'with' statement and if 'profile' provided - 'api_key' authentication with 'profile' used.
        If 'profile' not provided, authentication method will be 'resource_principal'.
        """
        if self.profile:
            ads.set_auth(auth=AuthType.API_KEY, profile=self.profile)
            logger.info(f"OCI profile set to {self.profile}")
        else:
            ads.set_auth(auth=AuthType.RESOURCE_PRINCIPAL)
            logger.info(f"OCI auth set to resource principal")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        When called by the 'with' statement restores initial state of authentication type and profile value.
        """
        ads.set_auth(auth=self.prev_mode, profile=self.prev_profile)
