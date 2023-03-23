#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from collections import defaultdict
from configparser import ConfigParser
from copy import copy
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import fsspec
import yaml

from ads.common import auth as authutil
from ads.common.decorator.argument_to_case import ArgumentCase, argument_to_case

try:
    from yaml import CSafeDumper as dumper
except:
    from yaml import SafeDumper as dumper


DEFAULT_CONFIG_PROFILE = "DEFAULT"
DEFAULT_CONFIG_PATH = "~/.ads/config"


class EventType(Enum):
    CHANGE = "change"


class Mode:
    READ = "r"
    WRITE = "w"


class Eventing:
    """The class helper to register event handlers."""

    def __init__(self):
        self._events = defaultdict(set)

    def trigger(self, event: str) -> None:
        """Triggers all the registered callbacks for the particular event."""
        for callback in self._events[event]:
            callback()

    def on(self, event_name: str, callback: Callable) -> None:
        """Registers a callback for the particular event."""
        self._events[event_name].add(callback)


class ConfigSection:
    """The class representing a config section."""

    def __init__(self):
        """Initializes the config section instance."""
        self.events = Eventing()
        self._info = {}

    def clear(self) -> None:
        """Clears the config section values.

        Returns
        -------
        None
            Nothing
        """
        self._info = {}
        self.events.trigger(EventType.CHANGE.value)

    def copy(self) -> "ConfigSection":
        """Makes a copy of a config section.

        Returns
        -------
        ConfigSection
            The instance of a copied ConfigSection.
        """
        return self.__class__()._with_dict(info=copy(self._info), replace=True)

    def _with_dict(
        self, info: Dict[str, Any], replace: Optional[bool] = False
    ) -> "ConfigSection":
        """Populates the config section from a dictionary.

        Parameters
        ----------
        info: Dict[str, Any]
            The config section information in a dictionary format.
        replace: (bool, optional). Defaults to False.
            If set as True, overwrites config section with the new information.

        Returns
        -------
        ConfigSection
            The instance of a ConfigSection.

        Raises
        -----
        TypeError
            If input data is not a dictionary.
        ValueError
            If config section is already contain provided fields
            and `replace` flag set to False.
        """

        if not isinstance(info, dict):
            raise TypeError("The `info` must be a dictionary.")

        common_keys = list(
            set(self._info.keys()).intersection(set(list(map(str.lower, info.keys()))))
        )

        if common_keys and not replace:
            raise ValueError(
                f"The config section is already contain fields: {common_keys}. "
                "Use `replace=True` to overwrite."
            )

        for k, v in info.items():
            self._set(key=k.lower(), value=v, replace=replace)

        return self

    def with_dict(
        self, info: Dict[str, Any], replace: Optional[bool] = False
    ) -> "ConfigSection":
        """Populates the config section from a dictionary.

        Parameters
        ----------
        info: Dict[str, Any]
            The config section information in a dictionary format.
        replace: (bool, optional). Defaults to False.
            If set as True, overwrites config section with the new information.
        """
        self._with_dict(info=info, replace=replace)
        self.events.trigger(EventType.CHANGE.value)
        return self

    def keys(self) -> Tuple[str]:
        """Gets the list of the keys of a config section.

        Returns
        -------
        Tuple[str]
            The list of config section keys.
        """
        return tuple(self._info.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Converts config section to a dictionary.

        Returns
        -------
        Dict[str, Any]
            The config section in a dictionary format.
        """
        return self._info

    @argument_to_case(case=ArgumentCase.LOWER, arguments=["key"])
    def get(self, key: str) -> str:
        """Gets the config section value by key.

        Returns
        -------
        str
            A specific config section value.
        """
        return self._info.get(key)

    @argument_to_case(case=ArgumentCase.LOWER, arguments=["key"])
    def _set(self, key: str, value: str, replace: Optional[bool] = False) -> None:
        """Sets the config section value by key.

        Parameters
        ----------
        key: str
            The config section field key.
        value: str
            The config section field value.

        Returns
        -------
        None
            Nothing

        Raises
        ------
        ValueError
            In case when field with provided key already exists and
            `replace` flag set to False.
        """
        if self._info.get(key) == value:
            return

        if key in self._info and not replace:
            raise ValueError(
                f"The field with key `{key}` already exists. "
                "Use `replace=True` to overwrite."
            )

        self._info[key] = value

    def set(self, key: str, value: str, replace: Optional[bool] = False) -> None:
        """Sets the config section value by key.

        Parameters
        ----------
        key: str
            The config section field key.
        value: str
            The config section field value.

        Returns
        -------
        None
            Nothing
        """
        self._set(key, value, replace)
        self.events.trigger(EventType.CHANGE.value)

    @argument_to_case(case=ArgumentCase.LOWER, arguments=["key"])
    def remove(self, key: str) -> None:
        """Removes the config section field by key.

        Parameters
        ----------
        key: str
            The config section field key.

        Returns
        -------
        None
            Nothing
        """
        self._info.pop(key, None)
        self.events.trigger(EventType.CHANGE.value)

    def __getitem__(self, key: str):
        return self.get(key)

    def __setitem__(self, key: str, value: str):
        self.set(key=key, value=value, replace=True)

    def __bool__(self):
        return any(self._info.values())

    def __repr__(self):
        return yaml.dump(self.to_dict(), Dumper=dumper)


class Config:
    """The class representing a config."""

    __DEFAULT_SECTIONS = {
        DEFAULT_CONFIG_PROFILE: ConfigSection,
    }

    def __init__(
        self,
        uri: Optional[str] = DEFAULT_CONFIG_PATH,
        auth: Optional[Dict] = None,
    ):
        """Initializes a config instance.

        Parameters
        ----------
        uri: (str, optional). Defaults to `~/.ads/config`.
            The path to the config file. Can be local or Object Storage file.
        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        """
        self._config = {}
        self.auth = auth or authutil.default_signer()

        # configure default config sections
        for key, default_section in self.__DEFAULT_SECTIONS.items():
            self._config[key] = default_section()
            self._config[key].events.on(EventType.CHANGE.value, self._on_change)

        self.uri = os.path.expanduser(uri)
        self._config_parser = ExtendedConfigParser(uri=self.uri, auth=self.auth)

    def _on_change(self):
        """This method will be called when config modified."""
        pass

    def default(self) -> ConfigSection:
        """Gets default config section.

        Returns
        -------
        ConfigSection
            A default config section.
        """
        return self.section_get(DEFAULT_CONFIG_PROFILE)

    @argument_to_case(case=ArgumentCase.UPPER, arguments=["key"])
    def section_exists(self, key: str) -> bool:
        """Checks if a config section exists.

        Parameters
        ----------
        key: str
            A key of a config section.

        Returns
        -------
        bool
            True if a config section exists, Fasle otherwise.
        """
        return key in self._config

    @argument_to_case(case=ArgumentCase.UPPER, arguments=["key"])
    def section_get(self, key: str) -> ConfigSection:
        """Gets the config section by key.

        Returns
        -------
        ConfigSection
            A config section object.

        Raises
        ------
        KeyError
            If a config section not exists.
        """
        if key not in self._config:
            raise KeyError(f"The config section `{key}` not found.")
        return self._config.get(key)

    @argument_to_case(case=ArgumentCase.UPPER, arguments=["key"])
    def section_set(
        self,
        key: str,
        info: Union[dict, ConfigSection],
        replace: Optional[bool] = False,
    ) -> ConfigSection:
        """
        Sets a config section to config.
        The new config section will be added in case if it doesn't exist.
        Otherwise the existing config section will be merged with the new fields.

        Parameters
        ----------
        key: str
            A key of a config section.
        info: Union[dict, ConfigSection]
            The config section information in a dictionary or ConfigSection format.
        replace: (bool, optional). Defaults to False.
            If set as True, overwrites config section with the new information.

        Returns
        -------
        ConfigSection
            A config section object.

        Raises
        ------
        ValueError
            If section with given key is already exist and `replace` flag set to False.
        TypeError
            If input `info` has a wrong format.
        """
        if key in self._config and not replace:
            raise ValueError(
                f"A config section `{key}` is already exist. "
                "Use `replace=True` if you want to overwrite."
            )

        if not isinstance(info, (dict, ConfigSection)):
            raise TypeError(
                "Parameter `info` must be either a `dictionary` or `ConfigSection` object."
            )

        if key not in self._config:
            self._config[key] = ConfigSection()
            self._config[key].events.on(EventType.CHANGE.value, self._on_change)

        if isinstance(info, ConfigSection):
            self._config[key].with_dict(info.copy().to_dict(), replace=replace)
        else:
            self._config[key].with_dict(copy(info), replace=replace)

        return self._config[key]

    @argument_to_case(case=ArgumentCase.UPPER, arguments=["key"])
    def section_remove(self, key: str) -> "Config":
        """Removes config section form config.

        Parameters
        ----------
        key: str
            A key of a config section that needs to be removed.

        Returns
        -------
        None
            Nothing
        """
        self._config.pop(key, None)
        self._on_change()
        return self

    def save(
        self,
        uri: Optional[str] = None,
        auth: Optional[Dict] = None,
        force_overwrite: Optional[bool] = False,
    ) -> "Config":
        """Saves config to a config file.

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
        None
            Nothing
        """
        uri = uri or self.uri
        auth = auth or self.auth or authutil.default_signer()
        self._config_parser.with_dict(self.to_dict()).save(
            uri=uri, auth=auth, force_overwrite=force_overwrite
        )
        return self

    def load(self, uri: Optional[str] = None, auth: Optional[Dict] = None) -> "Config":
        """Loads config from a config file.

        Parameters
        ----------
        uri: (str, optional). Defaults to `~/.ads/config`.
            The path where the config file needs to be saved. Can be local or Object Storage file.
        auth: (Dict, optional). Defaults to None.
            The default authentication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        Config
            A config object.
        """
        uri = uri or self.uri
        auth = auth or self.auth or authutil.default_signer()

        return self.with_dict(
            self._config_parser.read(uri=uri, auth=auth).to_dict(), replace=True
        )

    def with_dict(
        self,
        info: Dict[str, Union[Dict[str, Any], ConfigSection]],
        replace: Optional[bool] = False,
    ) -> "Config":
        """Merging dictionary to config.

        Parameters
        ----------
        info: Dict[str, Union[Dict[str, Any], ConfigSection]]
            A dictionary that needs to be merged to config.
        replace: (bool, optional). Defaults to False.
            If set as True, overwrites config section with the new information.

        Returns
        -------
        Config
            A config object.
        """

        self._validate(info)
        try:
            for key, value in info.items():
                self.section_set(key, value, replace=replace)
        finally:
            self._on_change()

        return self

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Converts config to a dictionary format.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            A config in a dictionary format.
        """
        return {key: value.to_dict() for key, value in self._config.items()}

    def keys(self) -> List[str]:
        """Gets the all registered config section keys.

        Returns
        -------
        List[str]
            The list of the all registered config section keys.
        """
        return self._config.keys()

    def _validate(self, info: Dict[str, Union[Dict[str, Any], ConfigSection]]) -> None:
        """Validates input dictionary."""
        if not info or not isinstance(info, Dict):
            raise TypeError("The input data should be a dictionary.")
        for key, value in info.items():
            if value and not isinstance(value, (Dict, ConfigSection)):
                raise ValueError(
                    f"The `{key}` must be a dictionary or a `ConfigSection` instance."
                )

    def __getitem__(self, key: str):
        return self.section_get(key)

    def __setitem__(self, key, value: Union[Dict, ConfigSection]):
        self.section_set(key, value, replace=True)

    def __repr__(self):
        return yaml.dump(self.to_dict(), Dumper=dumper)


class ExtendedConfigParser(ConfigParser):
    """Class helper to read/write information to the config file."""

    def __init__(
        self, uri: Optional[str] = DEFAULT_CONFIG_PATH, auth: Optional[Dict] = None
    ):
        """Initializes a config parser instance.

        Parameters
        ----------
        uri: (str, optional). Defaults to `~/.ads/config`.
            The path to the config file. Can be local or Object Storage file.
        auth: (Dict, optional). Defaults to None.
            The default authentication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        """
        super().__init__(default_section="EXCLUDE_DEFAULT_SECTION")
        self.auth = auth or authutil.default_signer()
        self.uri = uri

    def save(
        self,
        uri: Optional[str] = None,
        auth: Optional[Dict] = None,
        force_overwrite: Optional[bool] = False,
    ) -> None:
        """Saves the config to the file.

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
        None

        Raise
        -----
        FileExistsError
            In case if file exists and force_overwrite is false.
        """
        uri = uri or self.uri
        auth = auth or self.auth or authutil.default_signer()

        if not force_overwrite:
            dst_path_scheme = urlparse(uri).scheme or "file"
            if fsspec.filesystem(dst_path_scheme, **auth).exists(uri):
                raise FileExistsError(
                    f"The `{uri}` exists. Set `force_overwrite` to True "
                    "if you wish to overwrite."
                )

        with fsspec.open(uri, mode="w", **auth) as f:
            self.write(f)

    def to_dict(self) -> Dict[str, Any]:
        """Converts config to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Config in a dictionary format.
        """
        return {s: dict(self[s]) for s in self.keys() if self[s]}

    def read(
        self, uri: Optional[str] = None, auth: Optional[Dict] = None
    ) -> "ExtendedConfigParser":
        """Reads config file.

        uri: (str, optional). Defaults to `~/.ads/config`.
            The path to the config file. Can be local or Object Storage file.
        auth: (Dict, optional). Defaults to None.
            The default authentication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        ExtendedConfigParser
           Config parser object.
        """
        uri = uri or self.uri
        auth = auth or self.auth or authutil.default_signer()

        with fsspec.open(uri, "r", **auth) as f:
            self.read_string(f.read())
        return self

    def with_dict(self, info: Dict[str, Dict[str, Any]]) -> "ExtendedConfigParser":
        """Populates config with values from a dictionary.

        Parameters
        ----------
        info: Dict[str, Dict[str, Any]]
            Config in a dictionary format.

        Returns
        -------
        ExtendedConfigParser
           Config parser object.
        """
        self.read_dict(info)
        return self
