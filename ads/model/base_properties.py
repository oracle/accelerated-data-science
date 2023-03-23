#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import dataclasses
import json
import os
from typing import Any, Dict, Optional, Union

from ads.common.config import Config, ConfigSection
from ads.common.serializer import Serializable


@dataclasses.dataclass(repr=False)
class BaseProperties(Serializable):
    """Represents base properties class.

    Methods
    -------
    with_prop(name: str, value: Any) -> BaseProperties
        Sets property value.
    with_dict(obj_dict: Dict) -> BaseProperties
        Populates properties values from dict.
    with_env() -> BaseProperties
        Populates properties values from environment variables.
    to_dict() -> Dict
        Serializes instance of class into a dictionary.
    with_config(config: ads.config.ConfigSection) -> BaseProperties
        Sets properties values from the config profile.
    from_dict(obj_dict: Dict[str, Any]) -> "BaseProperties"
        Creates an instance of the properties class from a dictionary.
    from_config(uri: str, profile: str, auth: Optional[Dict] = None) -> "BaseProperties":
        Loads properties from the config file.
    to_config(uri: str, profile: str, force_overwrite: Optional[bool] = False, auth: Optional[Dict] = None) -> None
        Saves properties to the config file.
    """

    def __setattr__(self, name: str, value: Any):
        """Adds type validation when attribute assignment is attempted."""
        if value is None:
            self.__dict__[name] = value
        elif name in self.__annotations__:
            if hasattr(self.__annotations__[name], "__origin__"):
                if self.__annotations__[name].__origin__ is Union and not isinstance(
                    value, self.__annotations__[name].__args__
                ):
                    raise TypeError(
                        f"Field `{name}` was expected to be of type `{self.__annotations__[name].__args__}` "
                        f"but type `{type(value)}` was provided."
                    )

            elif self.__annotations__[name] != type(value):
                raise TypeError(
                    f"Field `{name}` was expected to be of type `{self.__annotations__[name]}` "
                    f"but type `{type(value)}` was provided."
                )

            self.__dict__[name] = value

    def with_prop(self, name: str, value: Any) -> "BaseProperties":
        """Sets property value.

        Parameters
        ----------
        name: str
            Property name.
        value:
            Property value.

        Returns
        -------
        BaseProperties
            Instance of the BaseProperties.
        """
        setattr(self, name, value)
        return self

    def with_dict(self, obj_dict: Dict[str, Any]) -> "BaseProperties":
        """Sets properties from a dict.

        Parameters
        ----------
        obj_dict: Dict[str, Any]
            List of properties and values in dictionary format.

        Returns
        -------
        BaseProperties
            Instance of the BaseProperties.

        Raises
        ------
        TypeError
            If input object has a wrong type.
        """
        if obj_dict is None:
            return self

        if not isinstance(obj_dict, Dict):
            raise TypeError("The `obj_dict` should be a dictionary.")

        for key, value in obj_dict.items():
            # if expected type of input value is not a string, but
            # actual type of the value is string, then try to convert value to the
            # expected format by using JSON.loads()
            if (
                not value is None
                and key in self.__annotations__
                and self.__annotations__[key] != str
                and isinstance(value, str)
            ):
                try:
                    value = json.loads(value)
                except:
                    pass

            self.with_prop(key, value)
        return self

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Any]) -> "BaseProperties":
        """Creates an instance of the properties class from a dictionary.

        Parameters
        ----------
        obj_dict: Dict[str, Any]
            List of properties and values in dictionary format.

        Returns
        -------
        BaseProperties
            Instance of the BaseProperties.
        """
        return cls().with_dict(obj_dict)

    def with_env(self) -> "BaseProperties":
        """Sets properties values from environment variables.

        Returns
        -------
        BaseProperties
            Instance of the BaseProperties.
        """
        self.with_dict(
            {env_k.lower(): env_v for env_k, env_v in sorted(os.environ.items())}
        )
        self._adjust_with_env()
        return self

    def with_config(self, config: ConfigSection) -> "BaseProperties":
        """Sets properties values from the config profile.

        Returns
        -------
        BaseProperties
            Instance of the BaseProperties.
        """
        if not isinstance(config, ConfigSection):
            raise TypeError("`config` should be an instance of `ConfigSection`.")

        return self.with_dict(config.to_dict())

    def to_dict(self, **kwargs):
        """Serializes instance of class into a dictionary.

        Returns
        -------
        Dict
            A dictionary.
        """
        return dataclasses.asdict(self)

    def to_config(
        self,
        uri: str,
        profile: str,
        force_overwrite: Optional[bool] = False,
        auth: Optional[Dict] = None,
    ) -> None:
        """Saves properties to the config file.

        Parameters
        ----------
        uri: str
            The URI of the config file.
            Can be local path or OCI object storage URI.
        profile: str
            The config profile name.
        force_overwrite: (bool, optional). Defaults to False.
            Whether to overwrite existing files or not.
        auth: (Dict, optional). Defaults to None.
            The default authentication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        None
            Nothing
        """
        config = Config(uri=uri, auth=auth)
        config.section_set(key=profile, info=self.to_dict(), replace=force_overwrite)
        config.save(force_overwrite=force_overwrite)

    @classmethod
    def from_config(
        cls,
        uri: str,
        profile: str,
        auth: Optional[Dict] = None,
    ) -> "BaseProperties":
        """Loads properties from the config file.

        Parameters
        ----------
        uri: str
            The URI of the config file.
            Can be local path or OCI object storage URI.
        profile: str
            The config profile name.
        auth: (Dict, optional). Defaults to None.
            The default authentication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        BaseProperties
            Instance of the BaseProperties.
        """
        config = Config(uri=uri, auth=auth).load()
        return cls().with_config(config[profile])

    def _adjust_with_env(self) -> None:
        """Adjusts env variables. The method is used in `with_env` method."""
        pass
