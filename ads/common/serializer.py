#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
This module provides a base class for serializable items, as well as methods for serializing and
deserializing objects to and from JSON and YAML formats. It also includes methods for reading and
writing serialized objects to and from files.
"""

import dataclasses
import json
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Union
from urllib.parse import urlparse

import fsspec
import yaml

from ads.common import logger
from ads.common.auth import default_signer

try:
    from yaml import CSafeDumper as dumper
    from yaml import CSafeLoader as loader
except:
    from yaml import SafeDumper as dumper
    from yaml import SafeLoader as loader


class SideEffect(Enum):
    CONVERT_KEYS_TO_LOWER = "lower"
    CONVERT_KEYS_TO_UPPER = "upper"


class Serializable(ABC):
    """Base class that represents a serializable item.

    Methods
    -------
    to_dict(self) -> dict
        Serializes the object into a dictionary.
    from_dict(cls, obj_dict) -> cls
        Returns an instance of the class instantiated from the dictionary provided.
    _write_to_file(s, uri, **kwargs)
        Write string s into location specified by uri
    _read_from_file(uri, **kwargs)
        Returns contents from location specified by URI
    to_json(self, uri=None, **kwargs)
        Returns object serialized as a JSON string
    from_json(cls, json_string=None, uri=None, **kwargs)
        Creates an object from JSON string provided or from URI location containing JSON string
    to_yaml(self, uri=None, **kwargs)
        Returns object serialized as a YAML string
    from_yaml(cls, yaml_string=None, uri=None, **kwargs)
        Creates an object from YAML string provided or from URI location containing YAML string
    from_string(cls, obj_string=None: str, uri=None, **kwargs)
        Creates an object from string provided or from URI location containing string
    """

    @abstractmethod
    def to_dict(self, **kwargs: Dict) -> Dict:
        """Serializes an instance of class into a dictionary.

        Parameters
        ----------
        **kwargs: Dict
            Additional arguments.

        Returns
        -------
        Dict
            The result dictionary.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, obj_dict: dict, **kwargs) -> "Serializable":
        """Returns an instance of the class instantiated by the dictionary provided.

        Parameters
        ----------
        obj_dict: (dict)
            Dictionary representation of the object.

        Returns
        -------
        Serializable
            A Serializable instance.
        """
        pass

    @staticmethod
    def serialize(obj):
        """JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable.")

    @staticmethod
    def _write_to_file(s: str, uri: str, **kwargs) -> None:
        """Write string s into location specified by uri.

        Parameters
        ----------
        s: (string)
            content
        uri: (string)
            URI location to save string s
        kwargs : dict
            keyword arguments to be passed into fsspec.open().
            For OCI object storage, this can be config="path/to/.oci/config".

        Returns
        -------
        None
            Nothing
        """

        overwrite = kwargs.pop("overwrite", True)
        if not overwrite:
            dst_path_scheme = urlparse(uri).scheme or "file"
            if fsspec.filesystem(dst_path_scheme, **kwargs).exists(uri):
                raise FileExistsError(
                    f"The `{uri}` is already exists. Set `overwrite` to True "
                    "if you wish to overwrite."
                )

        with fsspec.open(uri, "w", **kwargs) as f:
            f.write(s)

    @staticmethod
    def _read_from_file(uri: str, **kwargs) -> str:
        """Returns contents from location specified by URI

        Parameters
        ----------
        uri: (string)
            URI location
        kwargs : dict
            keyword arguments to be passed into fsspec.open().
            For OCI object storage, this can be config="path/to/.oci/config".

        Returns
        -------
        string: Contents in file specified by URI
        """
        # Add default signer if the uri is an object storage uri, and
        # the user does not specify config or signer.
        if (
            uri.startswith("oci://")
            and "config" not in kwargs
            and "signer" not in kwargs
        ):
            kwargs.update(default_signer())
        with fsspec.open(uri, "r", **kwargs) as f:
            return f.read()

    def to_json(
        self,
        uri: str = None,
        encoder: callable = json.JSONEncoder,
        default: callable = None,
        **kwargs,
    ) -> str:
        """Returns object serialized as a JSON string

        Parameters
        ----------
        uri: (string, optional)
            URI location to save the JSON string. Defaults to None.
        encoder: (callable, optional)
            Encoder for custom data structures. Defaults to JSONEncoder.
        default: (callable, optional)
            A function that gets called for objects that can't otherwise be serialized.
            It should return JSON-serializable version of the object or original object.

        kwargs
        ------
        overwrite: (bool, optional). Defaults to True.
            Whether to overwrite existing file or not.

        keyword arguments to be passed into fsspec.open().
        For OCI object storage, this could be config="path/to/.oci/config".
        For other storage connections consider e.g. host, port, username, password, etc.

        Returns
        -------
        Union[str, None]
            Serialized version of object.
            `None` in case when `uri` provided.
        """
        json_string = json.dumps(
            self.to_dict(**kwargs),
            cls=encoder,
            default=default or self.serialize,
            indent=4,
        )
        if uri:
            self._write_to_file(s=json_string, uri=uri, **kwargs)
            return None
        return json_string

    @classmethod
    def from_json(
        cls,
        json_string: str = None,
        uri: str = None,
        decoder: callable = json.JSONDecoder,
        **kwargs,
    ):
        """Creates an object from JSON string provided or from URI location containing JSON string

        Parameters
        ----------
        json_string: (string, optional)
            JSON string. Defaults to None.
        uri: (string, optional)
            URI location of file containing JSON string. Defaults to None.
        decoder: (callable, optional)
            Custom decoder. Defaults to simple JSON decoder.
        kwargs
        ------
        keyword arguments to be passed into fsspec.open(). For OCI object storage, this should be config="path/to/.oci/config".
            For other storage connections consider e.g. host, port, username, password, etc.

        Raises
        ------
        ValueError
            Raised if neither string nor uri is provided

        Returns
        -------
        cls
            Returns instance of the class
        """
        if json_string:
            return cls.from_dict(json.loads(json_string, cls=decoder), **kwargs)
        if uri:
            json_dict = json.loads(cls._read_from_file(uri, **kwargs), cls=decoder)
            return cls.from_dict(json_dict)
        raise ValueError("Must provide either JSON string or URI location")

    def to_yaml(
        self, uri: str = None, dumper: callable = yaml.SafeDumper, **kwargs
    ) -> Union[str, None]:
        """Returns object serialized as a YAML string

        Parameters
        ----------
        uri : str, optional
            URI location to save the YAML string, by default None
        dumper : callable, optional
            Custom YAML Dumper, by default yaml.SafeDumper
        kwargs : dict
            overwrite: (bool, optional). Defaults to True.
                Whether to overwrite existing file or not.
            note: (str, optional)
                The note that needs to be added in the beginning of the YAML.
                It will be added as is without any formatting.
            side_effect: Optional[SideEffect]
                side effect to take on the dictionary. The side effect can be either
                convert the dictionary keys to "lower" (SideEffect.CONVERT_KEYS_TO_LOWER.value)
                or "upper"(SideEffect.CONVERT_KEYS_TO_UPPER.value) cases.

            The other keyword arguments to be passed into fsspec.open().
            For OCI object storage, this could be config="path/to/.oci/config".

        Returns
        -------
        Union[str, None]
            Serialized version of object.
            `None` in case when `uri` provided.
        """
        note = kwargs.pop("note", "")

        yaml_string = f"{note}\n" + yaml.dump(self.to_dict(**kwargs), Dumper=dumper)
        if uri:
            self._write_to_file(s=yaml_string, uri=uri, **kwargs)
            return None

        return yaml_string

    @classmethod
    def from_yaml(
        cls,
        yaml_string: str = None,
        uri: str = None,
        loader: callable = loader,
        **kwargs,
    ):
        """Creates an object from YAML string provided or from URI location containing YAML string

        Parameters
        ----------
        yaml_string (string, optional)
            YAML string. Defaults to None.
        uri (string, optional)
            URI location of file containing YAML string. Defaults to None.
        loader (callable, optional)
            Custom YAML loader. Defaults to CLoader/SafeLoader.
        kwargs (dict)
            keyword arguments to be passed into fsspec.open().
            For OCI object storage, this should be config="path/to/.oci/config".
            For other storage connections consider e.g. host, port, username, password, etc.

        Raises
        ------
        ValueError
            Raised if neither string nor uri is provided

        Returns
        -------
        cls
            Returns instance of the class
        """
        if yaml_string:
            return cls.from_dict(yaml.load(yaml_string, Loader=loader), **kwargs)
        if uri:
            yaml_dict = yaml.load(cls._read_from_file(uri=uri, **kwargs), Loader=loader)
            return cls.from_dict(yaml_dict, **kwargs)
        raise ValueError("Must provide either YAML string or URI location")

    @classmethod
    def from_string(
        cls,
        obj_string: str = None,
        uri: str = None,
        loader: callable = loader,
        **kwargs,
    ) -> "Serializable":
        """Creates an object from string provided or from URI location containing string

        Parameters
        ----------
        obj_string: (str, optional)
            String representing the object
        uri: (str, optional)
            URI location of file containing string. Defaults to None.
        loader: (callable, optional)
            Custom YAML loader. Defaults to CLoader/SafeLoader.
        kwargs: (dict)
            keyword arguments to be passed into fsspec.open(). For OCI object storage, this should be config="path/to/.oci/config".
            For other storage connections consider e.g. host, port, username, password, etc.

        Returns
        -------
        Serializable
            A Serializable instance
        """
        return cls.from_yaml(yaml_string=obj_string, uri=uri, loader=loader, **kwargs)

    def __repr__(self):
        """Returns printable version of object.

        Returns
        ----------
        string
            Serialized version of object as a YAML string
        """
        return self.to_yaml()

    def __str__(self):
        """Returns printable version of object.

        Returns
        ----------
        string
            Serialized version of object as a YAML string
        """
        return self.to_json()


class DataClassSerializable(Serializable):
    """Wrapper class that inherit from Serializable class.

    Methods
    -------
    to_dict(self) -> dict
        Serializes the object into a dictionary.
    from_dict(cls, obj_dict) -> cls
        Returns an instance of the class instantiated from the dictionary provided.
    """

    @classmethod
    def _validate_dict(cls, obj_dict: Dict) -> bool:
        """validate the dictionary.

        Parameters
        ----------
        obj_dict: (dict)
            Dictionary representation of the object

        Returns
        -------
        bool
            True if the validation passed, else False.
        """
        pass

    def to_dict(self, **kwargs) -> Dict:
        """Serializes instance of class into a dictionary

        kwargs
        ------
        side_effect: Optional[SideEffect]
            side effect to take on the dictionary. The side effect can be either
            convert the dictionary keys to "lower" (SideEffect.CONVERT_KEYS_TO_LOWER.value)
            or "upper"(SideEffect.CONVERT_KEYS_TO_UPPER.value) cases.

        Returns
        -------
        Dict
            A dictionary.
        """
        obj_dict = dataclasses.asdict(self)
        if "side_effect" in kwargs and kwargs["side_effect"]:
            obj_dict = DataClassSerializable._normalize_dict(
                obj_dict=obj_dict, case=kwargs["side_effect"], recursively=True
            )
        return obj_dict

    @classmethod
    def from_dict(
        cls,
        obj_dict: dict,
        side_effect: Optional[SideEffect] = SideEffect.CONVERT_KEYS_TO_LOWER.value,
        ignore_unknown: Optional[bool] = False,
        **kwargs,
    ) -> "DataClassSerializable":
        """Returns an instance of the class instantiated by the dictionary provided.

        Parameters
        ----------
        obj_dict: (dict)
            Dictionary representation of the object
        side_effect: Optional[SideEffect]
            side effect to take on the dictionary. The side effect can be either
            convert the dictionary keys to "lower" (SideEffect.CONVERT_KEYS_TO_LOWER.value)
            or "upper"(SideEffect.CONVERT_KEYS_TO_UPPER.value) cases.
        ignore_unknown: (bool, optional). Defaults to `False`.
            Whether to ignore unknown fields or not.

        Returns
        -------
        DataClassSerializable
            A DataClassSerializable instance.
        """
        assert obj_dict, "`obj_dict` must not be None."
        if not isinstance(obj_dict, dict):
            raise TypeError("`obj_dict` must be a dictionary.")
        # check if dict not is None and not empty and type is dict
        cls._validate_dict(obj_dict=obj_dict)
        if side_effect:
            obj_dict = cls._normalize_dict(obj_dict, case=side_effect)

        allowed_fields = set([f.name for f in dataclasses.fields(cls)])
        wrong_fields = set(obj_dict.keys()) - allowed_fields
        if wrong_fields and not ignore_unknown:
            logger.warning(
                f"The class {cls.__name__} doesn't contain attributes: `{list(wrong_fields)}`. "
                "These fields will be ignored."
            )

        obj = cls(**{key: obj_dict.get(key) for key in allowed_fields})

        for key, value in obj_dict.items():
            if (
                key in allowed_fields
                and isinstance(value, dict)
                and hasattr(getattr(cls(), key).__class__, "from_dict")
            ):
                attribute = getattr(cls(), key).__class__.from_dict(
                    value,
                    ignore_unknown=ignore_unknown,
                    side_effect=side_effect,
                    **kwargs,
                )
                setattr(obj, key, attribute)

        return obj

    @staticmethod
    def _normalize_dict(
        obj_dict: Dict,
        recursively: bool = False,
        case: str = SideEffect.CONVERT_KEYS_TO_LOWER.value,
        **kwargs,
    ) -> Dict:
        """lower all the keys.

        Parameters
        ----------
        obj_dict: (Dict)
            Dictionary representation of the object.
        case: (optional, str). Defaults to "lower".
            the case to normalized to. can be either "lower" (SideEffect.CONVERT_KEYS_TO_LOWER.value)
            or "upper"(SideEffect.CONVERT_KEYS_TO_UPPER.value).
        recursively: (bool, optional). Defaults to `False`.
            Whether to recursively normalize the dictionary or not.

        Returns
        -------
        Dict
            Dictionary representation of the object.
        """
        normalized_obj_dict = {}
        for key, value in obj_dict.items():
            if recursively and isinstance(value, dict):
                value = DataClassSerializable._normalize_dict(
                    value, case=case, recursively=recursively, **kwargs
                )
            normalized_obj_dict = DataClassSerializable._normalize_key(
                normalized_obj_dict=normalized_obj_dict,
                key=key,
                value=value,
                case=case,
                **kwargs,
            )
        return normalized_obj_dict

    @staticmethod
    def _normalize_key(
        normalized_obj_dict: Dict, key: str, value: Union[str, Dict], case: str
    ) -> Dict:
        """helper function to normalize the key in the case specified and add it back to the dictionary.

        Parameters
        ----------
        normalized_obj_dict: (Dict)
            the dictionary to append the key and value to.
        key: (str)
            key to be normalized.
        value: (Union[str, Dict])
            value to be added.
        case: (str)
            The case to normalized to. can be either "lower" (SideEffect.CONVERT_KEYS_TO_LOWER.value)
            or "upper"(SideEffect.CONVERT_KEYS_TO_UPPER.value).

        Raises
        ------
        NotImplementedError
            Raised when `case` is not supported.

        Returns
        -------
        Dict
            Normalized dictionary with the key and value added in the case specified.
        """
        if case.lower() == SideEffect.CONVERT_KEYS_TO_LOWER.value:
            normalized_obj_dict[key.lower()] = value
        elif case.lower() == SideEffect.CONVERT_KEYS_TO_UPPER.value:
            normalized_obj_dict[key.upper()] = value
        else:
            raise NotImplementedError("`case` is not supported.")
        return normalized_obj_dict
