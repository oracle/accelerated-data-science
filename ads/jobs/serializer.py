#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import json
from abc import ABC, abstractmethod
from typing import Union, TypeVar, Type

import fsspec
import yaml


Self = TypeVar("Self", bound="Serializable")
"""Special type to represent the current enclosed class.

This type is used by factory class method or when a method returns ``self``.
"""


class Serializable(ABC):
    """Base class that represents a serializable item."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serializes an instance of class into a dictionary"""

    @classmethod
    @abstractmethod
    def from_dict(cls, obj_dict: dict):
        """Initialize an instance of the class from a dictionary

        Parameters
        ----------
        obj_dict : dict
            Dictionary representation of the object
        """

    @staticmethod
    def _write_to_file(s: str, uri: str, **kwargs) -> None:
        """Write string content into a file specified by uri

        Parameters
        ----------
        s : str
            The content to be written.
        uri : str
            URI of the file to save the content.
        kwargs : dict
            keyword arguments to be passed into fsspec.open().
            For OCI object storage, this can be config="path/to/.oci/config".
        """
        with fsspec.open(uri, "w", **kwargs) as f:
            f.write(s)

    @staticmethod
    def _read_from_file(uri: str, **kwargs) -> str:
        """Returns contents from a file specified by URI

        Parameters
        ----------
        uri : str
            The URI of the file.

        Returns
        -------
        str
            The content of the file as a string.
        """
        with fsspec.open(uri, "r", **kwargs) as f:
            return f.read()

    def to_json(
        self, uri: str = None, encoder: callable = json.JSONEncoder, **kwargs
    ) -> str:
        """Returns the object serialized as a JSON string

        Parameters
        ----------
        uri : str, optional
            URI location to save the JSON string, by default None
        encoder : callable, optional
            Encoder for custom data structures, by default json.JSONEncoder
        kwargs : dict
            keyword arguments to be passed into fsspec.open().
            For OCI object storage, this can be config="path/to/.oci/config".

        Returns
        -------
        str
            object serialized as a JSON string
        """
        json_string = json.dumps(self.to_dict(), cls=encoder)
        if uri:
            self._write_to_file(s=json_string, uri=uri, **kwargs)
        return json_string

    @classmethod
    def from_json(
        cls: Type[Self],
        json_string: str = None,
        uri: str = None,
        decoder: callable = json.JSONDecoder,
        **kwargs
    ) -> Self:
        """Creates an object from JSON string provided or from URI location containing JSON string

        Args:
            json_string (string, optional): JSON string. Defaults to None.
            uri (string, optional): URI location of file containing JSON string. Defaults to None.
            decoder (callable, optional): Custom decoder. Defaults to simple JSON decoder.
            kwargs (dict): keyword arguments to be passed into fsspec.open(). For OCI object storage, this should be config="path/to/.oci/config".
                           For other storage connections consider e.g. host, port, username, password, etc.

        Raises:
            ValueError: Raised if neither string nor uri is provided

        Returns:
            cls: Returns instance of the class

        Parameters
        ----------
        json_string : str, optional
            JSON string, by default None
        uri : str, optional
            URI location of file containing JSON string, by default None
        decoder : callable, optional
            Decoder for custom data structures, by default json.JSONDecoder
        kwargs : dict
            keyword arguments to be passed into fsspec.open().
            For OCI object storage, this can be config="path/to/.oci/config".

        Returns
        -------
        Type[Self]
            Object initialized from JSON data.

        Raises
        ------
        ValueError
            Both json_string and uri are empty, or
            The input is not a valid JSON.
        """
        if json_string:
            return cls.from_dict(json.loads(json_string, cls=decoder))
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
            keyword arguments to be passed into fsspec.open().
            For OCI object storage, this can be config="path/to/.oci/config".

        Returns
        -------
        Union[str, None]
            If uri is specified, None will be returned.
            Otherwise, the yaml content will be returned.
        """
        yaml_string = yaml.dump(self.to_dict(), Dumper=dumper)
        if uri:
            self._write_to_file(s=yaml_string, uri=uri, **kwargs)
            return None

        return yaml_string

    @classmethod
    def from_yaml(
        cls: Type[Self],
        yaml_string: str = None,
        uri: str = None,
        loader: callable = yaml.SafeLoader,
        **kwargs
    ) -> Self:
        """Initializes an object from YAML string or URI location containing the YAML

        Parameters
        ----------
        yaml_string : str, optional
            YAML string, by default None
        uri : str, optional
            URI location of file containing YAML, by default None
        loader : callable, optional
            Custom YAML loader, by default yaml.SafeLoader

        Returns
        -------
        Self
            Object initialized from the YAML.

        Raises
        ------
        ValueError
            Raised if neither string nor uri is provided
        """
        if yaml_string:
            return cls.from_dict(yaml.load(yaml_string, Loader=loader))
        if uri:
            yaml_dict = yaml.load(cls._read_from_file(uri=uri, **kwargs), Loader=loader)
            return cls.from_dict(yaml_dict)
        raise ValueError("Must provide either YAML string or URI location")

    @classmethod
    def from_string(
        cls: Type[Self],
        obj_string: str = None,
        uri: str = None,
        loader: callable = yaml.SafeLoader,
        **kwargs
    ) -> Self:
        """Initializes an object from YAML/JSON string or URI location containing the YAML/JSON

        Parameters
        ----------
        obj_string : str, optional
            YAML/JSON string, by default None
        uri : str, optional
            URI location of file containing YAML/JSON, by default None
        loader : callable, optional
            Custom YAML loader, by default yaml.SafeLoader

        Returns
        -------
        Self
            Object initialized from the YAML.
        """
        return cls.from_yaml(yaml_string=obj_string, uri=uri, loader=loader, **kwargs)

    def __repr__(self) -> str:
        """Returns printable version of object

        Returns:
            string: Serialized version of object as a YAML string
        """
        return self.to_yaml()
