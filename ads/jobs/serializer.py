#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Union, Dict
from urllib.parse import urlparse

import fsspec
import json
import yaml

Self = TypeVar("Self", bound="Serializable")
"""Special type to represent the current enclosed class.

This type is used by factory class method or when a method returns ``self``.
"""


class Serializable(ABC):
    """Base class that represents a serializable item."""

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
    def from_dict(cls, obj_dict: dict):
        """Returns an instance of the class instantiated by the dictionary provided

        Args:
            obj_dict (dict): Dictionary representation of the object
        """
        pass

    @staticmethod
    def _write_to_file(s: str, uri: str, **kwargs) -> None:
        """Write string s into location specified by uri

        Args:
            s (string): content
            uri (string): URI location to save string s
            kwargs (dict): keyword arguments to be passed into fsspec.open(). For OCI object storage, this should be config="path/to/.oci/config".
                           For other storage connections consider e.g. host, port, username, password, etc.
        """
        with fsspec.open(uri, "w", **kwargs) as f:
            f.write(s)

    @staticmethod
    def _read_from_file(uri: str, **kwargs) -> str:
        """Returns contents from location specified by URI

        Args:
            uri (string): URI location
            kwargs (dict): keyword arguments to be passed into fsspec.open(). For OCI object storage, this should be config="path/to/.oci/config".
                           For other storage connections consider e.g. host, port, username, password, etc.

        Returns:
            string: Contents in file specified by URI
        """
        with fsspec.open(uri, "r", **kwargs) as f:
            return f.read()

    def to_json(
        self, uri: str = None, encoder: callable = json.JSONEncoder, **kwargs
    ) -> str:
        """Returns object serialized as a JSON string

        Parameters
        ----------
        uri : str, optional
            URI location to save the JSON string, by default None
        encoder : callable, optional
            Encoder for custom data structures, by default json.JSONEncoder
        kwargs : dict
            overwrite: (bool, optional). Defaults to True.
                Whether to overwrite existing file or not.

            The other keyword arguments to be passed into fsspec.open().
            For OCI object storage, this could be config="path/to/.oci/config".

        Returns
        -------
        Union[str, None]
            Serialized version of object.
            `None` in case when `uri` provided.
        """
        json_string = json.dumps(self.to_dict(), cls=encoder)
        if uri:
            self._write_to_file(s=json_string, uri=uri, **kwargs)
        return json_string

    @classmethod
    def from_json(
        cls,
        json_string: str = None,
        uri: str = None,
        decoder: callable = json.JSONDecoder,
        **kwargs
    ):
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
        """
        if json_string:
            return cls.from_dict(json.loads(json_string, cls=decoder))
        if uri:
            json_dict = json.loads(cls._read_from_file(uri, **kwargs), cls=decoder)
            return cls.from_dict(json_dict)
        raise ValueError("Must provide either JSON string or URI location")

    def to_yaml(self, uri: str = None, dumper: callable = dumper, **kwargs) -> str:
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

            The other keyword arguments to be passed into fsspec.open().
            For OCI object storage, this could be config="path/to/.oci/config".

        Returns
        -------
        Union[str, None]
            Serialized version of object.
            `None` in case when `uri` provided.
        """
        yaml_string = yaml.dump(self.to_dict(), Dumper=dumper)
        if uri:
            self._write_to_file(s=yaml_string, uri=uri, **kwargs)
        return yaml_string

    @classmethod
    def from_yaml(
        cls,
        yaml_string: str = None,
        uri: str = None,
        loader: callable = loader,
        **kwargs
    ):
        """Creates an object from YAML string provided or from URI location containing YAML string

        Args:
            yaml_string (string, optional): YAML string. Defaults to None.
            uri (string, optional): URI location of file containing YAML string. Defaults to None.
            loader (callable, optional): Custom YAML loader. Defaults to CLoader/SafeLoader.
            kwargs (dict): keyword arguments to be passed into fsspec.open(). For OCI object storage, this should be config="path/to/.oci/config".
                           For other storage connections consider e.g. host, port, username, password, etc.

        Raises:
            ValueError: Raised if neither string nor uri is provided

        Returns:
            cls: Returns instance of the class
        """
        if yaml_string:
            return cls.from_dict(yaml.load(yaml_string, Loader=loader))
        if uri:
            yaml_dict = yaml.load(cls._read_from_file(uri=uri, **kwargs), Loader=loader)
            return cls.from_dict(yaml_dict)
        raise ValueError("Must provide either YAML string or URI location")

    @classmethod
    def from_string(
        cls,
        obj_string: str = None,
        uri: str = None,
        loader: callable = loader,
        **kwargs
    ):
        """Creates an object from string provided or from URI location containing string

        Args:
            obj_string (str, optional): String representing the object
            uri (str, optional): URI location of file containing string. Defaults to None.
            loader (callable, optional): Custom YAML loader. Defaults to CLoader/SafeLoader.
            kwargs (dict): keyword arguments to be passed into fsspec.open(). For OCI object storage, this should be config="path/to/.oci/config".
                           For other storage connections consider e.g. host, port, username, password, etc.

        Returns:
            cls: Returns instance of the class
        """
        return cls.from_yaml(yaml_string=obj_string, uri=uri, loader=loader, **kwargs)

    def __repr__(self):
        """Returns printable version of object

        Returns:
            string: Serialized version of object as a YAML string
        """
        return self.to_yaml()
