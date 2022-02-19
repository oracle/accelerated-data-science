#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from abc import ABC, abstractmethod
from ads.jobs.schema.validator import ValidatorFactory
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Union

import fsspec
import json
import yaml

try:
    from yaml import CSafeDumper as dumper
    from yaml import CSafeLoader as loader
except:
    from yaml import SafeDumper as dumper
    from yaml import SafeLoader as loader


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
    def to_dict(self) -> dict:
        """Serializes instance of class into a dictionary"""
        pass

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

        Args:
            uri (string, optional): URI location to save the JSON string. Defaults to None.
            encoder (callable, optional): Encoder for custom data structures. Defaults to JSONEncoder.
            kwargs (dict): keyword arguments to be passed into fsspec.open(). For OCI object storage, this should be config="path/to/.oci/config".
                           For other storage connections consider e.g. host, port, username, password, etc.

        Returns:
            string: Serialized version of object
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

        Args:
            uri (string, optional): URI location to save the YAML string. Defaults to None.
            dumper (callable, optional): Custom YAML Dumper. Defaults to CDumper/SafeDumper.
            kwargs (dict): keyword arguments to be passed into fsspec.open(). For OCI object storage, this should be config="path/to/.oci/config".
                           For other storage connections consider e.g. host, port, username, password, etc.

        Returns:
            string: Serialized version of object
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
