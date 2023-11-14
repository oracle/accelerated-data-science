#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import json
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Union, Dict
from urllib.parse import urlparse

import fsspec
import yaml
from ads.common.auth import default_signer

# Special type to represent the current enclosed class.
# This type is used by factory class method or when a method returns ``self``.
Self = TypeVar("Self", bound="Serializable")


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

        overwrite = kwargs.pop("overwrite", True)
        if not overwrite:
            dst_path_scheme = urlparse(uri).scheme or "file"
            if fsspec.filesystem(dst_path_scheme, **kwargs).exists(uri):
                raise FileExistsError(
                    f"The `{uri}` is already exists. Set `overwrite` to True "
                    "if you wish to overwrite."
                )

        # Add default signer if the uri is an object storage uri, and
        # the user does not specify config or signer.
        if (
            uri.startswith("oci://")
            and "config" not in kwargs
            and "signer" not in kwargs
        ):
            kwargs.update(default_signer())
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
        json_string = json.dumps(self.to_dict(**kwargs), cls=encoder)
        if uri:
            self._write_to_file(s=json_string, uri=uri, **kwargs)
            return None
        return json_string

    @classmethod
    def from_json(
        cls: Type[Self],
        json_string: str = None,
        uri: str = None,
        decoder: callable = json.JSONDecoder,
        **kwargs,
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
        note = kwargs.pop("note", "")

        yaml_string = f"{note}\n" + yaml.dump(self.to_dict(**kwargs), Dumper=dumper)
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
        **kwargs,
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
        **kwargs,
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
