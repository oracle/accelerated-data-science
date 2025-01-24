#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from typing import Union
from urllib.parse import urlparse

import fsspec
import yaml
from pydantic import BaseModel
from yaml import SafeLoader as Loader

from ads.common.auth import default_signer


class Serializable(BaseModel):
    """Base class that represents a serializable item.

    Methods
    -------
    to_json(self, uri=None, **kwargs)
        Returns object serialized as a JSON string
    from_json(cls, json_string=None, uri=None, **kwargs)
        Creates an object from JSON string provided or from URI location containing JSON string
    to_yaml(self, uri=None, **kwargs)
        Returns object serialized as a YAML string
    from_yaml(cls, yaml_string=None, uri=None, **kwargs)
        Creates an object from YAML string provided or from URI location containing YAML string
    """

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
            self.model_dump(exclude_none=kwargs.pop("exclude_none", False)),
            cls=encoder,
            default=default,
        )
        if uri:
            self._write_to_file(s=json_string, uri=uri, **kwargs)
            return None
        return json_string

    def to_dict(self) -> dict:
        """Returns object serialized as a dictionary

        Returns
        -------
        dict
            Serialized version of object
        """
        return json.loads(self.to_json())

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
            return cls(**json.loads(json_string, cls=decoder))
        if uri:
            return cls(**json.loads(cls._read_from_file(uri, **kwargs), cls=decoder))
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

        yaml_string = f"{note}\n" + yaml.dump(
            self.model_dump(exclude_none=kwargs.pop("exclude_none", False)),
            Dumper=dumper,
        )
        if uri:
            self._write_to_file(s=yaml_string, uri=uri, **kwargs)
            return None

        return yaml_string

    @classmethod
    def from_yaml(
        cls,
        yaml_string: str = None,
        uri: str = None,
        loader: callable = Loader,
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
            return cls(**yaml.load(yaml_string, Loader=loader))
        if uri:
            return cls(
                **yaml.load(cls._read_from_file(uri=uri, **kwargs), Loader=loader)
            )
        raise ValueError("Must provide either YAML string or URI location")

    @classmethod
    def schema_to_yaml(cls, uri: str = None, **kwargs) -> Union[str, None]:
        """Returns the schema serialized as a YAML string

        Parameters
        ----------
        uri : str, optional
            URI location to save the YAML string, by default None
        dumper : callable, optional
            Custom YAML Dumper, by default yaml.SafeDumper
        kwargs : dict
            overwrite: (bool, optional). Defaults to True.
                Whether to overwrite existing file or not.
        Returns
        -------
        Union[str, None]
            Serialized schema.
            `None` in case when `uri` provided.
        """
        yaml_string = yaml.dump(cls.model_json_schema(), sort_keys=False)

        if uri:
            cls._write_to_file(s=yaml_string, uri=uri, **kwargs)
            return None

        return yaml_string

    @classmethod
    def schema_to_json(
        cls,
        uri: str = None,
        encoder: callable = json.JSONEncoder,
        default: callable = None,
        **kwargs,
    ) -> Union[str, None]:
        """Returns the schema serialized as a JSON string

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
            cls.model_json_schema(),
            cls=encoder,
            default=default,
        )
        if uri:
            cls._write_to_file(s=json_string, uri=uri, **kwargs)
            return None
        return json_string
