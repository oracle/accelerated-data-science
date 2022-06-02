#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module loads and saves images from/to a local path or Oracle Object Storage.

Classes
-------
ADSImageReader
    The class reads image files from a local path
    or an Oracle Cloud Infrastructure Object Storage bucket.

Examples
--------
>>> from ads.feature_engineering import ADSImageReader
>>> image_reader = ADSImageReader.from_uri(path="oci://<bucket_name>@<namespace>/*.jpg")
>>> index=1
>>> for image in image_reader.read():
...     image.save(f"{index}.jpg")
...     index+=1
"""

import os
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional, Union

import fsspec
from ads.common import auth as authutil
from PIL import Image

from .image import ADSImage
from .interface.reader import Reader


class ADSImageReader:
    """
    The class reads image files from a local path
    or an Oracle Cloud Infrastructure Object Storage bucket.

    Methods
    -------
    from_uri(cls, path: Union[str, List[str]], ...) -> ADSImageReader
        Constructs the ADSImageReader object.
    read(self) -> Generator[ADSImage, Any, Any]
        Reads images.

    Examples
    --------
    >>> from ads.feature_engineering import ADSImageReader
    >>> image_reader = ADSImageReader.from_uri(path="oci://<bucket_name>@<namespace>/*.jpg")
    >>> index=1
    >>> for image in image_reader.read():
    ...     image.save(f"{index}.jpg")
    ...     index+=1
    """

    def __init__(self, reader: Reader) -> None:
        """Initializes ADSImageReader instance.

        Parameters
        ----------
        reader: Reader
            Can be any reader implementing the ```Reader``` interface.

        Returns
        -------
        None
            Nothing.
        """
        self._reader = reader

    @classmethod
    def from_uri(
        cls,
        path: Union[str, List[str]],
        auth: Optional[Dict] = None,
    ) -> "ADSImageReader":
        """Constructs the ADSImageReader object.

        Parameters
        ----------
        path: Union[str, List[str]]
            The path where the images located.
            Can be local path, OCI object storage URI or a list of paths.
            It is also support globbing.
            Example:
                `oci://<bucket_name>@<namespace>/1.jpg`
                [`oci://<bucket_name>@<namespace>/1.jpg`, `oci://<bucket_name>@<namespace>/2.jpg`]
                `oci://<bucket_name>@<namespace>/*.jpg`

        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        ADSImageReader
            The ADSImageReader object.
        """
        auth = auth or authutil.default_signer()
        return cls(reader=ImageFileReader(path=path, auth=auth))

    def read(self) -> Generator[ADSImage, Any, Any]:
        """Read image files.

        Yields
        ------
        Generator[ADSImage, Any, Any]
        """
        return self._reader.read()


class ImageFileReader:
    """The class reads image files from a local path or an Oracle Cloud Infrastructure Object Storage bucket."""

    def __init__(
        self, path: Union[str, List[str]], auth: Optional[Dict] = None
    ) -> None:
        """Initializes ImageFileReader instance.

        Parameters
        ----------
        path: Union[str, List[str]]
            The path where the images located.
            Can be local path, OCI object storage URI or a list of paths.
            It is also support globbing.
            Example:
                `oci://<bucket_name>@<namespace>/1.jpg`
                [`oci://<bucket_name>@<namespace>/1.jpg`, `oci://<bucket_name>@<namespace>/2.jpg`]
                `oci://<bucket_name>@<namespace>/*.jpg`
        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        TypeError
            If the `path` is not an instance of `str` or `List[str]`.
        ValueError
            If the `path` is not provided.
        """
        if not path:
            raise ValueError("The parameter `path` is required.")

        if not isinstance(path, (str, list)):
            raise TypeError("The `path` parameter must be a string or list of strings.")

        self.path = path
        self.auth = auth or authutil.default_signer()

    def read(self) -> Generator[ADSImage, Any, Any]:
        """Read image files.

        Yields
        ------
        Generator[ADSImage, Any, Any]
        """
        with fsspec.open_files(self.path, **self.auth) as files:
            for f in files:
                yield ADSImage(Image.open(BytesIO(f.read())), os.path.basename(f.path))
