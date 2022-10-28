#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module helping to load/save images from/to the local path or OCI object storage bucket.

Classes
-------
ADSImage
    Work with image files that are stored in Oracle Cloud Infrastructure Object Storage.

Examples
--------
>>> from ads.feature_engineering import ADSImage
>>> from IPython.core.display import display
>>> img = ADSImage.open("1.jpg")
>>> display(img)
>>> img.save("oci://<bucket_name>@<namespace>/1.jpg")
>>> img1 = ADSImage.open("oci://<bucket_name>@<namespace>/1.jpg")
>>> display(img1)
"""

import os
from io import BytesIO
from typing import Dict, Optional

import fsspec
from ads.common import auth as authutil
from PIL import Image


class ADSImage:
    """
    Work with image files that are stored in Oracle Cloud Infrastructure Object Storage.
    PIL (Python Imaging Library) is used as a backend to represent and manipulate images.
    The PIL adds support
    for opening, manipulating, processing and saving various image file formats.

    Attributes
    ----------
    img: Image.Image
        A PIL Image object.
    filename: str
        The image filename.

    Methods
    -------
    save(self, path: str, ...) -> None
        Saves the image under the given filename.
    open(cls, path: str, ...) -> ADSImage
        Opens the given image file.

    Examples
    --------
    >>> from ads.feature_engineering import ADSImage
    >>> from IPython.core.display import display
    >>> img = ADSImage.open("1.jpg")
    >>> img.save("oci://<bucket_name>@<namespace>/1.jpg")
    >>> img1 = ADSImage.open("oci://<bucket_name>@<namespace>/1.jpg")
    >>> display(img1)
    """

    def __init__(self, img: Image.Image, filename: Optional[str] = None) -> None:
        """Initializes ADSImage object.

        Parameters
        ----------
        img: PIL.Image.Image
            The PIL Image object.
        filename: (str, optional). Defaults to None.
            The image filename.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        TypeError
            If `img` is not an instance of `PIL.Image.Image`.
        ValueError
            If `img` is not provided.
        """
        if not img:
            raise ValueError("The parameter `img` is required.")

        if not isinstance(img, Image.Image):
            raise TypeError("The `img` parameter must be a `PIL.Image.Image` object.")

        self.img = img
        self.filename = filename

    def save(
        self,
        path: str,
        format: Optional[str] = None,
        auth: Optional[Dict] = None,
        **kwargs: Optional[Dict]
    ) -> None:
        """Save the image under the given filename.
        If no format is specified, the format to use is determined from the image object
        or filename extension, if possible.

        Parameters
        ----------
        path: str
            The file path to save image. It can be a local path or an Oracle Cloud Infrastructure Object Storage URI.
            Example: `oci://<bucket_name>@<namespace>/1.jpg`
        format: (str, optional). Defaults to None.
            If omitted and `path` has a file extension the format of the image will be based on the extension.
            Can be any format supported by PIL Image.
            The available options are described in the PIL image format documentation:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html

        auth: (Dict, optional). Defaults to None.
            The default authentication is set using `ads.set_auth()` API. To override the
            default behavior, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create an
            authentication signer and provide an `IdentityClient` object.
        kwargs:
            Additional keyword arguments that would be passed to the PIL Image `save()` method.

        Returns
        -------
        None
            Nothing.
        """
        imgByteArr = BytesIO()
        self.img.save(imgByteArr, format=format or self.img.format, params=kwargs)
        auth = auth or authutil.default_signer()
        with fsspec.open(path, mode="wb", **auth) as f:
            f.write(imgByteArr.getvalue())

    @classmethod
    def open(cls, path: str, storage_options: Optional[Dict] = None) -> "ADSImage":
        """Opens the given image file.

        Parameters
        ----------
        path: str
            The file path to open image. Can be local path or OCI object storage URI.
            Example: `oci://<bucket_name>@<namespace>/1.jpg`
        storage_options: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        ADSImage
            The instance of `ADSImage`.
        """
        if storage_options == None:
            storage_options = authutil.default_signer()
        with fsspec.open(path, mode="rb", **storage_options) as f:
            return cls(
                img=Image.open(BytesIO(f.read())),
                filename=os.path.basename(f.path),
            )

    def to_bytes(self, **kwargs: Optional[Dict]) -> BytesIO:
        """Converts image to bytes.
        If no format is specified, the format to use is determined from the image object if it possible.

        Parameters
        ---------------
        kwargs:
            format: (str, optional). Defaults to None.
                If omitted and `path` has a file extension the format of the image will be based on the extension.
                Can be any format supported by PIL Image.
                The available options are described in the PIL image format documentation:
                https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
            Additional keyword arguments that would be passed to the PIL Image `save()` method.

        Returns
        -------
        BytesIO
            The image converted to bytes.
        """
        imgByteArr = BytesIO()
        self.img.save(
            imgByteArr, format=kwargs.get("format") or self.img.format, params=kwargs
        )
        return imgByteArr.getvalue()

    def __getattr__(self, key: str):
        if key == "img":
            raise AttributeError()
        return getattr(self.img, key)
