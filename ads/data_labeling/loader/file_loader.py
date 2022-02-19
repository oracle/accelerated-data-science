#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, List, Union

import fsspec
import PIL
from ads.common import auth as authutil
from ads.data_labeling.constants import DatasetType
from ads.data_labeling.interface.loader import Loader
from ads.text_dataset.dataset import TextDatasetFactory, backends
from PIL import Image

THREAD_POOL_MAX_WORKERS = 10


class FileLoader:
    """FileLoader Base Class.

    Attributes:
    ----------
    auth: (dict, optional). Defaults to None.
        The default authetication is set using `ads.set_auth` API. If you need to override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
        authentication signer and kwargs required to instantiate IdentityClient object.

    Examples
    --------
    >>> from ads.data_labeling.loader.file_loader import FileLoader
    >>> import oci
    >>> import os
    >>> from ads.common import auth as authutil
    >>> path = "path/to/your_text_file.txt"
    >>> file_content = FileLoader(auth=authutil.api_keys()).load(path)
    """

    def __init__(self, auth: Dict = None) -> "FileLoader":
        """Initiates a FileLoader instance.

        Parameters
        ----------
        auth: (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        """
        self.auth = auth or authutil.default_signer()

    def load(self, path: str, **kwargs) -> BytesIO:
        """Loads the file content from the path.

        Parameters
        ----------
        path: str
            The file path, can be local or object storage path.
        kwargs:
            Nothing.

        Returns
        -------
        BytesIO
            The data in BytesIO format.
        """
        data = None
        try:
            with fsspec.open(path, **self.auth) as f:
                data = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"{path} not found.")
        except Exception as e:
            raise e

        return data

    def bulk_load(self, paths: List[str], **kwargs) -> Dict[str, Any]:
        """Loads the files content from the list of paths.
        The ThreadPoolExecutor is used to load the files in parallel threads.

        Parameters
        ----------
        paths: List[str]
            The list of file paths, can be local or object storage paths.

        Returns
        -------
        Dict[str, Any]
            The map between file path and file content.
        """
        result = {}
        if not paths or not isinstance(paths, list) or len(paths) == 0:
            return result

        with ThreadPoolExecutor(max_workers=THREAD_POOL_MAX_WORKERS) as pool:
            futures = {pool.submit(self.load, path, **kwargs): path for path in paths}
            result = {futures[task]: task.result() for task in as_completed(futures)}
        return result


class TextFileLoader(FileLoader):
    """
    TextFileLoader class which loads text files.

    Examples
    --------
    >>> from ads.data_labeling import TextFileLoader
    >>> import oci
    >>> import os
    >>> from ads.common import auth as authutil
    >>> path = "path/to/your_text_file.txt"
    >>> file_content = TextFileLoader(auth=authutil.api_keys()).load(path)
    """

    def load(
        self, path: str, backend: Union[str, backends.Base] = "default", **kwargs
    ) -> str:
        """Loads the content from the path.

        Parameters
        ----------
        path: str
            Text file path, can be local or object storage path.
        backend: Union[str, backends.Base]
            Default to "default". Valid options are "default" and "tika" or
            ads.text_dataset.backends.Base, ads.text_dataset.backends.Tika
        kwargs:
            encoding: (str, optional). Defaults to 'utf-8'.
                Encoding for text files. Used only to extract the content of the text dataset contents.

        Returns
        -------
        str
            Content of the text file.
        """
        format = os.path.splitext(path)[1].replace(".", "")
        try:
            content = next(
                TextDatasetFactory.format(format.lower())
                .backend(backend)
                .read_text(path, storage_options=self.auth, **kwargs)
            )[0]
        except FileNotFoundError:
            raise FileNotFoundError(f"{path} not found.")
        except Exception as e:
            raise e
        return content


class ImageFileLoader(FileLoader):
    """
    ImageFileLoader class which loads image files.

    Examples
    --------
    >>> from ads.data_labeling import ImageFileLoader
    >>> import oci
    >>> import os
    >>> from ads.common import auth as authutil
    >>> path = "path/to/image.png"
    >>> image = ImageFileLoader(auth=authutil.api_keys()).load(path)
    """

    def load(self, path: str, **kwargs) -> PIL.ImageFile.ImageFile:
        """Loads the image from the path.

        Parameters
        ----------
        path: str
            Image file path, can be local or object storage path.
        kwargs:
            Nothing.

        Returns
        -------
        PIL.ImageFile.ImageFile
            Image opened by Pillow.
        """
        data = None
        data = super().load(path=path)

        return Image.open(BytesIO(data))


class FileLoaderFactory:
    """FileLoaderFactory class to create/register FileLoaders."""

    _loaders = {
        DatasetType.TEXT: TextFileLoader,
        DatasetType.IMAGE: ImageFileLoader,
        DatasetType.DOCUMENT: FileLoader,
    }

    @staticmethod
    def loader(dataset_type: str, auth: Dict = None) -> FileLoader:
        """Gets the loader based on the dataset_type.

        Parameters
        ----------
        dataset_type: str
            Dataset type. Currently supports TEXT, IMAGE and DOCUMENT.
        auth: (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        FileLoader
            A FileLoader instance corresponding to the dataset_type.
        """
        if not dataset_type in FileLoaderFactory._loaders:
            raise ValueError(
                f"The wrong dataset type has been provided. "
                f"Supported dataset types are: `{DatasetType.TEXT}`, "
                f"`{DatasetType.IMAGE}` and `{DatasetType.DOCUMENT}`."
            )

        return FileLoaderFactory._loaders[dataset_type](auth=auth)

    @classmethod
    def register(cls, dataset_type: str, loader: Loader) -> None:
        """Registers a new loader for a given dataset_type.

        Parameters
        ----------
        dataset_type: str
            Dataset type. Currently supports TEXT and IMAGE.
        loader: Loader
            A Loader class which supports loading content of the given dataset_type.

        Returns
        -------
        None
            Nothing.
        """
        cls._parsers[dataset_type] = loader
