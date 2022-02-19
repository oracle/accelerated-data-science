#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import os
from typing import Dict, Generator, List, Union

from ads.text_dataset import backends
from ads.text_dataset.backends import OITCC, Base, PDFPlumber, Tika
from ads.text_dataset.utils import NotSupportedError
from fsspec.core import OpenFile

logger = logging.getLogger("ads.text_dataset")


class FileProcessor:
    """
    Base class for all the file processor. Files are opened using fsspec library.
    The default implementation in the base class assumes text files.

    This class is expected to be used inside `ads.text_dataset.dataset.DataLoader`.
    """

    backend_map = {"default": Base, "tika": Tika}

    def __init__(self, backend: Union[str, backends.Base] = "default") -> None:
        self.backend(backend)

    def backend(self, backend: Union[str, backends.Base]) -> None:
        """Set backend for file processor.

        Parameters
        ----------
        backend : `ads.text_dataset.backends.Base`
            a backend for file processor

        Returns
        -------
        None

        Raises
        ------
        NotSupportedError
            when specified backend is not supported.
        """
        if isinstance(backend, str) and backend in self.backend_map:
            self._backend = self.backend_map[backend]()
        elif isinstance(backend, Base):
            self._backend = backend
        else:
            raise NotSupportedError(
                f"backend {backend} is not recognized or not a subclass of ads.text_dataset.backends.Base."
            )
        return self

    def read_line(
        self, fhandler: OpenFile, **format_reader_kwargs: Dict
    ) -> Generator[Union[str, List[str]], None, None]:
        """Yields lines from a file.

        Parameters
        ----------
        fhandler : `fsspec.core.OpenFile`
            file handler returned by `fsspec`

        Returns
        -------
        Generator
            a generator that yields lines from a file
        """
        return self._backend.read_line(fhandler, **format_reader_kwargs)

    def read_text(
        self, fhandler: OpenFile, **format_reader_kwargs: Dict
    ) -> Generator[Union[str, List[str]], None, None]:
        """Yield contents from the entire file.

        Parameters
        ----------
        fhandler : `fsspec.core.OpenFile`
            a file handler returned by fsspec

        Returns
        -------
        Generator
            a generator that yield text from a file
        """
        return self._backend.read_text(fhandler, **format_reader_kwargs)

    def convert_to_text(
        self,
        fhandler: OpenFile,
        dst_path: str,
        fname: str = None,
        storage_options: Dict = None,
    ) -> str:
        """Convert input file to a text file.

        Parameters
        ----------
        fhandler : `fsspec.core.OpenFile`
            a file handler returned by `fsspec`
        dst_path: str
            local folder or cloud storage (e.g. OCI object storage) prefix to save converted text files
        fname: str, optional
            filename for converted output, relative to dirname or prefix, by default None
        storage_options: dict, optional
            storage options for cloud storage, by default None

        Returns
        -------
        str
            path to saved output
        """
        return self._backend.convert_to_text(fhandler, dst_path, fname, storage_options)

    def get_metadata(self, fhandler: OpenFile) -> Dict:
        """Get metadata of a file.

        Parameters
        ----------
        fhandler : `fsspec.core.OpenFile`
            a file handler returned by fsspec

        Returns
        -------
        dict
            dictionary of metadata
        """
        return self._backend.get_metadata(fhandler)


class PDFProcessor(FileProcessor):
    """
    Extracts text content from PDF
    """

    backend_map = {"tika": Tika, "pdfplumber": PDFPlumber, "default": Tika}


class WordProcessor(FileProcessor):
    """
    Extracts text content from doc or docx format.
    """

    backend_map = {"default": Tika, "tika": Tika}


class FileProcessorFactory:
    """Factory that manages all file processors.
    Provides functionality to get a processor corresponding to a given file type,
    or register custom processor for a specific file format.

    Examples
    --------
    >>> from ads.text_dataset.extractor import FileProcessor, FileProcessorFactory
    >>> FileProcessorFactory.get_processor('pdf')
    >>> class CustomProcessor(FileProcessor):
    ... # custom logic here
    ... pass
    >>> FileProcessorFactory.register('new_format', CustomProcessor)
    """

    processor_map = {
        "pdf": PDFProcessor,
        "docx": WordProcessor,
        "doc": WordProcessor,
        "txt": FileProcessor,
    }

    @classmethod
    def register(cls, fmt: str, processor: FileProcessor) -> None:
        """Register custom file processor for a file format.

        Parameters
        ----------
        fmt : str
            file format
        processor : `FileProcessor`
            custom processor

        Raises
        ------
        TypeError
            raised when processor is not a subclass of `FileProcessor`.
        """
        if issubclass(processor, FileProcessor):
            cls.processor_map[fmt] = processor
        else:
            raise TypeError(f"Processor must inherit from FileProcessor class.")

    @staticmethod
    def get_processor(format):
        if format in FileProcessorFactory.processor_map:
            return FileProcessorFactory.processor_map[format]
        else:
            logger.warning(
                f"""
                Format {format} is not supported natively.
                A generic FileProcessor is returned.
                You can define and register a custom processor.
                """
            )
            return FileProcessor
