#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import fsspec
import io
import json
import os
from typing import Dict, Generator, List, Union

from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.text_dataset.utils import PY4JGateway, experimental
from fsspec.core import OpenFile


class Base:
    """Base class for backends."""

    def read_line(
        self, fhandler: OpenFile
    ) -> Generator[Union[str, List[str]], None, None]:
        """Read lines from a file.

        Parameters
        ----------
        fhandler : `fsspec.core.OpenFile`
            a file handler returned by `fsspec`

        Yields
        -------
        Generator
            a generator that yields lines
        """
        with fhandler as f:
            for line in f:
                yield line.decode(fhandler.encoding)

    def read_text(
        self, fhandler: OpenFile
    ) -> Generator[Union[str, List[str]], None, None]:
        """Read entire file into a string.

        Parameters
        ----------
        fhandler : `fsspec.core.OpenFile`
            a file handler returned by `fsspec`

        Yields
        -------
        Generator
            a generator that yields text in the file
        """
        with fhandler as f:
            yield f.read().decode(fhandler.encoding)

    @staticmethod
    def _validate_dest(src, dst_folder: str, fname: str = None) -> str:
        if fname is None:
            fname = os.path.splitext(os.path.basename(src))[0] + ".txt"
        return os.path.join(dst_folder, fname)

    def convert_to_text(
        self,
        fhandler: OpenFile,
        dst_path: str,
        fname: str = None,
        storage_options: Dict = None,
    ) -> str:
        """Convert input file to a text file

        Parameters
        ----------
        fhandler : `fsspec.core.OpenFile`
            a file handler returned by `fsspec`
        dst_path: str
            local folder or cloud storage prefix to save converted text files
        fname: str, optional
            filename for converted output, relative to dirname or prefix, by default None
        storage_options: dict, optional
            storage options for cloud storage

        Returns
        -------
        str
            path to saved output
        """
        dest = self._validate_dest(fhandler.path, dst_path, fname)
        storage_options = {} if storage_options is None else storage_options
        with fsspec.open(dest, mode="wb", **storage_options) as fout:
            with fhandler as fin:
                fout.write(fin.read())
        return dest

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
        return {}


class Tika(Base):
    def read_line(self, fhandler):
        with PY4JGateway() as gateway:
            parser = gateway.jvm.parsers.TikaParser()
            with fhandler as f:
                reader = parser.reader(f.read())
            while True:
                line = reader.readLine()
                if line is not None:
                    yield line
                else:
                    break

    def read_text(self, fhandler):
        with PY4JGateway() as gateway:
            parser = gateway.jvm.parsers.TikaParser()
            with fhandler as f:
                yield parser.readText(f.read()).decode()

    def convert_to_text(self, fhandler, dst_path, fname=None, storage_options=None):
        dest = self._validate_dest(fhandler.path, dst_path, fname)
        storage_options = {} if storage_options is None else storage_options
        with fsspec.open(dest, mode="w", **storage_options) as f:
            f.write(next(self.read_text(fhandler)))
        return dest

    def get_metadata(self, fhandler):
        with PY4JGateway() as gateway:
            parser = gateway.jvm.parsers.TikaParser()
            meta = gateway.jvm.org.apache.tika.metadata.Metadata()
            with fhandler as f:
                parser.readText(f.read(), meta)
            return json.loads(parser.getMetadata(meta))

    def detect_encoding(self, fhandler: OpenFile):
        with PY4JGateway() as gateway:
            with fhandler as f:
                return gateway.jvm.parsers.TikaParser.detectEncoding(f.read())


class PDFPlumber(Base):
    @runtime_dependency(
        module="pdfplumber", err_msg="pdfplumber must be installed first."
    )
    def __init__(self):
        super().__init__()

    def read_line(self, fhandler):
        import pdfplumber

        with fhandler as f:
            pdf = pdfplumber.PDF(f)
            for page in pdf.pages:
                reader = io.StringIO(page.extract_text())
                for line in reader:
                    yield line

    def read_text(self, fhandler):
        import pdfplumber

        with fhandler as f:
            pdf = pdfplumber.PDF(f)
            texts = (page.extract_text() for page in pdf.pages)
            yield "\n".join([text for text in texts if text is not None])

    def convert_to_text(
        self,
        fhandler,
        dst_path,
        fname=None,
        storage_options=None,
    ):
        import pdfplumber

        dest = self._validate_dest(fhandler.path, dst_path, fname)
        storage_options = {} if storage_options is None else storage_options
        with fhandler as fin:
            pdf = pdfplumber.PDF(fin)
            with fsspec.open(dest, mode="w", **storage_options) as fout:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text is not None:
                        fout.write(text)
        return dest

    def get_metadata(self, fhandler):
        import pdfplumber

        with fhandler as f:
            pdf = pdfplumber.PDF(f)
            return pdf.metadata


@experimental
class OITCC(Base):  # pragma: no cover
    def convert_to_text(self, fhandler, output_path, fname=None, storage_options=None):
        dest = self._validate_dest(fhandler.path, output_path, fname)
        storage_options = {} if storage_options is None else storage_options
        with fsspec.open(dest, mode="w", **storage_options) as f:
            f.write(next(self.read_text(fhandler)))
