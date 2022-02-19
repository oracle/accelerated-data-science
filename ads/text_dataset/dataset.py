#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import itertools
from typing import Any, Callable, Dict, Generator, List, Union

import ads
import ads.text_dataset.extractor as te
import fsspec
import pandas as pd
from ads.text_dataset import backends
from ads.text_dataset.options import OptionFactory, Options
from ads.text_dataset.udfs import UDF
from ads.text_dataset.utils import NotSupportedError


class DataLoader:
    """
    DataLoader binds engine, FileProcessor and File handler(in this case it is fsspec)
    together to produce a dataframe of parsed text from files.

    This class is expected to be used mainly from TextDatasetFactory class.

    Attributes
    ----------
    processor: `ads.text_dataset.extractor.FileProcessor`
        processor that is used for loading data.

    Examples
    --------
    >>> import oci
    >>> from ads.text_dataset.dataset import TextDatasetFactory as textfactory
    >>> from ads.text_dataset.options import Options
    >>> df = textfactory.format('pdf').engine('pandas').read_line(
    ...     'oci://<bucket-name>@<namespace>/<path>/*.pdf',
    ...     storage_options={"config": oci.config.from_file(os.path.join("~/.oci", "config"))},
    ... )
    >>> data_gen = textfactory.format('pdf').option(Options.FILE_NAME).backend('pdfplumber').read_text(
    ...     'oci://<bucket-name>@<namespace>/<path>/*.pdf',
    ...     storage_options={"config": oci.config.from_file(os.path.join("~/.oci", "config"))},
    ... )
    >>> textfactory.format('docx').convert_to_text(
    ...     'oci://<bucket-name>@<namespace>/<path>/*.docx',
    ...     './extracted',
    ...     storage_options={"config": oci.config.from_file(os.path.join("~/.oci", "config"))},
    ... )
    >>> textfactory.format('docx').convert_to_text(
    ...     'oci://<bucket-name>@<namespace>/<path>/*.docx',
    ...     'oci://<bucket-name>@<namespace>/<out_path>',
    ...     storage_options={"config": oci.config.from_file(os.path.join("~/.oci", "config"))},
    ... )
    >>> meta_gen = textfactory.format('docx').metadata_schema(
    ...     'oci://<bucket-name>@<namespace>/papers/*.pdf',
    ...     storage_options={"config": oci.config.from_file(os.path.join("~/.oci", "config"))},
    ... )
    >>> df = textfactory.format('pdf').engine('pandas').option(Options.FILE_METADATA, {'extract': ['Author']}).read_text(
    ...     'oci://<bucket-name>@<namespace>/<path>/*.pdf',
    ...     storage_options={"config": oci.config.from_file(os.path.join("~/.oci", "config"))},
    ...     total_files=10,
    ... )
    >>> df = textfactory.format('txt').engine('cudf').read_line(
    ...     'oci://<bucket-name>@<namespace>/<path>/*.log',
    ...      udf=r'^\[(\S+)\s(\S+)\s(\d+)\s(\d+\:\d+\:\d+)\s(\d+)]\s(\S+)\s(\S+)\s(\S+)\s(\S+)',
    ...      df_args={"columns":["day", "month", "date", "time", "year", "type", "method", "status", "file"]},
    ...      n_lines_per_file=10,
    ... )
    """

    def __init__(self, engine: str = None) -> None:
        """Initialize a DataLoader object.

        Parameters
        ----------
        engine : str, optional
            dataframe engine, by default None.

        Returns
        -------
        None
        """
        self.engine(engine)
        self.filemanager = fsspec
        self.processor = te.FileProcessorFactory.get_processor("txt")
        self.options = []
        self._data = None

    def with_processor(self, processor_type: str) -> None:
        """Set file processor.

        Parameters
        ----------
        processor_type : str
            type of processor, which corresponds to format of the file.

        Returns
        -------
        None
        """
        self.processor = te.FileProcessorFactory.get_processor(processor_type)()
        return self

    def engine(self, eng: str) -> None:
        """Set engine for dataloader. Can be pandas or cudf.

        Parameters
        ----------
        eng : str
            name of engine

        Returns
        -------
        None

        Raises
        ------
        NotSupportedError
            raises error if engine passed in is not supported.
        """
        if eng is None:
            self._engine = None
            self._format_output = lambda *args, **kwargs: args[0]
            return self
        if eng not in ["pandas", "cudf"]:
            raise NotSupportedError("Only pandas and cudf currently.")
        else:
            if eng == "pandas":
                import pandas

                self._engine = pandas
                self._format_output = pandas.DataFrame
            else:
                import cudf

                self._engine = cudf
                self._format_output = lambda data, **kwargs: cudf.DataFrame(
                    [row for row in data], **kwargs
                )  # cuDF cannot be initialized with a generator
        return self

    def backend(self, backend: Union[str, backends.Base]) -> None:
        """Set backend used for extracting text from files.

        Parameters
        ----------
        backend : (str | `ads.text_dataset.backends.Base`)
            backend for extracting text from raw files.

        Returns
        -------
        None
        """
        self.processor.backend(backend)
        return self

    def option(self, opt: Options, spec: Any = None) -> None:
        """Set extraction options.

        Parameters
        ----------
        opt : `ads.text_dataset.options.Options`
            an option defined in `ads.text_dataset.options.Options`
        spec : Any, optional
            specifications that will be passed to option handler, by default None

        Returns
        -------
        None
        """
        self.options.append((OptionFactory.option_handler(opt), spec))
        return self

    def __load_data__(
        self,
        reader: Callable,
        path: str,
        udf: Union[str, Callable] = None,
        storage_options: Dict = None,
        encoding: str = "utf-8",
        n_rows_per_file: int = None,
        total_rows: int = None,
    ) -> Generator[Union[str, List[str]], None, None]:
        storage_options = storage_options if storage_options is not None else {}
        fhs = self.filemanager.open_files(
            path, mode="rb", encoding=encoding, **storage_options
        )
        if udf is not None:
            if isinstance(udf, str):
                fn = UDF.from_regex(udf)
            else:
                fn = udf
        else:
            fn = lambda x: x

        total_line_count = [0]

        # function to apply to each element
        def func(fh, reader):
            out = [option(self).handle(fh, spec) for option, spec in self.options]
            line_count = 0
            for text in reader(fh):
                if total_rows is None or total_line_count[0] < total_rows:
                    if n_rows_per_file is None or line_count < n_rows_per_file:
                        content = fn(text)
                        if content is not None:
                            yield out + list(content) if (
                                isinstance(content, list) or isinstance(content, tuple)
                            ) else out + [content]
                            line_count += 1
                            total_line_count[0] += 1

        return itertools.chain.from_iterable((func(fh, reader) for fh in fhs))

    def read_line(
        self,
        path: str,
        udf: Union[str, Callable] = None,
        n_lines_per_file: int = None,
        total_lines: int = None,
        df_args: Dict = None,
        storage_options: Dict = None,
        encoding: str = "utf-8",
    ) -> Union[Generator[Union[str, List[str]], None, None], "DataFrame"]:
        """Read each file into lines. If path matches multiple files, will combine lines from all files.

        Parameters
        ----------
        path : str
            path to data files. can have glob pattern.
        udf : (callable | str), optional
            user defined function for processing each line, can be a callable or regex, by default None
        n_lines_per_file : int, optional
            max number of lines read from each file, by default None
        total_lines : int, optional
            max number of lines read from all files, by default None
        df_args : dict, optional
            arguments passed to dataframe engine (e.g. pandas), by default None
        storage_options : dict, optional
            storage options for cloud storage, by default None
        encoding : str, optional
            encoding of files, by default 'utf-8'

        Returns
        -------
        (Generator | DataFrame)
            returns either a data generator or a dataframe.
        """
        df_args = df_args if df_args is not None else {}
        self._data = self.__load_data__(
            self.processor.read_line,
            path,
            udf,
            storage_options,
            encoding,
            n_lines_per_file,
            total_lines,
        )
        return self._format_output(self._data, **df_args)

    def read_text(
        self,
        path: str,
        udf: Union[str, Callable] = None,
        total_files: int = None,
        storage_options: Dict = None,
        df_args: Dict = None,
        encoding: str = "utf-8",
    ) -> Union[Generator[Union[str, List[str]], None, None], "DataFrame"]:
        """Read each file into a text string. If path matches multiple files, each file corresponds to one record.

        Parameters
        ----------
        path : str
            path to data files. can have glob pattern.
        udf : (callable | str), optional
            user defined function for processing each line, can be a callable or regex, by default None
        total_files : int, optional
            max number of files to read, by default None
        df_args : dict, optional
            arguments passed to dataframe engine (e.g. pandas), by default None
        storage_options : dict, optional
            storage options for cloud storage, by default None
        encoding : str, optional
            encoding of files, by default 'utf-8'

        Returns
        -------
        (Generator | DataFrame)
            returns either a data generator or a dataframe.
        """
        df_args = df_args if df_args is not None else {}
        self._data = self.__load_data__(
            self.processor.read_text,
            path,
            udf,
            storage_options,
            encoding,
            1,
            total_files,
        )
        return self._format_output(self._data, **df_args)

    def convert_to_text(
        self,
        src_path: str,
        dst_path: str,
        encoding: str = "utf-8",
        storage_options: Dict = None,
    ) -> None:
        """Convert files to plain text files.

        Parameters
        ----------
        src_path : str
            path to source data file(s). can use glob pattern
        dst_path: str
            local folder or cloud storage (e.g., OCI object storage) prefix to save converted text files
        encoding: str, optional
            encoding for files, by default utf-8
        storage_options : Dict, optional
            storage options for cloud storage, by default None

        Returns
        -------
        None
        """
        storage_options = storage_options if storage_options is not None else {}
        fhs = self.filemanager.open_files(
            src_path, mode="rb", encoding=encoding, **storage_options
        )
        for fh in fhs:
            self.processor.convert_to_text(
                fh,
                dst_path,
                storage_options=storage_options,
            )

    def metadata_all(
        self, path: str, storage_options: Dict = None, encoding: str = "utf-8"
    ) -> Generator[Dict[str, Any], None, None]:
        """Get metadata of all files that matches the given path. Return a generator.

        Parameters
        ----------
        path : str
            path to data files. can use glob pattern.
        storage_options : Dict, optional
            storage options for cloud storage, by default None
        encoding : str, optional
            encoding of files, by default 'utf-8'

        Returns
        -------
        Generator
            generator of extracted metedata from files.
        """
        storage_options = storage_options if storage_options is not None else {}
        fhs = self.filemanager.open_files(
            path, mode="rb", encoding=encoding, **storage_options
        )
        return (self.processor.get_metadata(fh) for fh in fhs)

    def metadata_schema(
        self,
        path: str,
        n_files: int = 1,
        storage_options: Dict = None,
        encoding: str = "utf-8",
    ) -> List[str]:
        """
        Get available fields in metadata by looking at the first `n_files` that
        matches the given path.

        Parameters
        ----------
        path: str
            path to data files. can have glob pattern
        n_files: int, optional
            number of files to look up, default to be 1
        storage_options: dict, optional
            storage options for cloud storage, by default None
        encoding: str, optional
            encoding of files, by default utf-8

        Returns
        -------
        List[str]
            list of available fields in metadata
        """

        metadata = self.metadata_all(
            path, storage_options=storage_options, encoding=encoding
        )
        fields = set()
        for _ in range(n_files):
            try:
                fields.update(list(next(metadata).keys()))
            except StopIteration:
                break
        return list(fields)

    # ----- not currently used, but in case we want to consider chaining in the future -----
    def _transform(self, udf, udf_type="fn"):  # pragma: no cover
        if udf_type == "fn":
            func = UDF.from_lambda(udf)
        elif udf_type == "regex":
            func = UDF.from_regex(udf)
        else:
            raise NotImplementedError("Other types of UDF not yet supported.")

        # convert df into iterator
        if isinstance(self._data, pd.DataFrame) or isinstance(self._data, pd.Series):
            self._data = (
                row.values if len(row.values) > 1 else row.values[0]
                for i, row in self._data.iterrows()
            )

        self._data = (func(row) for row in self._data)
        self._data = (row for row in self._data if row is not None)
        return self


class TextDatasetFactory:
    """A class that generates a dataloader given a file format."""

    @staticmethod
    def format(format_name: str) -> DataLoader:
        """
        Instantiates DataLoader class and seeds it with the right kind of FileProcessor.
        Eg. PDFProcessor for pdf. The FileProcessorFactory returns the processor based
        on the format Type.

        Parameters
        ----------
        format_name : str
            name of format

        Returns
        -------
        `ads.text_dataset.dataset.DataLoader`
            a `DataLoader` object.
        """
        return DataLoader().with_processor(format_name)
