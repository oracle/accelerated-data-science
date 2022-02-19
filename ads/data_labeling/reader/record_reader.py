#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict, Generator, List, Tuple, Union

from ads.config import JOB_RUN_COMPARTMENT_OCID, NB_SESSION_COMPARTMENT_OCID
from ads.data_labeling.interface.loader import Loader
from ads.data_labeling.interface.parser import Parser
from ads.data_labeling.interface.reader import Reader
from ads.data_labeling.loader.file_loader import FileLoaderFactory
from ads.data_labeling.parser.dls_record_parser import DLSRecordParserFactory
from ads.data_labeling.parser.export_record_parser import RecordParserFactory
from ads.data_labeling.reader.dls_record_reader import DLSRecordReader
from ads.data_labeling.reader.export_record_reader import ExportRecordReader


class RecordReader:
    """Record Reader Class consists of parser, reader and loader. Reader reads the
    the content from the record file. Parser parses the label for each record. And
    Loader loads the content of the file path in that record.

    Examples
    --------
    >>> import os
    >>> import oci
    >>> from ads.data_labeling import RecordReader
    >>> from ads.common import auth as authutil
    >>> file_path = "/path/to/your_record.jsonl"
    >>> dataset_type = "IMAGE"
    >>> annotation_type = "BOUNDING_BOX"
    >>> record_reader = RecordReader.from_export_file(file_path, dataset_type, annotation_type, "image_file_path", authutil.api_keys())
    >>> next(record_reader.read())
    """

    def __init__(
        self,
        reader: Reader,
        parser: Parser,
        loader: Loader = None,
        include_unlabeled: bool = False,
        encoding: str = "utf-8",
        materialize: bool = False,
    ) -> "RecordReader":
        """Initiates a RecordReader instance.

        Parameters
        ----------
        reader: Reader
            Reader instance to read content from the record file.
        parser: Parser
            Parser instance to parse the labels from record file.
        loader: Loader. Defaults to None.
            Loader instance to load the content from the file path in the record.
        materialize: bool, optional. Defaults to False.
            Whether to materialize the content using loader.
        include_unlabeled: (bool, optional). Default to False.
            Whether to load the unlabeled records or not.
        encoding: str, optional
            Encoding for text files. Used only to extract the content of the text dataset contents.

        Raises
        ------
        ValueError
            If the record reader and record parser must be specified.
            If the loader is not specified when materialize if True.
        """
        if not reader:
            raise ValueError("The record reader must be specified.")
        if not parser:
            raise ValueError("The record parser must be specified.")
        if materialize and not loader:
            raise ValueError("The content loader must be specified.")

        self.reader = reader
        self.parser = parser
        self.loader = loader
        self.materialize = materialize
        self.include_unlabeled = include_unlabeled
        self.encoding = encoding

    @classmethod
    def from_export_file(
        cls,
        path: str,
        dataset_type: str,
        annotation_type: str,
        dataset_source_path: str,
        auth: Dict = None,
        include_unlabeled: bool = False,
        encoding: str = "utf-8",
        materialize: bool = False,
        format: str = None,
        categories: List[str] = None,
        includes_metadata=False,
    ) -> "RecordReader":
        """Initiates a RecordReader instance.

        Parameters
        ----------
        path: str
            Record file path.
        dataset_type: str
            Dataset type. Currently supports TEXT, IMAGE and DOCUMENT.
        annotation_type: str
            Annotation Type. Currently TEXT supports SINGLE_LABEL, MULTI_LABEL,
            ENTITY_EXTRACTION. IMAGE supports SINGLE_LABEL, MULTI_LABEL and BOUNDING_BOX.
            DOCUMENT supports SINGLE_LABEL and MULTI_LABEL.
        dataset_source_path: str
            Dataset source path.
        auth: (dict, optional). Default None
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        include_unlabeled: (bool, optional). Default to False.
            Whether to load the unlabeled records or not.
        encoding : (str, optional). Defaults to "utf-8".
            Encoding for text files. Used only to extract the content of the text dataset contents.
        materialize: (bool, optional). Defaults to False.
            Whether to materialize the content by loader.
        format: (str, optional). Defaults to None.
            Output format of annotations. Can be None, "spacy" for dataset
            Entity Extraction type or "yolo" for Object Detection type.
            When None, it outputs List[NERItem] or List[BoundingBoxItem].
            When "spacy", it outputs List[Tuple].
            When "yolo", it outputs List[List[Tuple]].
        categories: (List[str], optional). Defaults to None.
            The list of object categories in proper order for model training.
            Example: ['cat','dog','horse']
        includes_metadata: (bool, optional). Defaults to False.
            Determines whether the export file includes metadata or not.

        Returns
        -------
        RecordReader
            A RecordReader instance.
        """
        reader = ExportRecordReader(
            path=path, auth=auth, encoding="utf-8", includes_metadata=includes_metadata
        )
        parser = RecordParserFactory.parser(
            annotation_type=annotation_type,
            dataset_source_path=dataset_source_path,
            format=format,
            categories=categories,
        )
        loader = FileLoaderFactory.loader(
            dataset_type=dataset_type,
            auth=auth,
        )
        return cls(
            reader=reader,
            parser=parser,
            loader=loader,
            materialize=materialize,
            include_unlabeled=include_unlabeled,
            encoding=encoding,
        )

    @classmethod
    def from_DLS(
        cls,
        dataset_id: str,
        dataset_type: str,
        annotation_type: str,
        dataset_source_path: str,
        compartment_id: str = None,
        auth: Dict = None,
        include_unlabeled: bool = False,
        encoding: str = "utf-8",
        materialize: bool = False,
        format: str = None,
        categories: List[str] = None,
    ) -> "RecordReader":
        """Constructs Record Reader instance.

        Parameters
        ----------
        dataset_id: str
            The dataset OCID.
        dataset_type: str
            Dataset type. Currently supports TEXT, IMAGE and DOCUMENT.
        annotation_type: str
            Annotation Type. Currently TEXT supports SINGLE_LABEL, MULTI_LABEL,
            ENTITY_EXTRACTION. IMAGE supports SINGLE_LABEL, MULTI_LABEL and BOUNDING_BOX.
            DOCUMENT supports SINGLE_LABEL and MULTI_LABEL.
        dataset_source_path: str
            Dataset source path.
        compartment_id: (str, optional). Defaults to None.
            The compartment OCID of the dataset.
        auth: (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        encoding: (str, optional). Defaults to 'utf-8'.
            Encoding for files.
        materialize: (bool, optional). Defaults to False.
            Whether the content of the dataset file should be loaded or it should return the file path to the content.
            By default the content will not be loaded.
        format: (str, optional). Defaults to None.
            Output format of annotations. Can be None, "spacy" for dataset
            Entity Extraction type or "yolo" for Object Detection type.
            When None, it outputs List[NERItem] or List[BoundingBoxItem].
            When "spacy", it outputs List[Tuple].
            When "yolo", it outputs List[List[Tuple]].
        categories: (List[str], optional). Defaults to None.
            The list of object categories in proper order for model training.
            Example: ['cat','dog','horse']

        Returns
        -------
        RecordReader
            The RecordReader instance.
        """

        if compartment_id is None:
            compartment_id = NB_SESSION_COMPARTMENT_OCID or JOB_RUN_COMPARTMENT_OCID

        if not compartment_id:
            raise ValueError("The `compartment_id` must be provided.")

        reader = DLSRecordReader(
            compartment_id=compartment_id, dataset_id=dataset_id, auth=auth
        )
        parser = DLSRecordParserFactory.parser(
            annotation_type=annotation_type,
            dataset_source_path=dataset_source_path,
            format=format,
            categories=categories,
        )
        loader = FileLoaderFactory.loader(
            dataset_type=dataset_type,
            auth=auth,
        )
        return cls(
            reader=reader,
            parser=parser,
            loader=loader,
            materialize=materialize,
            include_unlabeled=include_unlabeled,
            encoding=encoding,
        )

    def read(self) -> Generator[Tuple[str, Union[List, str]], Any, Any]:
        """Reads the record.

        Yields
        ------
        Generator[Tuple[str, Union[List, str]], Any, Any]
            File path, content and labels in a tuple.
        """
        for item in self.reader.read():
            if not item:
                return None
            record = self.parser.parse(item)
            if record.annotation or self.include_unlabeled:
                if self.materialize:
                    record.content = self.loader.load(
                        record.path, **{"encoding": self.encoding}
                    )
                yield record.to_tuple()
