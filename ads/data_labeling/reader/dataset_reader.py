#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module containing classes to read labeled datasets.
Allows to read labeled datasets from exports or from the cloud.

Classes
-------
    LabeledDatasetReader
        The LabeledDatasetReader class to read labeled dataset.
    ExportReader
        The ExportReader class to read labeled dataset from the export.
    DLSDatasetReader
        The DLSDatasetReader class to read labeled dataset from the cloud.

Examples
--------
    >>> from ads.common import auth as authutil
    >>> from ads.data_labeling import LabeledDatasetReader
    >>> ds_reader = LabeledDatasetReader.from_export(
    ...    path="oci://bucket_name@namespace/dataset_metadata.jsonl",
    ...    auth=authutil.api_keys(),
    ...    materialize=True
    ... )
    >>> ds_reader.info()
        ------------------------------------------------------------------------
        annotation_type	                                            SINGLE_LABEL
        compartment_id	                                        TEST_COMPARTMENT
        dataset_id                                                  TEST_DATASET
        dataset_name                                           test_dataset_name
        dataset_type                                                        TEXT
        labels	                                                   ['yes', 'no']
        records_path                                             path/to/records
        source_path                                              path/to/dataset

    >>> ds_reader.read()
                                 Path            Content            Annotations
        -----------------------------------------------------------------------
        0   path/to/the/content/file1       file content                    yes
        1   path/to/the/content/file2       file content                     no
        2   path/to/the/content/file3       file content                     no

    >>> next(ds_reader.read(iterator=True))
        ("path/to/the/content/file1", "file content", "yes")

    >>> next(ds_reader.read(iterator=True, chunksize=2))
        [("path/to/the/content/file1", "file content", "yes"),
        ("path/to/the/content/file2", "file content", "no")]

    >>> next(ds_reader.read(chunksize=2))
                                Path            Content            Annotations
        ----------------------------------------------------------------------
        0   path/to/the/content/file1       file content                    yes
        1   path/to/the/content/file2       file content                     no

    >>> ds_reader = LabeledDatasetReader.from_DLS(
    ...    dataset_id="dataset_OCID",
    ...    compartment_id="compartment_OCID",
    ...    auth=authutil.api_keys(),
    ...    materialize=True
    ... )
"""

from functools import lru_cache
from typing import Any, Dict, Generator, Tuple, Union

import pandas as pd
from ads.common import auth as authutil
from ads.common.serializer import Serializable
from ads.data_labeling.interface.reader import Reader
from ads.data_labeling.reader.metadata_reader import Metadata, MetadataReader
from ads.data_labeling.reader.record_reader import RecordReader
from ads.data_labeling.constants import (
    FORMATS_TO_ANNOTATION_TYPE,
    ANNOTATION_TYPE_TO_FORMATS,
)
from ads.config import NB_SESSION_COMPARTMENT_OCID, JOB_RUN_COMPARTMENT_OCID

_LABELED_DF_COLUMNS = ["Path", "Content", "Annotations"]


class LabeledDatasetReader:
    """The labeled dataset reader class.

    Methods
    -------
    info(self) -> Metadata
        Gets labeled dataset metadata.
    read(self, iterator: bool = False) -> Union[Generator[Any, Any, Any], pd.DataFrame]
        Reads labeled dataset.
    from_export(cls, path: str, auth: Dict = None, encoding="utf-8", materialize: bool = False) -> "LabeledDatasetReader"
        Constructs a Labeled Dataset Reader instance.

    Examples
    --------
    >>> from ads.common import auth as authutil
    >>> from ads.data_labeling import LabeledDatasetReader

    >>> ds_reader = LabeledDatasetReader.from_export(
    ...    path="oci://bucket_name@namespace/dataset_metadata.jsonl",
    ...    auth=authutil.api_keys(),
    ...    materialize=True
    ... )

    >>> ds_reader = LabeledDatasetReader.from_DLS(
    ...    dataset_id="dataset_OCID",
    ...    compartment_id="compartment_OCID",
    ...    auth=authutil.api_keys(),
    ...    materialize=True
    ... )

    >>> ds_reader.info()
        ------------------------------------------------------------------------
        annotation_type	                                            SINGLE_LABEL
        compartment_id	                                        TEST_COMPARTMENT
        dataset_id                                                  TEST_DATASET
        dataset_name                                           test_dataset_name
        dataset_type                                                        TEXT
        labels	                                                   ['yes', 'no']
        records_path                                             path/to/records
        source_path                                              path/to/dataset

    >>> ds_reader.read()
                                 Path            Content            Annotations
        -----------------------------------------------------------------------
        0   path/to/the/content/file1       file content                    yes
        1   path/to/the/content/file2       file content                     no
        2   path/to/the/content/file3       file content                     no

    >>> next(ds_reader.read(iterator=True))
        ("path/to/the/content/file1", "file content", "yes")

    >>> next(ds_reader.read(iterator=True, chunksize=2))
        [("path/to/the/content/file1", "file content", "yes"),
        ("path/to/the/content/file2", "file content", "no")]

    >>> next(ds_reader.read(chunksize=2))
                                Path            Content            Annotations
        ----------------------------------------------------------------------
        0   path/to/the/content/file1       file content                    yes
        1   path/to/the/content/file2       file content                     no
    """

    def __init__(self, reader: Reader):
        """Initializes the labeled dataset reader instance.

        Parameters
        ----------
        reader: Reader
            The Reader instance which reads and extracts the labeled dataset.
        """
        self._reader = reader

    @classmethod
    def from_export(
        cls,
        path: str,
        auth: dict = None,
        encoding: str = "utf-8",
        materialize: bool = False,
        include_unlabeled: bool = False,
    ) -> "LabeledDatasetReader":
        """Constructs Labeled Dataset Reader instance.

        Parameters
        ----------
        path: str
            The metadata file path, can be either local or object storage path.
        auth: (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        encoding: (str, optional). Defaults to 'utf-8'.
            Encoding for files.
        materialize: (bool, optional). Defaults to False.
            Whether the content of the dataset file should be loaded or it should return the file path to the content.
            By default the content will not be loaded.

        Returns
        -------
        LabeledDatasetReader
            The LabeledDatasetReader instance.
        """
        auth = auth or authutil.default_signer()

        return cls(
            reader=ExportReader(
                path=path,
                auth=auth,
                encoding=encoding,
                materialize=materialize,
                include_unlabeled=include_unlabeled,
            )
        )

    @classmethod
    def from_DLS(
        cls,
        dataset_id: str,
        compartment_id: str = None,
        auth: dict = None,
        encoding: str = "utf-8",
        materialize: bool = False,
        include_unlabeled: bool = False,
    ) -> "LabeledDatasetReader":
        """Constructs Labeled Dataset Reader instance.

        Parameters
        ----------
        dataset_id: str
            The dataset OCID.
        compartment_id: str. Defaults to the compartment_id from the env variable.
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

        Returns
        -------
        LabeledDatasetReader
            The LabeledDatasetReader instance.
        """
        if compartment_id is None:
            compartment_id = NB_SESSION_COMPARTMENT_OCID or JOB_RUN_COMPARTMENT_OCID

        if not compartment_id:
            raise ValueError("The `compartment_id` must be provided.")

        return cls(
            reader=DLSDatasetReader(
                compartment_id=compartment_id,
                dataset_id=dataset_id,
                auth=auth or authutil.default_signer(),
                encoding=encoding,
                materialize=materialize,
                include_unlabeled=include_unlabeled,
            )
        )

    def info(self) -> Serializable:
        """Gets the labeled dataset metadata.

        Returns
        -------
        Metadata
            The labeled dataset metadata.
        """
        return self._reader.info()

    def read(
        self, iterator: bool = False, format: str = None, chunksize: int = None
    ) -> Union[Generator[Any, Any, Any], pd.DataFrame]:
        """Reads the labeled dataset records.

        Parameters
        ----------
        iterator: (bool, optional). Defaults to False.
            True if the result should be represented as a Generator.
            Fasle if the result should be represented as a Pandas DataFrame.
        format: (str, optional). Defaults to None.
            Output format of annotations. Can be None, "spacy" or "yolo".
        chunksize: (int, optional). Defaults to None.
            The number of records that should be read in one iteration.
            The result will be returned in a generator format.

        Returns
        -------
        Union[
            Generator[Tuple[str, str, Any], Any, Any],
            Generator[List[Tuple[str, str, Any]], Any, Any],
            Generator[pd.DataFrame, Any, Any],
            pd.DataFrame
        ]
            `pd.Dataframe` if `iterator` and `chunksize` are not specified.
            `Generator[pd.Dataframe] ` if `iterator` equal to False and `chunksize` is specified.
            `Generator[List[Tuple[str, str, Any]]]` if `iterator` equal to True and `chunksize` is specified.
            `Generator[Tuple[str, str, Any]]` if `iterator` equal to True and `chunksize` is not specified.
        """

        if chunksize:
            return self._bulk_read(
                iterator=iterator, format=format, chunksize=chunksize
            )

        if iterator:
            return self._reader.read(format=format)

        return pd.DataFrame(
            self._reader.read(format=format), columns=_LABELED_DF_COLUMNS
        )

    def _bulk_read(
        self, iterator: bool = False, format: str = None, chunksize: int = None
    ) -> Generator[Union[pd.DataFrame, Tuple[str, str, Any]], Any, Any]:
        """Reads the labeled dataset records by chunks.

        Parameters
        ----------
        iterator: (bool, optional). Defaults to False.
            True if the result should be represented as a Generator.
            Fasle if the result should be represented as a Pandas DataFrame.
        format: (str, optional). Defaults to None.
            Output format of annotations. Can be None, "spacy" or "yolo".
        chunksize: (int, optional). Defaults to None.
            The number of records that should be read in one iteration.
            Result will be represented as a generator.

        Yields
        -------
        Generator[Union[pd.DataFrame, Tuple[str, str, Any]], Any, Any]
            The generator that yields records either in Dataframe format or Tuple.

        Raises
        ------
            ValueError: If chunksize is empty or not a positive integer.
        """
        if not chunksize or not isinstance(chunksize, int) or chunksize < 1:
            raise ValueError("`chunksize` must be a positive integer.")

        result = []
        i = 0
        for record in self._reader.read(format=format):
            result.append(record)
            i += 1
            if i >= chunksize:
                yield result if iterator else pd.DataFrame(
                    result, columns=_LABELED_DF_COLUMNS
                )
                result = []
                i = 0
        if result:
            yield result if iterator else pd.DataFrame(
                result, columns=_LABELED_DF_COLUMNS
            )


class DLSDatasetReader(Reader):
    """The DLSDatasetReader class to read labeled dataset from the cloud.

    Methods
    -------
    info(self) -> Metadata
        Gets the labeled dataset metadata.
    read(self) -> Generator[Tuple, Any, Any]
        Reads the labeled dataset.
    """

    def __init__(
        self,
        dataset_id: str,
        compartment_id: str,
        auth: Dict,
        encoding="utf-8",
        materialize: bool = False,
        include_unlabeled: bool = False,
    ):
        """Initializes the DLS dataset reader instance.

        Parameters
        ----------
        dataset_id: str
            The dataset OCID.
        compartment_id: str
            The compartment OCID of the dataset.
        auth: (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        encoding: (str, optional). Defaults to 'utf-8'.
            Encoding for files. The encoding is used to extract the metadata information
            of the labeled dataset and also to extract the content of the text dataset records.
        materialize: (bool, optional). Defaults to False.
            Whether the content of dataset files should be loaded/materialized or not.
            By default the content will not be materialized.
        include_unlabeled: (bool, optional). Defaults to False.
            Whether to load the unlabeled records or not.

        Raises
        ------
            ValueError: When dataset_id is empty or not a string.
            TypeError: When dataset_id not a string.
        """
        if not dataset_id:
            raise ValueError("The dataset OCID must be specified.")

        if not isinstance(dataset_id, str):
            raise TypeError("The dataset_id must be a string.")

        if not compartment_id:
            raise ValueError("The compartment OCID must be specified.")

        self.dataset_id = dataset_id
        self.compartment_id = compartment_id
        self.auth = auth or authutil.default_signer()
        self.encoding = encoding
        self.materialize = materialize
        self.include_unlabeled = include_unlabeled

    @lru_cache(maxsize=1)
    def info(self) -> Metadata:
        """Gets the labeled dataset metadata.

        Returns
        -------
        Metadata
            The labeled dataset metadata.
        """
        return MetadataReader.from_DLS(
            compartment_id=self.compartment_id,
            dataset_id=self.dataset_id,
        ).read()

    def read(self, format: str = None) -> Generator[Tuple, Any, Any]:
        """Reads the labeled dataset records.

        Parameters
        ----------
        format: (str, optional). Defaults to None.
            Output format of annotations. Can be None, "spacy" for dataset
            Entity Extraction type or "yolo" for Object Detection type.
            When None, it outputs List[NERItem] or List[BoundingBoxItem].
            When "spacy", it outputs List[Tuple].
            When "yolo", it outputs List[List[Tuple]].

        Returns
        -------
        Generator[Tuple, Any, Any]
            The labeled dataset records.
        """

        metadata = self.info()

        records_reader = RecordReader.from_DLS(
            dataset_type=metadata.dataset_type,
            annotation_type=metadata.annotation_type,
            dataset_source_path=metadata.source_path,
            compartment_id=self.compartment_id,
            dataset_id=self.dataset_id,
            auth=self.auth,
            encoding=self.encoding,
            materialize=self.materialize,
            include_unlabeled=self.include_unlabeled,
            format=format,
            categories=metadata.labels,
        )
        return records_reader.read()


class ExportReader(Reader):
    """The ExportReader class to read labeled dataset from the export.

    Methods
    -------
    info(self) -> Metadata
        Gets the labeled dataset metadata.
    read(self) -> Generator[Tuple, Any, Any]
        Reads the labeled dataset.
    """

    def __init__(
        self,
        path: str,
        auth: Dict = None,
        encoding="utf-8",
        materialize: bool = False,
        include_unlabeled: bool = False,
    ):
        """Initializes the labeled dataset export reader instance.

        Parameters
        ----------
        path: str
            The metadata file path, can be either local or object storage path.
        auth: (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        encoding: (str, optional). Defaults to 'utf-8'.
            Encoding for files. The encoding is used to extract the metadata information
            of the labeled dataset and also to extract the content of the text dataset records.
        materialize: (bool, optional). Defaults to False.
            Whether the content of dataset files should be loaded/materialized or not.
            By default the content will not be materialized.
        include_unlabeled: (bool, optional). Defaults to False.
            Whether to load the unlabeled records or not.

        Raises
        ------
            ValueError: When path is empty or not a string.
            TypeError: When path not a string.
        """

        if not path:
            raise ValueError("The parameter `path` is required.")

        if not isinstance(path, str):
            raise TypeError("The parameter `path` must be a string.")

        self.path = path
        self.auth = auth or authutil.default_signer()
        self.encoding = encoding
        self.materialize = materialize
        self.include_unlabeled = include_unlabeled

    @lru_cache(maxsize=1)
    def info(self) -> Metadata:
        """Gets the labeled dataset metadata.

        Returns
        -------
        Metadata
            The labeled dataset metadata.
        """
        return MetadataReader.from_export_file(
            path=self.path,
            auth=self.auth,
        ).read()

    def read(self, format: str = None) -> Generator[Tuple, Any, Any]:
        """Reads the labeled dataset records.

        Parameters
        ----------
        format: (str, optional). Defaults to None.
            Output format of annotations. Can be None, "spacy" for dataset
            Entity Extraction type or "yolo" for Object Detection type.
            When None, it outputs List[NERItem] or List[BoundingBoxItem].
            When "spacy", it outputs List[Tuple].
            When "yolo", it outputs List[List[Tuple]].

        Returns
        -------
        Generator[Tuple, Any, Any]
            The labeled dataset records.
        """
        metadata = self.info()
        if (
            format
            and isinstance(format, str)
            and (
                format.lower() not in FORMATS_TO_ANNOTATION_TYPE.keys()
                or FORMATS_TO_ANNOTATION_TYPE[format.lower()]
                != metadata.annotation_type
            )
        ):
            raise TypeError(
                "Wrong format. `format` can only be None or "
                f"`{ANNOTATION_TYPE_TO_FORMATS[metadata.annotation_type]}`."
            )

        records_reader = RecordReader.from_export_file(
            path=metadata.records_path or self.path,
            dataset_type=metadata.dataset_type,
            annotation_type=metadata.annotation_type,
            dataset_source_path=metadata.source_path,
            auth=self.auth,
            encoding=self.encoding,
            materialize=self.materialize,
            include_unlabeled=self.include_unlabeled,
            format=format,
            categories=metadata.labels,
            includes_metadata=not metadata.records_path,
        )
        return records_reader.read()
