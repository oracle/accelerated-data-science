#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict, Generator
from ads.data_labeling.reader.jsonl_reader import JsonlReader


class ExportRecordReader(JsonlReader):
    """The ExportRecordReader class to read labeled dataset records from the export.

    Methods
    -------
    read(self) -> Generator[Dict, Any, Any]
        Reads labeled dataset records.
    """

    def __init__(
        self,
        path: str,
        auth: Dict = None,
        encoding="utf-8",
        includes_metadata: bool = False,
    ) -> "ExportRecordReader":
        """Initiates an ExportRecordReader instance.

        Parameters
        ----------
        path: str
            object storage path or local path for a file.
        auth: (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        encoding : (str, optional). Defaults to 'utf-8'.
            Encoding of files. Only used for "TEXT" dataset.
        includes_metadata: (bool, optional). Defaults to False.
            Determines whether the export file includes metadata or not.

        Examples
        --------
        >>> from ads.data_labeling.reader.export_record_reader import ExportRecordReader
        >>> path = "your/path/to/jsonl/file.jsonl"
        >>> from ads.common import auth as authutil
        >>> reader = ExportRecordReader(path=path, auth=authutil.api_keys(), encoding="utf-8")
        >>> next(reader.read())
        """
        super().__init__(path=path, auth=auth, encoding=encoding)
        self._includes_metadata = includes_metadata

    def read(self) -> Generator[Dict, Any, Any]:
        """Reads labeled dataset records.

        Returns
        -------
        Generator[Dict, Any, Any]
            The labeled dataset records.
        """
        skip = 1 if self._includes_metadata else None
        return super().read(skip=skip)
