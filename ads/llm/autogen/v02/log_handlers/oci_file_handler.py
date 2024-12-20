# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import io
import json
import logging
import os
import threading

import fsspec

from ads.common.auth import default_signer

logger = logging.getLogger(__name__)


class OCIFileHandler(logging.FileHandler):
    """Log handler for saving log file to OCI object storage."""

    def __init__(
        self,
        filename: str,
        session_id: str,
        mode: str = "a",
        encoding: str | None = None,
        delay: bool = False,
        errors: str | None = None,
        auth: dict | None = None,
    ) -> None:
        self.session_id = session_id
        self.auth = auth

        if filename.startswith("oci://"):
            self.baseFilename = filename
        else:
            self.baseFilename = os.path.abspath(os.path.expanduser(filename))
            os.makedirs(os.path.dirname(self.baseFilename), exist_ok=True)

        # The following code are from the `FileHandler.__init__()`
        self.mode = mode
        self.encoding = encoding
        if "b" not in mode:
            self.encoding = io.text_encoding(encoding)
        self.errors = errors
        self.delay = delay

        if delay:
            # We don't open the stream, but we still need to call the
            # Handler constructor to set level, formatter, lock etc.
            logging.Handler.__init__(self)
            self.stream = None
        else:
            logging.StreamHandler.__init__(self, self._open())

    def _open(self):
        """
        Open the current base file with the (original) mode and encoding.
        Return the resulting stream.
        """
        auth = self.auth or default_signer()
        return fsspec.open(
            self.baseFilename,
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
            **auth,
        ).open()

    def format(self, record: logging.LogRecord):
        """Formats the log record as JSON payload and add session_id."""
        msg = record.getMessage()
        try:
            data = json.loads(msg)
        except Exception as e:
            data = {"message": msg}

        if "session_id" not in data:
            data["session_id"] = self.session_id
        if "thread_id" not in data:
            data["thread_id"] = threading.get_ident()

        record.msg = json.dumps(data)
        return super().format(record)

