#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import dataclass


@dataclass
class ObjectStorageDetails:
    """ObjectStorageDetails class which contains bucket, namespace and filepath
    of a file.

    Attributes
    ----------
    bucket: str
        Bucket under which the file is stored at.
    namespace: str
        Namespace of the bucket.
    filepath: str
        File path in the bucket.

    """

    bucket: str
    namespace: str
    filepath: str = ""

    def __repr__(self):
        return self.path

    @property
    def path(self):
        """Full object storage path of this file."""
        return os.path.join(
            "oci://",
            self.bucket + "@" + self.namespace,
            self.filepath.lstrip("/") if self.filepath else "",
        )
