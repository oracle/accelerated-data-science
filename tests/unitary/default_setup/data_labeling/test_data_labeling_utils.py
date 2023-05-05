#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import patch
import pytest
from ads.common.object_storage_details import ObjectStorageDetails


class TestObjectStorageDetails:
    @pytest.mark.parametrize(
        "bucket, namespace, prefix",
        [
            ("test_bucket", "test_namespace", "/prefix/"),
            ("test_bucket", "test_namespace", "prefix/"),
            ("test_bucket", "test_namespace", "///prefix/"),
        ],
    )
    @patch("ads.common.auth.default_signer")
    def test_path(self, mock_signer, bucket, namespace, prefix):
        path = ObjectStorageDetails(bucket, namespace, prefix).path
        assert path == "oci://test_bucket@test_namespace/prefix/"
