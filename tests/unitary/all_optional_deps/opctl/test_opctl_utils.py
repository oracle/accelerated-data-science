#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import oci
import pytest

from ads.opctl.constants import DEFAULT_OCI_CONFIG_FILE, DEFAULT_PROFILE
from ads.opctl.utils import (
    parse_conda_uri,
    get_oci_region,
)
from ads.common.auth import AuthType, create_signer


class TestOpctlUtils:
    def test_parse_conda_uri(self):
        uri = "oci://bucket@namespace/path/to/pack_slug"
        ns, bucket, path, slug = parse_conda_uri(uri)
        assert (
            ns == "namespace"
            and bucket == "bucket"
            and path == "path/to/pack_slug"
            and slug == "pack_slug"
        )

    @pytest.fixture(scope="class")
    def oci_auth(self):
        return create_signer(AuthType.API_KEY, DEFAULT_OCI_CONFIG_FILE, DEFAULT_PROFILE)

    def test_get_oci_region(self, oci_auth):
        config_from_file = oci.config.from_file(
            DEFAULT_OCI_CONFIG_FILE, DEFAULT_PROFILE
        )
        assert get_oci_region(oci_auth) == config_from_file["region"]
