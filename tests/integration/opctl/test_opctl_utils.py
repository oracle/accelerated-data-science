#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import tempfile
import os

from ads.opctl.constants import DEFAULT_OCI_CONFIG_FILE, DEFAULT_PROFILE
from ads.opctl.utils import (
    get_region_key,
    get_namespace,
)
from ads.opctl.config.utils import convert_notebook
from ads.common.auth import AuthType, create_signer, default_signer
from tests.integration.config import secrets

import pytest
import fsspec


class TestOpctlUtils:
    @pytest.fixture(scope="class")
    def oci_auth(self):
        return create_signer(AuthType.API_KEY, DEFAULT_OCI_CONFIG_FILE, DEFAULT_PROFILE)

    def test_get_regional_key(self, oci_auth):
        assert get_region_key(oci_auth) == "IAD"

    def test_get_namespace(self, oci_auth):
        assert get_namespace(oci_auth) == secrets.common.NAMESPACE

    def test_convert_notebook(self):
        with tempfile.TemporaryDirectory() as td:
            curr_folder = os.path.dirname(os.path.abspath(__file__))
            input_path = os.path.join(
                curr_folder,
                "..",
                "fixtures",
                "exclude_check.ipynb",
            )
            convert_notebook(
                input_path,
                {},
                ["ignore", "remove"],
                output_path=os.path.join(td, "exclude_check.py"),
                overwrite=True,
            )
            with open(os.path.join(td, "exclude_check.py")) as f:
                content = f.read()
            assert 'print("ignore")' not in content
            assert 'c = 4\n"ignore"' not in content

            with pytest.raises(FileExistsError):
                convert_notebook(
                    input_path,
                    {},
                    ["ignore", "remove"],
                    output_path=os.path.join(td, "exclude_check.py"),
                )

            convert_notebook(
                input_path,
                default_signer(),
                ["ignore", "remove"],
                output_path=f"oci://ADS_INT_TEST@{secrets.common.NAMESPACE}/tests/exclude_check.py",
                overwrite=True,
            )
            file_system_clz = fsspec.get_filesystem_class("oci")
            file_system = file_system_clz(**default_signer())
            assert file_system.exists(
                f"oci://ADS_INT_TEST@{secrets.common.NAMESPACE}/tests/exclude_check.py"
            )

        input_path = f"oci://ADS_INT_TEST@{secrets.common.NAMESPACE}/opctl_test_files/exclude_check.ipynb"
        convert_notebook(
            input_path, default_signer(), ["ignore", "remove"], overwrite=True
        )
        file_system_clz = fsspec.get_filesystem_class("oci")
        file_system = file_system_clz(**default_signer())
        assert file_system.exists(
            f"oci://ADS_INT_TEST@{secrets.common.NAMESPACE}/opctl_test_files/exclude_check.py"
        )
        with pytest.raises(FileExistsError):
            convert_notebook(input_path, default_signer(), ["ignore", "remove"])
