#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib
import re
import sys
from pathlib import Path
from unittest import mock


def test_import_ads_does_not_eagerly_load_plotting_runtime():
    import ads

    assert "matplotlib.font_manager" not in sys.modules


def test_import_ads_uses_pyproject_version_when_package_metadata_is_unavailable():
    import ads

    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    expected = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"\s*$',
        pyproject.read_text(encoding="utf-8"),
    ).group(1)

    with mock.patch(
        "importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError,
    ):
        reloaded = importlib.reload(ads)

    assert reloaded.__version__ == expected
    importlib.reload(ads)
