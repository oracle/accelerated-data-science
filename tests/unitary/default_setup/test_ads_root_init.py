#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock


def test_import_ads_does_not_eagerly_load_plotting_runtime():
    script = textwrap.dedent(
        """
        import sys
        import ads

        assert "matplotlib.font_manager" not in sys.modules
        print("ok")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"


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


def test_register_pandas_accessors_is_explicit_and_idempotent():
    script = textwrap.dedent(
        """
        import pandas as pd
        import ads

        assert not hasattr(pd.DataFrame, "ads")
        ads.register_pandas_accessors()
        assert hasattr(pd.DataFrame, "ads")
        assert hasattr(pd.Series, "ads")
        ads.register_pandas_accessors()
        print("ok")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"


def test_data_labeling_import_registers_pandas_accessors():
    script = textwrap.dedent(
        """
        import pandas as pd

        assert not hasattr(pd.DataFrame, "ads")
        from ads.data_labeling import LabeledDatasetReader  # noqa: F401

        assert hasattr(pd.DataFrame, "ads")
        print("ok")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"


def test_import_ads_does_not_expose_removed_deprecated_root_helpers():
    import ads

    assert not hasattr(ads, "set_documentation_mode")
    assert not hasattr(ads, "set_expert_mode")


def test_star_import_from_ads_root_namespace_succeeds():
    script = textwrap.dedent(
        """
        from ads import *

        assert Config is not None
        print("ok")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"
