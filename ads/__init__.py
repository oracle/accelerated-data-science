#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import absolute_import, division, print_function

import importlib
import logging
import os
import re
import sys
from pathlib import Path

from ads.pandas_accessors import register_pandas_accessors

# https://packaging.python.org/en/latest/guides/single-sourcing-package-version/#single-sourcing-the-package-version
if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

def _read_local_version():
    """Fallback to the checked-out source tree when package metadata is unavailable."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        match = re.search(
            r'(?m)^version\s*=\s*"([^"]+)"\s*$',
            pyproject.read_text(encoding="utf-8"),
        )
    except OSError:
        match = None
    return match.group(1) if match else "0+unknown"


def _resolve_version():
    try:
        return metadata.version("oracle_ads")
    except metadata.PackageNotFoundError:
        return _read_local_version()


__version__ = _resolve_version()

_LAZY_ATTRS = {
    "auth": ("ads.common.auth", None),
    "set_auth": ("ads.common.auth", "set_auth"),
    "Config": ("ads.common.config", "Config"),
    "deprecated": ("ads.common.decorator.deprecate", "deprecated"),
}

__all__ = [
    "__version__",
    "Config",
    "auth",
    "debug_mode",
    "deprecated",
    "documentation_mode",
    "getLogger",
    "hello",
    "logger",
    "orig_ipython_traceback",
    "register_pandas_accessors",
    "resource_principal_mode",
    "set_auth",
    "set_debug_mode",
    "test_mode",
]


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


os.environ["GIT_PYTHON_REFRESH"] = "quiet"


debug_mode = os.environ.get("DEBUG_MODE", False)
documentation_mode = os.environ.get("DOCUMENTATION_MODE", "False") == "True"
test_mode = os.environ.get("TEST_MODE", False)
resource_principal_mode = bool(
    os.environ.get("RESOURCE_PRINCIPAL_MODE", False)
)  # deprecated with is_resource_principal_mode() from ads.common.utils
orig_ipython_traceback = None


def getLogger(name="ads"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    return logger


logger = getLogger(__name__)
logger.addHandler(logging.NullHandler())

# All warnings are logged by default
logging.captureWarnings(True)


def set_debug_mode(mode=True):
    """
    Enable/disable printing stack traces on notebook.

    Parameters
    ----------
    mode: bool (default True)
         Enable/disable print stack traces on notebook

    """
    global debug_mode
    debug_mode = mode
    import IPython
    from ads.common.ipython import _log_traceback, orig_ipython_traceback as ipy_orig

    if debug_mode:
        global orig_ipython_traceback
        orig_ipython_traceback = ipy_orig
        IPython.core.interactiveshell.InteractiveShell.showtraceback = (
            orig_ipython_traceback
        )
    else:
        IPython.core.interactiveshell.InteractiveShell.showtraceback = _log_traceback

#
# ***FOR TESTING PURPOSE ONLY***
#
def _set_test_mode(mode=False):
    """
    Enable/disable intercept the automl call and rewrite it to always use
    the two algorithms (LogisticRegression for classification and LinearRegression for regression).
    Enable only during tests to reduce nb convert notebook tests run time.

    Parameters
    ----------
    mode: bool (default False)
         Enable/disable the ability to intercept automl call

    """
    global test_mode
    test_mode = mode


def hello():
    import oci
    import ocifs

    print(
        f"""

  O  o-o   o-o
 / \\ |  \\ |
o---o|   O o-o
|   ||  /     |
o   oo-o  o--o

ads v{__version__}
oci v{oci.__version__}
ocifs v{ocifs.__version__}

"""
    )
