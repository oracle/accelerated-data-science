#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, division, absolute_import
import os
import logging
import json
from typing import Callable, Dict, Optional, Union

__version__ = ""
with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ads_version.json")
) as version_file:
    __version__ = json.load(version_file)["version"]
import oci

import matplotlib.font_manager  # causes matplotlib to regenerate its fonts

import ocifs
from ads.common.decorator.deprecate import deprecated
from ads.common.ipython import configure_plotting, _log_traceback
from ads.feature_engineering.accessor.series_accessor import ADSSeriesAccessor
from ads.feature_engineering.accessor.dataframe_accessor import ADSDataFrameAccessor
from ads.common import auth
from ads.common.auth import set_auth

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

    if debug_mode:
        from ads.common.ipython import orig_ipython_traceback

        IPython.core.interactiveshell.InteractiveShell.showtraceback = (
            orig_ipython_traceback
        )
    else:
        IPython.core.interactiveshell.InteractiveShell.showtraceback = _log_traceback


@deprecated("2.3.1")
def set_documentation_mode(mode=False):
    """
    This method is deprecated and will be removed in future releases.
    Enable/disable printing user tips on notebook.

    Parameters
    ----------
    mode: bool (default False)
        Enable/disable print user tips on notebook
    """
    global documentation_mode
    documentation_mode = mode


@deprecated("2.3.1")
def set_expert_mode():
    """
    This method is deprecated and will be removed in future releases.
    Enables the debug and documentation mode for expert users all in one method.
    """
    set_debug_mode(True)
    set_documentation_mode(False)


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
    """
    Imports Pandas, sets the documentation mode, and prints a fancy "Hello".
    """
    import pandas

    global documentation_mode
    global debug_mode

    print(
        f"""

  O  o-o   o-o
 / \\ |  \\ |
o---o|   O o-o
|   ||  /     |
o   oo-o  o--o

ADS SDK version: {__version__}
Pandas version: {pandas.__version__}
Debug mode: {debug_mode}
"""
    )


configure_plotting()
