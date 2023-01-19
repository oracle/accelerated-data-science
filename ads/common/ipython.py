#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import sys

# TODO - Revisit this as part of ADS logging changes https://jira.oci.oraclecorp.com/browse/ODSC-36245
# Use a unique logger that we can individually configure without impacting other log statements.
# We don't want the logger name to mention "ads", since this logger will report any exception that happens in a
# notebook cell, and we don't want customers incorrectly assuming that ADS is somehow responsible for every error.
logger = logging.getLogger("ipython.traceback")
# Set propagate to False so logs aren't passed back up to the root logger handlers. There are some places in ADS
# where logging.basicConfig() is called. This changes root logger configurations. The user could import/use code that
# invokes the logging.basicConfig() function at any time, making the behavior of the root logger unpredictable.
logger.propagate = False
logger.handlers.clear()
traceback_handler = logging.StreamHandler()
traceback_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(traceback_handler)


def _log_traceback(self, exc_tuple=None, **kwargs):
    try:
        etype, value, tb = self._get_exc_info(exc_tuple)
    except ValueError:
        print("No traceback available to show.", file=sys.stderr)
        return
    msg = etype.__name__, str(value)
    # User a generic message that makes no mention of ADS.
    logger.error("Exception", exc_info=(etype, value, tb))
    sys.stderr.write("{0}: {1}".format(*msg))


def set_ipython_traceback():
    pass


def configure_plotting():
    try:
        import IPython
        from IPython import get_ipython
        from IPython.core.error import UsageError

        ipy = get_ipython()
        if ipy is not None:
            try:
                # show matplotlib plots inline
                ipy.run_line_magic("matplotlib", "inline")
            except UsageError:
                #  ignore error and use the default matplotlib mode
                pass
    except ModuleNotFoundError:
        import matplotlib as mpl

        mpl.rcParams["backend"] = "agg"
        import matplotlib.pyplot as plt

        plt.switch_backend("agg")


try:
    import IPython

    global orig_ipython_traceback
    if IPython.core.interactiveshell.InteractiveShell.showtraceback != _log_traceback:
        orig_ipython_traceback = (
            IPython.core.interactiveshell.InteractiveShell.showtraceback
        )

    # Override the default showtraceback behavior of ipython, to show only the error message and log the stacktrace
    IPython.core.interactiveshell.InteractiveShell.showtraceback = _log_traceback
except ModuleNotFoundError:
    pass
