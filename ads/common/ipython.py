#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import sys
from ads.common import logger


def _log_traceback(self, exc_tuple=None, **kwargs):
    try:
        etype, value, tb = self._get_exc_info(exc_tuple)
    except ValueError:
        print("No traceback available to show.", file=sys.stderr)
        return
    msg = etype.__name__, str(value)
    logger.error("ADS Exception", exc_info=(etype, value, tb))
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
