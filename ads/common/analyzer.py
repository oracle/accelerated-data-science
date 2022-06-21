#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.decorator.deprecate import deprecated
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)

try:
    from dask.diagnostics import ResourceProfiler
except ImportError as e:
    raise ModuleNotFoundError("Install dask to use the analyzer class.") from e

try:
    from IPython import get_ipython

    if get_ipython():
        try:
            from bokeh.io import output_notebook
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"The `bokeh` module was not found. Please run "
                f"`pip install {OptionalDependency.VIZ}`."
            )
        except Exception as e:
            raise

        output_notebook()

except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `IPython` module was not found. Please run "
        f"`pip install {OptionalDependency.NOTEBOOK}`."
    )


@deprecated("2.5.2")
def resource_analyze(dask_fun):
    """
    Profiles Dask operation and renders profiler output as a bokeh graph.

    >>> from ads.common.analyzer import resource_analyze
    >>> from ads.dataset.factory import DatasetFactory
    >>> @resource_analyze
    >>> def fetch_data():
    >>>     ds = DatasetFactory.open("/home/datascience/ads-examples/oracle_data/orcl_attrition.csv", target="Attrition").set_positive_class('Yes')
    >>>     return ds
    >>> fetch_data()

    E.g Output:

    A graph showing CPU and memory utilzation
    """

    def analyzer(*args, **kwargs):
        with ResourceProfiler(dt=0.25) as rProf:
            output = dask_fun(*args, **kwargs)
        rProf.visualize()
        return output

    return analyzer
