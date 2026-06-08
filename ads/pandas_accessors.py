#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Bootstrap helpers for ADS pandas accessors."""

import importlib

_PANDAS_ACCESSOR_MODULES = (
    "ads.feature_engineering.accessor.series_accessor",
    "ads.feature_engineering.accessor.dataframe_accessor",
)
_BOOTSTRAPPING = False
_REGISTERED = False


def register_pandas_accessors():
    """Register ADS pandas accessors for the current Python process."""

    global _BOOTSTRAPPING
    global _REGISTERED

    if _REGISTERED or _BOOTSTRAPPING:
        return

    _BOOTSTRAPPING = True
    try:
        for module_name in _PANDAS_ACCESSOR_MODULES:
            importlib.import_module(module_name)
    finally:
        _BOOTSTRAPPING = False

    _REGISTERED = True
