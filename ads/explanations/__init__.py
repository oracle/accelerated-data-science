#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib
import logging

logger = logging.getLogger(__name__)

_LAZY_ATTRS = {
    "MLXGlobalExplainer": ("ads.explanations.mlx_global_explainer", "MLXGlobalExplainer"),
    "MLXLocalExplainer": ("ads.explanations.mlx_local_explainer", "MLXLocalExplainer"),
    "MLXWhatIfExplainer": ("ads.explanations.mlx_whatif_explainer", "MLXWhatIfExplainer"),
}

__all__ = ["MLXGlobalExplainer", "MLXLocalExplainer", "MLXWhatIfExplainer", "logger"]


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
