#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib
import logging

logger = logging.getLogger(__name__)

__all__ = ["auth", "logger"]


def __getattr__(name):
    if name == "auth":
        module = importlib.import_module("ads.common.auth")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
