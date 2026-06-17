#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib

__all__ = ["AquaModelApp"]


def __getattr__(name):
    if name == "AquaModelApp":
        value = getattr(importlib.import_module("ads.aqua.model.model"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
