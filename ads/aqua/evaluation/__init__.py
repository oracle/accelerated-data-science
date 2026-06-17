#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib

__all__ = ["AquaEvaluationApp"]


def __getattr__(name):
    if name == "AquaEvaluationApp":
        value = getattr(importlib.import_module("ads.aqua.evaluation.evaluation"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
