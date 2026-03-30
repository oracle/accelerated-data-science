#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib

__all__ = ["AquaDeploymentApp"]


def __getattr__(name):
    if name == "AquaDeploymentApp":
        value = getattr(importlib.import_module("ads.aqua.modeldeployment.deployment"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
