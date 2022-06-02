#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from .abstract_operator_spec import V0Spec, DistributedV1Spec
import os
from ads.opctl.utils import suppress_traceback
import yaml


class OperatorSpecFactory:
    """
    Factory class for creating provider instance.
    """

    spec = {
        "distributed": {
            "v1": DistributedV1Spec,
        },
        "unknown": {
            "v0": V0Spec,
        },
    }

    @staticmethod
    def load_spec(yaml_file, *args, **kwargs):
        debug = kwargs["debug"]
        spec = _ingest_yaml_file(yaml_file, debug)
        kind = spec.get("kind", "unknown")
        assert (
            kind in spec.keys()
        ), f"Kind {kind} is unknown. Kind should be one of {spec.keys()}"
        version = spec.get("version", "v0")
        assert (
            version in spec[kind].keys()
        ), f"Verion {version} is unknown. Kind should be one of {spec[kind].keys()}"
        return OperatorSpecFactory.spec[kind][version](spec, *args, **kwargs)


def _ingest_yaml_file(yaml_file, debug=False):
    if yaml_file:
        if os.path.exists(yaml_file):
            with open(yaml_file, "r") as f:
                spec = suppress_traceback(debug)(yaml.safe_load)(f.read())
        else:
            raise FileNotFoundError(f"{yaml_file} is not found")
    else:
        spec = {}
    return spec
