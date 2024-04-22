#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os


def __registered_operators():
    """Gets the list of registered operators."""

    target_dir = os.path.join(os.path.dirname(__file__), "lowcode")
    return [
        f
        for f in os.listdir(target_dir)
        if os.path.isdir(os.path.join(target_dir, f))
        and not f.startswith("__")
        and f != "common"
    ]


__operators__ = __registered_operators()


class OperatorNotFoundError(Exception):
    def __init__(self, operator: str):
        super().__init__(
            f"Operator: `{operator}` "
            f"is not registered. Available operators: {__operators__}"
        )
