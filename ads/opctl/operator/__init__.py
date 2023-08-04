#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

__operators__ = ["forecast", "responsible_ai"]


class OperatorNotFoundError(Exception):
    def __init__(self, operator: str):
        super().__init__(
            f"Operator: `{operator}` "
            f"is not registered. Available operators: {__operators__}"
        )
