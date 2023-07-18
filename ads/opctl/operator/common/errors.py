#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator import __operators__


class OperatorNotFoundError(Exception):
    def __init__(self, operator: str):
        super().__init__(
            f"The provided operator: `{operator}` "
            f"is not found. Available operators: {__operators__}"
        )
