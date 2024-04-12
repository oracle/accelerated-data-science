#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import sys
import types
from abc import ABC

from ads.common.auth import AuthState, default_signer

from ads.opctl.backend.marketplace.marketplace_operator_interface import (
    MarketplaceInterface,
)


class MarketplaceOperatorRunner(MarketplaceInterface, ABC):
    def run(self, *argv):
        argv = argv[0]
        AuthState().__dict__.update(argv[0].__dict__)
        func: types.MethodType = getattr(self, argv[1])
        if type(func) == types.MethodType:
            sys.runpy_result = func(*argv[2:])
        # print(argv)
