import sys
import types
from abc import ABC

from ads.opctl.backend.marketplace.marketplace_operator_interface import (
    MarketplaceInterface,
)


class MarketplaceOperatorRunner(MarketplaceInterface, ABC):
    def run(self, *argv):
        argv = argv[0]
        func: types.MethodType = getattr(self, argv[0])
        if type(func) == types.MethodType:
            sys.runpy_result = func(*argv[1:])
        print(argv)
