import runpy
import sys
import types

from ads.opctl.backend.marketplace.marketplace_type import MarketplaceListingDetails

from ads.opctl.backend.marketplace.marketplace_operator_interface import (
    MarketplaceInterface,
)


def runpy_backend_runner(func: types.FunctionType):
    def inner_backend(runner: "MarketplaceBackendRunner", *args, **kwargs: dict):
        if kwargs:
            raise RuntimeError("Kwargs are not supported")
        else:
            sys.argv = [f"{func.__name__}", *args]
            try:
                result = runpy.run_module(runner.module_name, run_name="__main__")
            except SystemExit as exception:
                return exception.code
            else:
                return result["sys"].runpy_result

    return inner_backend


class MarketplaceBackendRunner(MarketplaceInterface):
    def __init__(self, module_name: str = None):
        self.module_name = module_name

    @runpy_backend_runner
    def get_listing_details(self, operator_config: str) -> MarketplaceListingDetails:
        pass
