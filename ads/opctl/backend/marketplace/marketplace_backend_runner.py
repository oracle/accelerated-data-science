#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import runpy
import sys
import types

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from kubernetes.client import V1ServiceList

from ads.common.auth import AuthState

from ads.opctl.backend.marketplace.models.marketplace_type import (
    MarketplaceListingDetails,
)

from ads.opctl.backend.marketplace.marketplace_operator_interface import (
    MarketplaceInterface,
    Status,
)


def runpy_backend_runner(func: types.FunctionType):
    def inner_backend(runner: "MarketplaceBackendRunner", *args, **kwargs: dict):
        if kwargs:
            raise RuntimeError("Kwargs are not supported")
        else:
            sys.argv = [AuthState(), f"{func.__name__}", *args]
            try:
                result = runpy.run_module(runner.module_name, run_name="__main__")
            except SystemExit as exception:
                return exception.code
            else:
                return result["sys"].runpy_result

    return inner_backend


# TODO: Handle generic listings properly
class MarketplaceBackendRunner(MarketplaceInterface):
    def __init__(self, module_name: str = None):
        self.module_name = module_name

    @runpy_backend_runner
    def get_listing_details(
        self, operator_config: str, docker_registry_secret_name: str
    ) -> MarketplaceListingDetails:
        pass

    @runpy_backend_runner
    def get_oci_meta(
        self,
        operator_config: str,
        tags_map: Dict[str, str],
    ) -> dict:
        pass

    @runpy_backend_runner
    def finalise_installation(
        self,
        operator_config: str,
        status: Status,
        tags_map: Dict[str, str],
        kubernetes_service_list: "V1ServiceList",
    ):
        pass
