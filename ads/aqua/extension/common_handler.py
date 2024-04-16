#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from importlib import metadata

from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua import ODSC_MODEL_COMPARTMENT_OCID
from ads.aqua.exception import AquaResourceAccessError
from ads.aqua.utils import known_realm
from ads.aqua.decorator import handle_exceptions


class ADSVersionHandler(AquaAPIhandler):
    """The handler to get the current version of the ADS."""

    def get(self):
        self.finish({"data": metadata.version("oracle_ads")})


class CompatibilityCheckHandler(AquaAPIhandler):
    """The handler to check if the extension is compatible."""

    @handle_exceptions
    def get(self):
        if ODSC_MODEL_COMPARTMENT_OCID:
            return self.finish(dict(status="ok"))
        elif known_realm():
            return self.finish(dict(status="compatible"))
        else:
            raise AquaResourceAccessError(
                f"The AI Quick actions extension is not compatible in the given region."
            )


__handlers__ = [
    ("ads_version", ADSVersionHandler),
    ("hello", CompatibilityCheckHandler),
]
