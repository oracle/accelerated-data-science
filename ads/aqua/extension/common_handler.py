#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from importlib import metadata

import requests

from ads.aqua import ODSC_MODEL_COMPARTMENT_OCID
from ads.aqua.decorator import handle_exceptions
from ads.aqua.exception import AquaResourceAccessError
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.utils import fetch_service_compartment, known_realm


class ADSVersionHandler(AquaAPIhandler):
    """The handler to get the current version of the ADS."""

    @handle_exceptions
    def get(self):
        self.finish({"data": metadata.version("oracle_ads")})


class CompatibilityCheckHandler(AquaAPIhandler):
    """The handler to check if the extension is compatible."""

    @handle_exceptions
    def get(self):
        """This method provides the availability status of Aqua. If ODSC_MODEL_COMPARTMENT_OCID environment variable
        is set, then status `ok` is returned. For regions where Aqua is available but the environment variable is not
        set due to accesses/policies, we return the `compatible` status to indicate that the extension can be enabled
        for the selected notebook session.

        Returns
        -------
        status dict:
            ok or compatible

        Raises
        ------
            AquaResourceAccessError: raised when aqua is not accessible in the given session/region.

        """
        if ODSC_MODEL_COMPARTMENT_OCID or fetch_service_compartment():
            return self.finish(dict(status="ok"))
        elif known_realm():
            return self.finish(dict(status="compatible"))
        else:
            raise AquaResourceAccessError(
                f"The AI Quick actions extension is not compatible in the given region."
            )


class NetworkStatusHandler(AquaAPIhandler):
    """Handler to check internet connection."""

    @handle_exceptions
    def get(self):
        requests.get("https://huggingface.com", timeout=0.5)
        return self.finish("success")


__handlers__ = [
    ("ads_version", ADSVersionHandler),
    ("hello", CompatibilityCheckHandler),
    ("network_status", NetworkStatusHandler),
]
