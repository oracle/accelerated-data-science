#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from importlib import metadata

from ads.aqua.extension.base_handler import AquaAPIhandler


class ADSVersionHandler(AquaAPIhandler):
    """The handler to get the current version of the ADS."""

    def get(self):
        self.finish({"data": metadata.version("oracle_ads")})


__handlers__ = [
    ("ads_version", ADSVersionHandler),
]
