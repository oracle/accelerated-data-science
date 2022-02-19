#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from __future__ import print_function, absolute_import, division

import pandas as pd

from ads.type_discovery import logger
from ads.type_discovery.abstract_detector import AbstractTypeDiscoveryDetector
from ads.type_discovery.typed_feature import (
    UnknownTypedFeature,
    CategoricalTypedFeature,
)


class UnknownDetector(AbstractTypeDiscoveryDetector):
    def discover(self, name, series):
        candidate = series.loc[~series.isnull()].iloc[0]

        if series.dtype == "object":
            #
            # if we got all through all the other detectors and it's a string type of feature then we
            # just call it a high dimensional categorical
            #
            return CategoricalTypedFeature.build(name, series)
        else:
            logger.debug(
                "type discovery on column [{}]/[{}] result is Unknown".format(
                    name, series.dtype
                )
            )
            return UnknownTypedFeature.build(name, series)


if __name__ == "__main__":
    dd = UnknownDetector()
    print(dd.discover("unknown", pd.Series([None, "94065", "90210", None])))
