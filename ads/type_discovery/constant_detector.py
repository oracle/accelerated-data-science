#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import, division

import pandas as pd

from ads.type_discovery import logger
from ads.type_discovery.abstract_detector import AbstractTypeDiscoveryDetector
from ads.type_discovery.typed_feature import ConstantTypedFeature


class ConstantDetector(AbstractTypeDiscoveryDetector):
    def is_constant(self, name, values):
        #
        # if all the values are null we treat this as a const feature
        #

        return values.size == 0 or values.nunique() == 1

    def discover(self, name, series):
        candidates = series.loc[~series.isnull()]

        if self.is_constant(name, candidates):
            logger.debug("column [{}]/[{}] Constant".format(name, series.dtype))
            return ConstantTypedFeature.build(name, series)

        return False


if __name__ == "__main__":
    dd = ConstantDetector()
    print(dd.discover("zipcodes", pd.Series([None, "94065", "94065", "94065", None])))
    print(dd.discover("years", pd.Series([2008, 2008, 2008, 2008, 2008])))
    df = pd.read_csv(
        "https://artifacthub.oraclecorp.com/dsc-generic/advanced-ds/datasets/flights.csv"
    )
    print(dd.discover("flights_years", df["Year"]))
