#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from __future__ import print_function, absolute_import, division

import pandas as pd

from ads.type_discovery import logger
from ads.type_discovery.abstract_detector import AbstractTypeDiscoveryDetector
from ads.type_discovery.typed_feature import ContinuousTypedFeature
from ads.common import utils


class ContinuousDetector(AbstractTypeDiscoveryDetector):
    @staticmethod
    def _target_is_continuous(series):
        if str(series.dtype) in ["float16", "float32", "float64"]:
            return True  # treat target variable as continuous
        elif str(series.dtype) in ["int16", "int32", "int64"]:
            if series.nunique() >= 20:
                return True  # treat target variable as continuous

        return False

    def _is_continuous(self, series):
        if series.dtype.name in ["object"]:
            try:
                series.astype("float")
                return True
            except:
                pass

        if series.dtype.name in utils.numeric_pandas_dtypes():
            #
            # if the type is float we simply beleive pandas and go with continuous
            #
            return True

    def discover(self, name, series):

        if self._is_continuous(series):
            logger.debug("column [{}]/[{}] continuous".format(name, series.dtype))
            return ContinuousTypedFeature.build(name, series)

        return False


if __name__ == "__main__":
    dd = ContinuousDetector()
    print(dd.discover("continuous", pd.Series([None, 3.14, 12.0, 1, 2, 3, None])))
