#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from __future__ import print_function, absolute_import, division

import re

import pandas as pd

from ads.type_discovery import logger
from ads.type_discovery.abstract_detector import AbstractTypeDiscoveryDetector
from ads.type_discovery.typed_feature import ZipcodeTypedFeature


class ZipCodeDetector(AbstractTypeDiscoveryDetector):
    def _is_zip_code(self, values):
        return all(
            [re.match("^[0-9]{5}(?:-[0-9]{4})?$", str(v)) for v in values.head(10)]
        )

    def discover(self, name, series):
        candidates = series.loc[~series.isnull()]
        if (
            self._is_zip_code(candidates.head(1000))
            if candidates.dtype == "object"
            else self._is_zip_code(candidates.astype("object"))
        ):
            logger.debug(
                "type discovery on column [{}]/[{}] found to be a zipcode".format(
                    name, series.dtype
                )
            )
            return ZipcodeTypedFeature.build(name, series)

        return False


if __name__ == "__main__":
    dd = ZipCodeDetector()
    print(
        dd.discover(
            "zip", pd.Series([None, "00501", "94065", "94065-1107", "90210", None])
        )
    )
