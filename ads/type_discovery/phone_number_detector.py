#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from __future__ import print_function, absolute_import, division

import re

import pandas as pd
from pandas.api.types import is_string_dtype

from ads.type_discovery import logger
from ads.type_discovery.abstract_detector import AbstractTypeDiscoveryDetector
from ads.type_discovery.typed_feature import PhoneNumberTypedFeature


class PhoneNumberDetector(AbstractTypeDiscoveryDetector):

    _pattern_string = r"^(\+?\d{1,2}[\s-])?\(?(\d{3})\)?[\s.-]?\d{3}[\s.-]?\d{4}$"

    def __init__(self):
        self.cc = re.compile(PhoneNumberDetector._pattern_string, re.VERBOSE)

    def is_phone_number(self, name, values):
        cc = re.compile(PhoneNumberDetector._pattern_string, re.VERBOSE)
        return all([self.cc.match(str(x)) for x in values])

    def discover(self, name, series):
        if is_string_dtype(series):
            candidates = series.loc[~series.isnull()]

            if self.is_phone_number(name, candidates.head(1000)):
                logger.debug("column [{}]/[{}] phone number".format(name, series.dtype))
                return PhoneNumberTypedFeature.build(name, series)

        return False


if __name__ == "__main__":
    dd = PhoneNumberDetector()

    # test the postive case:
    test_series_1 = [
        "408-456-7890",
        "(123) 456-7890",
        "650 456 7890",
        "650.456.7890",
        "+91 (123) 456-7890",
    ]

    print(dd.discover("test_series_1", pd.Series(test_series_1)))

    # test the negative case
    test_series_2 = ["1234567890", "1234567890", "123456890", "1234567890"]

    print(dd.discover("test_series_2", pd.Series(test_series_2)))

    test_series_3 = ["1-640-124-5367", "1-573-916-4412"]

    print(dd.discover("test_series_3", pd.Series(test_series_3)))
