#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from __future__ import print_function, absolute_import, division

import pandas as pd

from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.type_discovery import logger
from ads.type_discovery.abstract_detector import AbstractTypeDiscoveryDetector
from ads.type_discovery.typed_feature import DateTimeTypedFeature


class DateTimeDetector(AbstractTypeDiscoveryDetector):
    @runtime_dependency(module="datefinder", install_from=OptionalDependency.DATA)
    def _is_date_time(self, name, values, low_level_type_name):
        if low_level_type_name.startswith("datetime64"):
            return lambda x: x
        else:
            #
            # if the column/feature contains the word "timestamp" then
            #
            if low_level_type_name.startswith("int") and "timestamp" in name.lower():
                # either s (max len 10) on ns (max len) 19
                unit = "s" if values.astype("str").str.len().max() <= 10 else "ns"
                try:
                    pd.to_datetime(values, unit=unit)
                    return lambda x: pd.to_datetime(x, unit=unit)
                except:
                    pass
            if values.dtype == "object":
                try:
                    pd.to_datetime(values, infer_datetime_format=True)
                    datefinder_result = all(
                        [bool(list(datefinder.find_dates(str(x)))) for x in values]
                    )
                    if datefinder_result:
                        return lambda x: pd.to_datetime(x, infer_datetime_format=True)
                except:
                    pass

        return None

    def discover(self, name, series):
        candidates = series.loc[~series.isnull()]
        fn = self._is_date_time(name, candidates.head(500), series.dtype.name)
        if fn:
            logger.debug("column [{}]/[{}] datetime".format(name, series.dtype))
            return DateTimeTypedFeature.build(name, fn(series))

        return False


if __name__ == "__main__":
    dd = DateTimeDetector()
    print(
        dd.discover(
            "date-range",
            pd.Series(pd.date_range(start="1/1/2018", end="1/08/2018", freq="H")),
        )
    )
    print(dd.discover("dates", pd.Series(["12/12/12", "12/12/13", None, "12/12/14"])))
    print(
        dd.discover(
            "dates-with-other-values",
            pd.Series(["12/12/12", "Monday", None, "12/12/14"]),
        )
    )
    print(
        dd.discover(
            "timestamp s",
            pd.Series(
                [
                    978300760,
                    978302109,
                    978301968,
                    978300275,
                    978824291,
                    978302268,
                    978302039,
                ]
            ),
        )
    )
    print(dd.discover("timestamp ns", pd.Series([1490195805433502912])))
