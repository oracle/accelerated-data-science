#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from __future__ import print_function, absolute_import, division

import re

import pandas as pd

from ads.type_discovery import logger
from ads.type_discovery.abstract_detector import AbstractTypeDiscoveryDetector
from ads.type_discovery.typed_feature import GISTypedFeature


class LatLonDetector(AbstractTypeDiscoveryDetector):

    _pattern_string = r"^[(]?(\-?\d+\.\d+?),\s*(\-?\d+\.\d+?)[)]?$"

    def __init__(self):
        self.cc = re.compile(LatLonDetector._pattern_string, re.VERBOSE)

    def is_lat_lon(self, name, values):
        return all([self.cc.match(str(x)) for x in values])

    def discover(self, name, series):
        candidates = series.loc[~series.isnull()]

        if self.is_lat_lon(name, candidates.head(1000)):

            logger.debug("column [{}]/[{}] lat/lon".format(name, series.dtype))

            samples = [
                tuple([float(x) for x in self.cc.search(v).groups()])
                for v in candidates.sample(frac=1).head(500).values
            ]
            return GISTypedFeature.build(name, series, samples)

        return False

    @staticmethod
    def extract_x_y(gis_series):
        """takes a GIS series and parses it into a new dataframe with X (longitude) and Y (latitude) columns."""
        cc = re.compile(LatLonDetector._pattern_string, re.VERBOSE)
        lats, lons = zip(*gis_series.dropna().apply(lambda v: cc.search(v).groups()))

        return pd.DataFrame({"X": lons, "Y": lats}).astype(
            {"X": "float64", "Y": "float64"}
        )


if __name__ == "__main__":
    dd = LatLonDetector()

    test_series_1 = [
        "-18.2193965, -93.587285",
        "-21.0255305, -122.478584",
        "85.103913, 19.405744",
        "82.913736, 178.225672",
        "62.9795085, -66.989705",
        "54.5604395, 95.235090",
        "33.970775, -140.939679",
        "40.9680285, -30.369376",
        "51.816119, 175.979008",
        "-48.7882365, 84.035621",
    ]

    test_series_2 = [
        "69.196241,-125.017615",
        "5.2272595,-143.465712",
        "-33.9855425,-153.445155",
        "43.340610,86.460554",
        "24.2811855,-162.380403",
        "2.7849025,-7.328156",
        "45.033805,157.490179",
        "-1.818319,-80.681214",
        "-44.510428,-169.269477",
        "-56.3344375,-166.407038",
    ]

    test_series_3 = ["(54.3392995,-11.801615)"]

    print(dd.discover("test_series_1", pd.Series(test_series_1)))
    print(dd.discover("test_series_2", pd.Series(test_series_2)))
    print(dd.discover("test_series_3", pd.Series(test_series_3)))

    print(LatLonDetector.extract_x_y(pd.Series(test_series_2)))
