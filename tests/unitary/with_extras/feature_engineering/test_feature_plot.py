#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import matplotlib as mpl
import pandas as pd
from sklearn.datasets import load_iris

from ads.feature_engineering.utils import (
    _is_float,
    _str_lat_long_to_point,
    _plot_gis_scatter,
    _to_lat_long,
    _zip_code,
)


class TestFeaturePlot:
    ### empty
    empty_series = pd.Series([], name="empty")

    ### address
    address = pd.Series(
        [
            "1 Miller Drive, New York, NY 12345",
            "1 Berkeley Street, Boston, MA 67891",
            "54305 Oxford Street, Seattle, WA 95132",
            "",
        ],
        name="address",
    )

    address_invalid = pd.Series(
        ["1 Miller Drive, New York, NY 0987", "", None, np.NaN], name="address"
    )

    ### boolean
    boolean = pd.Series([True, False, True, False, np.NaN, None], name="bool")
    boolean_invalid = pd.Series([np.NaN, None], name="bool")

    ### constant
    constant = pd.Series([1, 1, 1, 1, 1], name="constant")
    constant_invalid = pd.Series([np.NaN, None], name="constant")

    ### continuous
    cts = pd.Series([123.32, 23.243, 324.342, np.nan], name="cts")
    cts_invalid = pd.Series(["abc", None, np.nan], name="cts")

    ### category
    cat = pd.Series(
        [
            "S",
            "C",
            "S",
            "S",
            "S",
            "Q",
            "S",
            "S",
            "S",
            "C",
            "S",
            "S",
            "S",
            "S",
            "S",
            "S",
            "Q",
            "S",
            "S",
            "",
            np.NaN,
            None,
        ],
        name="category",
    )

    ### credit card
    visa = [
        "4532640527811543",
        "4556929308150929",
        "4539944650919740",
        "4485348152450846",
        "4556593717607190",
    ]
    mastercard = [
        "5334180299390324",
        "5111466404826446",
        "5273114895302717",
        "5430972152222336",
        "5536426859893306",
    ]
    amex = [
        "371025944923273",
        "374745112042294",
        "340984902710890",
        "375767928645325",
        "370720852891659",
    ]
    missing = [np.nan]
    invalid = ["354 435"]

    creditcard = pd.Series(
        visa + mastercard + amex + missing + invalid, name="creditcard"
    )
    creditcard_invalid = pd.Series(missing + invalid, name="creditcard")

    ### datetime
    datetime = pd.Series(
        [
            "3/11/2000",
            "3/12/2000",
            "3/13/2000",
            "",
            None,
            np.nan,
            "April/13/2011",
            "April/15/11",
        ],
        name="datetime",
    )
    datetime_invalid = pd.Series(["abc", None, np.nan], name="datetime")

    ### discrete
    discrete = pd.Series([35, 25, 13, 42], name="discrete")
    discrete_invalid = pd.Series([None, np.nan], name="discrete")

    ### integer
    integer = pd.Series([1, 0, 1, 2, 3, 4, np.nan], name="integer")
    integer_invalid = pd.Series(["abc", None, np.nan], name="integer")

    ### ordinal
    ordinal = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan], name="ordinal")
    ordinal_invalid = pd.Series([None, np.nan], name="ordinal")

    ### lat long
    latlong = pd.Series(
        [
            "(69.196241,-125.017615)",
            "5.2272595,-143.465712",
            "-33.9855425,-153.445155",
            "43.340610,86.460554",
            "24.2811855,-162.380403",
            "2.7849025,-7.328156",
            "45.033805,157.490179",
            "-1.818319,-80.681214",
            "-44.510428,-169.269477",
            "-56,-166",
            "",
            np.NaN,
            None,
        ],
        name="latlon",
    )

    latlong_invalid = pd.Series(
        ["[-56.3344375,-166.407038]", "", np.NaN, None], name="latlon"
    )

    ### string
    string = pd.Series(
        [
            "S",
            "C",
            "S",
            "S",
            "S",
            "Q",
            "S",
            "S",
            "S",
            "C",
            "S",
            "S",
            "S",
            "S",
            "S",
            "S",
            "Q",
            "S",
            "S",
            "",
            np.NaN,
            None,
        ],
        name="string",
    )
    string_invalid = pd.Series([123, "", np.NaN, None], name="string")

    ### text
    text = pd.Series(
        [
            "S",
            "C",
            "S",
            "S",
            "S",
            "Q",
            "S",
            "S",
            "S",
            "C",
            "S",
            "S",
            "S",
            "S",
            "S",
            "S",
            "Q",
            "S",
            "S",
            123,
            1.5,
            np.NaN,
            None,
        ],
        name="text",
    )
    text_invalid = pd.Series(["", np.NaN, None], name="text")

    ### zip code
    zipcode = pd.Series(["94065", "90210", np.NaN, None], name="zipcode")
    zipcode_invalid = pd.Series([94065, "cat", np.NaN, None], name="zipcode")
    zipcode_coord = _zip_code()

    def test_feature_plot_return_type(self):
        # Test Series
        self.boolean.ads.feature_type = ["boolean"]
        assert isinstance(self.boolean.ads.feature_plot(), mpl.axes._axes.Axes)

        # Test DataFrame
        df = load_iris(as_frame=True).data
        plots = df.ads.feature_plot()
        assert isinstance(plots, pd.DataFrame)
        assert set(df.columns).issubset(set(plots.reset_index().Column.unique()))

    def test_address(self):
        self.empty_series.ads.feature_type = ["address"]
        self.address.ads.feature_type = ["address"]
        self.address_invalid.ads.feature_type = ["address"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.address.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.address_invalid.ads.feature_plot() == None

    def test_boolean(self):
        self.empty_series.ads.feature_type = ["boolean"]
        self.boolean.ads.feature_type = ["boolean"]
        self.boolean_invalid.ads.feature_type = ["boolean"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.boolean.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.boolean_invalid.ads.feature_plot() == None

    def test_categorical(self):
        self.empty_series.ads.feature_type = ["category"]
        self.cat.ads.feature_type = ["category"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.cat.ads.feature_plot(), mpl.axes._axes.Axes)

    def test_constant(self):
        self.empty_series.ads.feature_type = ["constant"]
        self.constant.ads.feature_type = ["constant"]
        self.constant_invalid.ads.feature_type = ["constant"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.constant.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.constant_invalid.ads.feature_plot() == None

    def test_continuous(self):
        self.empty_series.ads.feature_type = ["continuous"]
        self.cts.ads.feature_type = ["continuous"]
        self.cts_invalid.ads.feature_type = ["continuous"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.cts.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.cts_invalid.ads.feature_plot() == None

    def test_credit_card(self):
        self.empty_series.ads.feature_type = ["credit_card"]
        self.creditcard.ads.feature_type = ["credit_card"]
        self.creditcard_invalid.ads.feature_type = ["credit_card"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.creditcard.ads.feature_plot(), mpl.axes._axes.Axes)
        assert isinstance(
            self.creditcard_invalid.ads.feature_plot(), mpl.axes._axes.Axes
        )

    def test_datetime(self):
        self.empty_series.ads.feature_type = ["date_time"]
        self.datetime.ads.feature_type = ["date_time"]
        self.datetime_invalid.ads.feature_type = ["date_time"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.datetime.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.datetime_invalid.ads.feature_plot() == None

    def test_discrete(self):
        self.empty_series.ads.feature_type = ["discrete"]
        self.discrete.ads.feature_type = ["discrete"]
        self.discrete_invalid.ads.feature_type = ["discrete"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.discrete.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.discrete_invalid.ads.feature_plot() == None

    def test_integer(self):
        self.empty_series.ads.feature_type = ["integer"]
        self.integer.ads.feature_type = ["integer"]
        self.integer_invalid.ads.feature_type = ["integer"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.integer.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.integer_invalid.ads.feature_plot() == None

    def test_ordinal(self):
        self.empty_series.ads.feature_type = ["ordinal"]
        self.ordinal.ads.feature_type = ["ordinal"]
        self.ordinal_invalid.ads.feature_type = ["ordinal"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.ordinal.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.ordinal_invalid.ads.feature_plot() == None

    def test_latlong(self):
        self.empty_series.ads.feature_type = ["lat_long"]
        self.latlong.ads.feature_type = ["lat_long"]
        self.latlong_invalid.ads.feature_type = ["lat_long"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.latlong.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.latlong_invalid.ads.feature_plot() == None

    def test_string(self):
        self.empty_series.ads.feature_type = ["string"]
        self.string.ads.feature_type = ["string"]
        self.string_invalid.ads.feature_type = ["string"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.string.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.string_invalid.ads.feature_plot() == None

    def test_text(self):
        self.empty_series.ads.feature_type = ["text"]
        self.text.ads.feature_type = ["text"]
        self.text_invalid.ads.feature_type = ["text"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.text.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.text_invalid.ads.feature_plot() == None

    def test_gis(self):
        self.empty_series.ads.feature_type = ["gis"]
        self.latlong.ads.feature_type = ["gis"]
        self.latlong_invalid.ads.feature_type = ["gis"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.latlong.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.latlong_invalid.ads.feature_plot() == None

    def test_zipcode(self):
        self.empty_series.ads.feature_type = ["zip_code"]
        self.zipcode.ads.feature_type = ["zip_code"]
        self.zipcode_invalid.ads.feature_type = ["zip_code"]

        assert self.empty_series.ads.feature_plot() == None
        assert isinstance(self.zipcode.ads.feature_plot(), mpl.axes._axes.Axes)
        assert self.zipcode_invalid.ads.feature_plot() == None

    def test_is_float(self):
        assert _is_float("") == False
        assert _is_float("3") == False
        assert _is_float("3.14") == True

    def test_str_lat_long_to_point(self):
        assert (
            _str_lat_long_to_point("85.103913, 19.405744")
            == "POINT(19.405744 85.103913)"
        )
        assert (
            _str_lat_long_to_point("(85.103913, 19.405744)")
            == "POINT(19.405744 85.103913)"
        )
        assert (
            _str_lat_long_to_point("(85.103913,19.405744)")
            == "POINT(19.405744 85.103913)"
        )
        assert np.isnan(_str_lat_long_to_point("85, 19"))
        assert np.isnan(_str_lat_long_to_point("[85.103913, 19.405744]"))
        assert np.isnan(_str_lat_long_to_point("cat, dog"))

    def test_to_lat_long(self):
        df_valid = _to_lat_long(self.zipcode, self.zipcode_coord)
        assert isinstance(df_valid, pd.DataFrame)
        assert len(df_valid.index) > 0
        df_invalid = _to_lat_long(self.zipcode_invalid, self.zipcode_coord)
        assert isinstance(df_invalid, pd.DataFrame)
        assert len(df_invalid.index) == 0
        df_empty = _to_lat_long(self.empty_series, self.zipcode_coord)
        assert isinstance(df_empty, pd.DataFrame)
        assert len(df_empty.index) == 0

    def test_plot_gis_scatter(self):
        df_valid = _to_lat_long(self.zipcode, self.zipcode_coord)
        df_invalid = _to_lat_long(self.zipcode_invalid, self.zipcode_coord)
        df_empty = _to_lat_long(self.empty_series, self.zipcode_coord)
        assert isinstance(
            _plot_gis_scatter(df_valid, "latitude", "longitude"),
            mpl.axes._axes.Axes,
        )
        assert _plot_gis_scatter(df_invalid, "latitude", "longitude") == None
        assert _plot_gis_scatter(df_empty, "latitude", "longitude") == None

    def test_zip_code(self):
        df_zip_code = _zip_code()
        assert isinstance(df_zip_code, pd.DataFrame)
        assert len(df_zip_code.loc["99950"]) == 2
