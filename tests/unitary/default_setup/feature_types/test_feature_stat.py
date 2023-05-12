#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import load_iris

from ads.feature_engineering.feature_type_manager import (
    FeatureTypeManager as feature_type_manager,
)
from ads.feature_engineering.feature_type.string import String


class TestFeatureStat:
    ### CreditCard
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

    ### continuous
    cts = pd.Series([123.32, 23.243, 324.342, np.nan], name="cts")

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

    ### Phone Number
    phonenumber = pd.Series(
        ["2068866666", "6508866666", "2068866666", "", np.NaN, np.nan, None],
        name="phone",
    )

    ### Lat Long
    latlong = pd.Series(
        [
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
            "",
            np.NaN,
            None,
        ],
        name="latlon",
    )
    ### zip code
    zipcode = pd.Series([94065, 90210, np.NaN, None], name="zipcode")

    ### boolean
    boolean = pd.Series([True, False, True, False, np.NaN, None], name="bool")

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

    ### Address
    address = pd.Series(
        [
            "1 Miller Drive, New York, NY 12345",
            "1 Berkeley Street, Boston, MA 67891",
            "54305 Oxford Street, Seattle, WA 95132",
            "",
        ],
        name="address",
    )

    ### constant
    constant = pd.Series([1, 1, 1, 1, 1], name="constant")

    ### discrete
    discrete_numbers = pd.Series([35, 25, 13, 42], name="discrete")

    ### gis
    gis = pd.Series(
        [
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
            "",
            np.NaN,
            None,
        ],
        name="gis",
    )

    ### ipaddress
    ip_address = pd.Series(
        ["2002:db8::", "192.168.0.1", "2001:db8::", "2002:db8::", np.NaN, None],
        name="ip_address",
    )

    ### ipaddressv4
    ip_address_v4 = pd.Series(
        ["192.168.0.1", "192.168.0.2", "192.168.0.3", "192.168.0.4", np.NaN, None],
        name="ip_address_v4",
    )

    ### ipaddressv6
    ip_address_v6 = pd.Series(
        ["2002:db8::", "2001:db8::", "2001:db8::", "2002:db8::", np.NaN, None],
        name="ip_address_v6",
    )

    def test_credit_card(self):
        self.creditcard.ads.feature_type = ["credit_card"]
        stat = self.creditcard.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == len(
            self.creditcard
        )
        assert stat.loc[stat["Metric"] == "missing"]["Value"].iloc[0] == 1
        assert stat.loc[stat["Metric"] == "count_Visa"]["Value"].iloc[0] == 5
        assert stat.loc[stat["Metric"] == "count_Amex"]["Value"].iloc[0] == 5
        assert stat.loc[stat["Metric"] == "count_MasterCard"]["Value"].iloc[0] == 3
        assert stat.loc[stat["Metric"] == "count_Diners Club"]["Value"].iloc[0] == 2
        assert stat.loc[stat["Metric"] == "count_Unknown"]["Value"].iloc[0] == 1

        class CustomizeCreditCard(String):
            pass

        feature_type_manager.feature_type_register(CustomizeCreditCard)
        self.creditcard.ads.feature_type = ["customize_credit_card"]
        stat1 = self.creditcard.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 17
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 16
        assert stat.loc[stat["Metric"] == "missing"]["Value"].iloc[0] == 1

    def test_continuous(self):
        self.cts.ads.feature_type = ["continuous"]
        stat = self.cts.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 4

    def test_datetime(self):
        self.datetime.ads.feature_type = ["date_time"]
        stat = self.datetime.ads.feature_stat()
        assert (
            stat.loc[stat["Metric"] == "sample maximum"]["Value"].iloc[0]
            == "April/15/11"
        )
        assert (
            stat.loc[stat["Metric"] == "sample minimum"]["Value"].iloc[0] == "3/11/2000"
        )

    def test_phone_number(self):
        self.phonenumber.ads.feature_type = ["phone_number", "category"]
        stat = self.phonenumber.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 7
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 2
        assert stat.loc[stat["Metric"] == "missing"]["Value"].iloc[0] == 4

    def test_lat_long(self):
        self.latlong.ads.feature_type = ["lat_long"]
        stat = self.latlong.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 13
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 10
        assert stat.loc[stat["Metric"] == "missing"]["Value"].iloc[0] == 3

    def test_zipcode(self):
        self.zipcode.ads.feature_type = ["zip_code"]
        stat = self.zipcode.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 4
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 2
        assert stat.loc[stat["Metric"] == "missing"]["Value"].iloc[0] == 2

    @pytest.mark.parametrize("feature_type", ["boolean", "category", "ordinal"])
    def test_categorical_bool_ordinal(self, feature_type):
        self.boolean.ads.feature_type = [feature_type]
        stat = self.boolean.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 6
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 2
        assert stat.loc[stat["Metric"] == "missing"]["Value"].iloc[0] == 2

    def test_string(self):
        self.string.ads.feature_type = ["string"]
        stat = self.string.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 22
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 3
        assert stat.loc[stat["Metric"] == "missing"]["Value"].iloc[0] == 3

    def test_address(self):
        self.address.ads.feature_type = ["address"]
        stat = self.address.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 4
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 3
        assert stat.loc[stat["Metric"] == "missing"]["Value"].iloc[0] == 1

    def test_constant(self):
        self.constant.ads.feature_type = ["constant"]
        stat = self.constant.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 5
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 1

    def test_discrete(self):
        self.discrete_numbers.ads.feature_type = ["discrete"]
        stat = self.discrete_numbers.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 4
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 4

    def test_gis(self):
        self.gis.ads.feature_type = ["gis"]
        stat = self.gis.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 13
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 10
        assert stat.loc[stat["Metric"] == "missing"]["Value"].iloc[0] == 3

    def test_ipaddress(self):
        self.ip_address.ads.feature_type = ["ip_address"]
        stat = self.ip_address.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 6
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 3
        assert stat.loc[stat["Metric"] == "missing"]["Value"].iloc[0] == 2

    def test_ipaddress_v4(self):
        self.ip_address_v4.ads.feature_type = ["ip_address_v4"]
        stat = self.ip_address_v4.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 6
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 4
        assert stat.loc[stat["Metric"] == "missing"]["Value"].iloc[0] == 2

    def test_ipaddress_v6(self):
        self.ip_address_v6.ads.feature_type = ["ip_address_v6"]
        stat = self.ip_address_v6.ads.feature_stat()
        assert stat.loc[stat["Metric"] == "count"]["Value"].iloc[0] == 6
        assert stat.loc[stat["Metric"] == "unique"]["Value"].iloc[0] == 2
        assert stat.loc[stat["Metric"] == "missing"]["Value"].iloc[0] == 2

    def test_dataframe_stat(self):
        df = load_iris(as_frame=True).data
        stat = df.ads.feature_stat()
        assert set(df.columns).issubset(set(stat.reset_index().Column.unique()))
