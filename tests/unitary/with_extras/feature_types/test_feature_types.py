#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from unittest.mock import MagicMock, patch

import nltk
import numpy as np
import pandas as pd
from ads.common.card_identifier import card_identify
from ads.feature_engineering.accessor.dataframe_accessor import ADSDataFrameAccessor
from ads.feature_engineering.accessor.series_accessor import ADSSeriesAccessor
from ads.feature_engineering.feature_type.address import Address
from ads.feature_engineering.feature_type.adsstring.parsers.nltk_parser import (
    NLTKParser,
)
from ads.feature_engineering.feature_type.adsstring.parsers.spacy_parser import (
    SpacyParser,
)
from ads.feature_engineering.feature_type.adsstring.string import ADSString
from ads.feature_engineering.feature_type.boolean import Boolean
from ads.feature_engineering.feature_type.continuous import Continuous
from ads.feature_engineering.feature_type.creditcard import CreditCard
from ads.feature_engineering.feature_type.datetime import DateTime
from ads.feature_engineering.feature_type.gis import GIS
from ads.feature_engineering.feature_type.integer import Integer
from ads.feature_engineering.feature_type.ip_address import IpAddress
from ads.feature_engineering.feature_type.ip_address_v4 import IpAddressV4
from ads.feature_engineering.feature_type.ip_address_v6 import IpAddressV6
from ads.feature_engineering.feature_type.lat_long import LatLong
from ads.feature_engineering.feature_type.phone_number import PhoneNumber
from ads.feature_engineering.feature_type.string import String
from ads.feature_engineering.feature_type.zip_code import ZipCode
from ads.feature_engineering.utils import is_boolean
from oci.ai_language import AIServiceLanguageClient
from ads.feature_engineering.feature_type.adsstring.oci_language import OCILanguage


class TestFeatureTypes:
    def setup_class(self):
        nltk.download("punkt", download_dir=os.environ["CONDA_PREFIX"] + "/nltk")
        nltk.download(
            "averaged_perceptron_tagger",
            download_dir=os.environ["CONDA_PREFIX"] + "/nltk",
        )

    def test_datatime_type(self):
        assert (
            DateTime.validator.is_datetime(
                pd.Series(pd.date_range(start="1/1/2018", end="1/05/2018", freq="D"))
            )
            == np.array([True, True, True, True, True])
        ).all()
        assert (
            DateTime.validator.is_datetime(
                pd.Series(["12/12/12", "12/12/13", None, "12/12/14"])
            )
            == np.array([True, True, False, True])
        ).all()

    def test_address_type(self):
        address = pd.Series(
            [
                "1 Miller Drive, New York, NY 12345",
                "1 Berkeley Street, Boston, MA 67891",
                "54305 Oxford Street, Seattle, WA 95132",
                "",
            ]
        )
        assert (
            Address.validator.is_address(address).values
            == np.array([True, True, True, False])
        ).all()

    def test_boolean_type(self):
        boolean = pd.Series([True, False, "true", "false", None])
        assert (
            Boolean.validator.is_boolean(boolean).values
            == np.array([True, True, True, True, False])
        ).all()

    def test_string_type(self):
        string = pd.Series(
            [
                "S",
                "C",
                "S",
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
        string.ads.feature_type = ["string"]
        assert (
            String.validator.is_string(string).values
            == np.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                ]
            )
        ).all()

    def test_ip_address_v4_type(self):
        ip_address = pd.Series(
            ["192.168.0.1", "2001:db8::", "", np.NaN, None], name="ip_address"
        )
        ip_address.ads.feature_type = ["ip_address_v4"]
        assert (
            IpAddressV4.validator.is_ip_address_v4(ip_address).values
            == np.array([True, False, False, False, False])
        ).all()

    def test_ip_address_v6_type(self):
        ip_address = pd.Series(
            ["192.168.0.1", "2001:db8::", "", np.NaN, None], name="ip_address"
        )
        ip_address.ads.feature_type = ["ip_address_v6"]
        assert (
            IpAddressV6.validator.is_ip_address_v6(ip_address).values
            == np.array([False, True, False, False, False])
        ).all()

    def test_ip_address_type(self):
        ip_address = pd.Series(
            ["192.168.0.1", "2001:db8::", "", np.NaN, None], name="ip_address"
        )
        ip_address.ads.feature_type = ["ip_address"]
        assert (
            IpAddress.validator.is_ip_address(ip_address).values
            == np.array([True, True, False, False, False])
        ).all()

    def test_latlong_type(self):
        test_series_1 = [
            "-18.2193965, -93.587285",
            "-21.0255305, -122.478584",
            "85.103913, 19.405744",
            "82.913736, 178.225672",
            "62.9795085,-66.989705",
            "54.5604395,95.235090",
            "24.2811855,-162.380403",
            "-1.818319,-80.681214",
            None,
            "(51.816119, 175.979008)",
            "(54.3392995,-11.801615)",
        ]

        assert LatLong.name == "lat_long"
        assert (
            LatLong.validator.is_lat_long(pd.Series(test_series_1)).values
            == np.array(
                [True, True, True, True, True, True, True, True, False, True, True]
            )
        ).all()

    def test_gis_type(self):
        test_series_1 = [
            "-18.2193965, -93.587285",
            "-21.0255305, -122.478584",
            "85.103913, 19.405744",
            "82.913736, 178.225672",
            "62.9795085,-66.989705",
            "54.5604395,95.235090",
            "24.2811855,-162.380403",
            "-1.818319,-80.681214",
            None,
            "(51.816119, 175.979008)",
            "(54.3392995,-11.801615)",
        ]

        assert GIS.name == "gis"
        assert (
            GIS.validator.is_gis(pd.Series(test_series_1)).values
            == np.array(
                [True, True, True, True, True, True, True, True, False, True, True]
            )
        ).all()

    def test_phone_number_type(self):
        test_series_1 = pd.Series(
            [
                "408-456-7890",
                "(123) 456-7890",
                "650 456 7890",
                "650.456.7890",
                "+91 (123) 456-7890",
            ]
        )

        test_series_2 = pd.Series(
            ["1234567890", "1234567890", "123456890", "1234567890"]
        )

        test_series_3 = pd.Series([None, "1-640-124-5367", "1-573-916-4412"])

        assert PhoneNumber.name == "phone_number"
        assert (
            PhoneNumber.validator.is_phone_number(test_series_1)
            == np.array([True, True, True, True, True])
        ).all()
        assert (
            PhoneNumber.validator.is_phone_number(test_series_2)
            == np.array([True, True, False, True])
        ).all()
        assert (
            PhoneNumber.validator.is_phone_number(test_series_3)
            == np.array([False, True, True])
        ).all()

    def test_zip_code_type(self):
        assert ZipCode.name == "zip_code"
        zipcode = pd.Series(["94065", 90210, np.NaN, None], name="zipcode")
        zipcode.ads.feature_type = ["zip_code"]

        assert (
            ZipCode.validator.is_zip_code(zipcode)
            == np.array([True, False, False, False])
        ).all()

    def test_credit_card_type(self):
        visa = [
            "4532640527811543",
            None,
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
        card = visa + mastercard + amex + missing + invalid
        s = pd.Series(card, name="creditcard")
        s.ads.feature_type = ["ip_address"]

        assert CreditCard.name == "credit_card"
        assert card_identify().identify_issue_network(visa[0]) == "Visa"
        assert card_identify().identify_issue_network(mastercard[2]) == "MasterCard"
        assert card_identify().identify_issue_network(amex[1]) == "Amex"
        assert (
            CreditCard.validator.is_credit_card(s).values
            == np.array(
                [
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                ]
            )
        ).all()

    def test_adsstring_type(self):
        ADSString.nlp_backend(backend="nltk")
        assert ADSString.name == "ads_string"
        # test attributes in series_accessor
        text1 = "get in touch with my associate at test.address@gmail.com to schedule"
        text2 = "she is born on Jan. 19th, 2014."
        text3 = "please follow the link www.oracle.com to homepage."
        series = pd.Series([text1, text2, text3], name="ADSString")
        series.ads.feature_type = ["ads_string", "string"]
        assert series.ads.email[0] == ["test.address@gmail.com"]
        redact_res = series.ads.redact(fields=["email", "date"])
        assert redact_res == [
            "get in touch with my associate at [EMAIL] to schedule",
            "she is born on [DATE].",
            "please follow the link www.oracle.com to homepage.",
        ]

        ADSString.plugin_register(OCILanguage)
        # test attributes in oci_language.py
        test_text1 = """
                    Lawrence Joseph Ellison (born August 17, 1944) is an American business magnate,
                    investor, and philanthropist who is a co-founder, the executive chairman and
                    chief technology officer (CTO) of Oracle Corporation. As of October 2019, he was
                    listed by Forbes magazine as the fourth-wealthiest person in the United States
                    and as the sixth-wealthiest in the world, with a fortune of $69.1 billion,
                    increased from $54.5 billion in 2018.[4] He is also the owner of the 41st
                    largest island in the United States, Lanai in the Hawaiian Islands with a
                    population of just over 3000
                """.strip()
        test_text2 = "This movie is awesome."
        series1 = pd.Series([test_text1, test_text2], name="ADSString")
        series1.ads.feature_type = ["ads_string"]

        with patch.object(
            AIServiceLanguageClient,
            "batch_detect_language_sentiments",
            return_value=MagicMock(
                data=MagicMock(documents=['{"aspects":{"sentiment":"Positive"}}'])
            ),
        ) as mock_detect_language_sentiments:
            test_res = series1.ads.absa
            assert test_res[0].get("sentiment") == "Positive"
            mock_detect_language_sentiments.assert_called()

        # test property in parsers/base.py
        test_txt1 = "Walking my dog on a breezy day is the best."
        test_txt2 = "get in touch with my associate"
        series2 = pd.Series([test_txt1, test_txt2], name="ADSString")
        series2.ads.feature_type = ["ads_string", "string"]
        nouns_res = series2.ads.noun
        assert len(nouns_res) == 2
        assert len(nouns_res[0]) == 2
        assert len(nouns_res[1]) == 3

        # test using spacy backend
        ADSString.plugin_clear()
        ADSString.nlp_backend(backend="spacy")
        assert SpacyParser in ADSString.plugins
        test_txt3 = "Walking my dog on a breezy day is the best."
        test_txt4 = "Get in touch with my associate at Museum Of Arts."
        test_txt5 = "The Indian Space Research Organisation or is the national space agency of India, headquartered in Bengaluru. It operates under Department of Space which is directly overseen by the Prime Minister of India while Chairman of ISRO acts as executive of DOS as well"
        series3 = pd.Series([test_txt3, test_txt4, test_txt5])
        series3.ads.feature_type = ["ads_string"]
        assert "ads_string" in series3.ads.feature_type
        loc_res = series3.ads.entity_location
        assert len(loc_res) == 3
        assert len(loc_res[0]) == 0
        assert len(loc_res[1]) == 0
        assert len(loc_res[2]) == 2

        ADSString.plugin_clear()
        assert SpacyParser not in ADSString.plugins
        # reset backend
        ADSString.nlp_backend(backend="nltk")
        assert NLTKParser in ADSString.plugins


class TestFeatureTypesUtils:
    """Unittests for the Feature Types Utils module."""

    def test_is_boolean(self):
        """Tests checking if value type is boolean."""
        assert is_boolean("1") is True
        assert is_boolean(1) is True
        assert is_boolean(True) is True
        assert is_boolean("Yes") is True
        assert is_boolean("yes") is True
        assert is_boolean("y") is True
        assert is_boolean("Y") is True
        assert is_boolean(False) is True
        assert is_boolean("No") is True
        assert is_boolean("no") is True
        assert is_boolean("n") is True
        assert is_boolean("N") is True
        assert is_boolean(None) is False
        assert is_boolean("3") is False
        assert is_boolean("a") is False
