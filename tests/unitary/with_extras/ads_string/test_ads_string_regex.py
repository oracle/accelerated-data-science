#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.feature_engineering.feature_type.adsstring.string import ADSString
import pytest


class TestADSStringRegexMixin:
    def test_email(self):
        x = ADSString(
            "get in touch with my associate at john.smith@gmail.com to schedule"
        )
        assert "john.smith@gmail.com" in x.email
        x_redacted = x.redact(fields=["email"])
        assert isinstance(x_redacted, ADSString)
        assert x_redacted == "get in touch with my associate at [EMAIL] to schedule"

    @pytest.mark.parametrize(
        "the_time", ["09:45", "9:45", "23:45", "9:00am", "9am", "9:00 A.M.", "9:00 pm"]
    )
    def test_time(self, the_time):
        x = ADSString(f"call me tomorrow at {the_time} to schedule")
        assert the_time == x.time[0]

    @pytest.mark.parametrize(
        "the_date",
        [
            "January 19th, 2014",
            "Jan. 19th, 2014",
            "Jan 19 2014",
            "19 Jan 2014",
            "September 19, 1965",
        ],
    )
    def test_date(self, the_date):
        x = ADSString(f"lunch on {the_date} with the team")
        assert the_date == x.date[0]

    @pytest.mark.parametrize(
        "the_phone",
        [
            "12345678900",
            "1234567890",
            "+1 234 567 8900",
            "234-567-8900",
            "1-234-567-8900",
            "1.234.567.8900",
            "5678900",
            "567-8900",
            "(123) 456 7890",
            "+41 22 730 5989",
            "(+41) 22 730 5989",
            "+442345678900",
            "(523)222-8888 ext 527",
            "(523)222-8888x623",
            "(523)222-8888 x623",
            "(523)222-8888 x 623",
            "(523)222-8888EXT623",
            "523-222-8888EXT623",
            "(523) 222-8888 x 623",
        ],
    )
    def test_phone_number(self, the_phone):
        x = ADSString(f"you can reach me on {the_phone} anytime")
        assert the_phone in x.phone_number_US
        x_redacted = x.redact(fields=["phone_number_US", "phone_number_US_with_ext"])
        assert isinstance(x_redacted, ADSString)

    @pytest.mark.parametrize(
        "the_link",
        [
            "www.oracle.com",
            "oracle.com",
            "http://www.oracle.com",
            "www.oracle.com/?query=cat" "sub.example.com",
            "http://www.oracle.com/%&#/?q=cat",
            "oracle.com",
        ],
    )
    def test_links(self, the_link):
        x = ADSString(f"you can always find me lurking on {the_link} :)")
        assert the_link == x.link[0]

    @pytest.mark.parametrize(
        "the_ips",
        [
            "127.0.0.1",
            "192.168.1.1",
            "8.8.8.8",
            "fe80:0000:0000:0000:0204:61ff:fe9d:f156",
            "fe80:0:0:0:204:61ff:fe9d:f156",
            "fe80::204:61ff:fe9d:f156",
            "fe80:0000:0000:0000:0204:61ff:254.157.241.86",
            "fe80:0:0:0:0204:61ff:254.157.241.86",
            "fe80::204:61ff:254.157.241.86",
            "::1",
        ],
    )
    def test_ip(self, the_ips):
        x = ADSString(f"you can always find me lurking on {the_ips} :)")
        assert the_ips in x.ip

    @pytest.mark.parametrize("the_price", ["$42.12", "$1", "$1,000", "$10,000.00"])
    def test_prices(self, the_price):
        x = ADSString(f"the cable cost me {the_price} - just for a new cable!")
        assert the_price == x.price[0]

    @pytest.mark.parametrize(
        "the_addr",
        [
            "504 parkwood drive",
            "3 elm boulevard",
            "500 elm street",
            "400 Oracle Parkway",
        ],
    )
    def test_address(self, the_addr):
        x = ADSString(f"she lived at {the_addr} back in 1982" * 2)
        assert the_addr == x.address[0]
        x_redacted = x.redact(fields=["address"])
        assert isinstance(x_redacted, ADSString)
        assert x_redacted == "she lived at [ADDRESS] back in 1982" * 2

    @pytest.mark.parametrize(
        "the_zipcode", ["02540", "02540-4119", "90210", "94065", "94501", "94037"]
    )
    def test_zip_code(self, the_zipcode):
        x = ADSString(f"her zipcode was {the_zipcode} back in college")
        assert the_zipcode == x.zip_code[0]

    @pytest.mark.parametrize("the_ssn", ["523 23 4566", "523-04-1234", "618-40-7041"])
    def test_ssn(self, the_ssn):
        x = ADSString(f"she used {the_ssn} back in 1982")
        assert the_ssn == x.ssn[0]
        x_redacted = x.redact(fields=["ssn"])
        assert isinstance(x_redacted, ADSString)
        assert x_redacted == "she used [SSN] back in 1982"

    def test_redact(self):
        x = ADSString(
            "Nana lives at 267 Epsilon Street. Her number is 314-159-2653. Her email is nana7@real.com."
        )
        x_redacted = x.redact(
            fields=["address", "phone_number_US", "email", "non-exist"]
        )
        assert isinstance(x_redacted, ADSString)
        assert (
            x_redacted
            == "Nana lives at [ADDRESS] Her number is [PHONE_NUMBER_US]. Her email is [EMAIL]."
        )

        x = ADSString(
            "Nana lives at 267 Epsilon Street. Her number is 314-159-2653. Her email is nana7@real.com."
        )
        x_redacted = x.redact(
            fields={
                "address": "HIDDEN_ADDRESS",
                "phone_number_US": "HIDDEN_PHONE_NUMBER",
                "email": "HIDDEN_EMAIL",
                "abs": "non-existing",
            }
        )
        assert isinstance(x_redacted, ADSString)
        assert (
            x_redacted
            == "Nana lives at [HIDDEN_ADDRESS] Her number is [HIDDEN_PHONE_NUMBER]. Her email is [HIDDEN_EMAIL]."
        )
