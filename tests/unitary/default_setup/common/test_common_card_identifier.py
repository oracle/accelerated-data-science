#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from faker import Faker
import pandas as pd

from ads.common.card_identifier import card_identify
from ads.type_discovery.credit_card_detector import CreditCardDetector
from ads.type_discovery.typed_feature import CreditCardTypedFeature


def test_credit_card():
    fake = Faker()
    Faker.seed(42)

    dd = CreditCardDetector()
    _visa = [
        "4532640527811543",
        "4556929308150929",
        "4539944650919740",
        "4485348152450846",
        "4556593717607190",
    ]
    _mastercard = [
        "5334180299390324",
        "5111466404826446",
        "5273114895302717",
        "5430972152222336",
        "5536426859893306",
    ]
    _amex = [
        "371025944923273",
        "374745112042294",
        "340984902710890",
        "375767928645325",
        "370720852891659",
    ]

    assert not isinstance(
        dd.discover("not_cc", pd.Series([None, "94065", "94065", "94065"])),
        CreditCardTypedFeature,
    )
    assert isinstance(dd.discover("visa", pd.Series(_visa)), CreditCardTypedFeature)
    assert isinstance(
        dd.discover("mastercard", pd.Series(_mastercard)), CreditCardTypedFeature
    )
    assert isinstance(dd.discover("amex", pd.Series(_amex)), CreditCardTypedFeature)

    assert isinstance(
        dd.discover(
            "fake", pd.Series([fake.credit_card_number() for i in range(1000)])
        ),
        CreditCardTypedFeature,
    )

    assert card_identify().identify_issue_network(_visa[0]) == "Visa"
    assert card_identify().identify_issue_network(_mastercard[2]) == "MasterCard"
    assert card_identify().identify_issue_network(_amex[1]) == "Amex"
