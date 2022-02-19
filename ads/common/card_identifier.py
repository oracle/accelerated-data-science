#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
credit card patterns refer to https://en.wikipedia.org/wiki/Payment_card_number#Issuer_identification_number_(IIN)
Active and frequent card information
American Express: 34, 37
Diners Club (US & Canada): 54,55
Discover Card: 6011, 622126 - 622925, 624000 - 626999, 628200 - 628899, 64, 65
Master Card: 2221-2720, 51–55
Visa: 4

"""


class card_identify:
    def identify_issue_network(self, card_number):

        """
        Returns the type of credit card based on its digits

        Parameters
        ----------
        card_number: String

        Returns
        -------
        String: A string corresponding to the kind of credit card.
        """
        try:
            first_six_digit = int(str(card_number)[:6])
            first_four_digit = int(str(card_number)[:4])
            first_two_digit = int(str(card_number)[:2])
            first_one_digit = int(str(card_number)[:1])

            if first_one_digit == 4:
                return "Visa"
            elif first_two_digit == 34 or first_two_digit == 37:
                return "Amex"
            elif first_two_digit == 54 or first_two_digit == 55:
                return "Diners Club"
            elif (
                first_two_digit == 64
                or first_two_digit == 65
                or first_four_digit == 6011
                or 622126 <= first_six_digit <= 622925
                or 624000 <= first_six_digit <= 626999
                or 628200 <= first_six_digit <= 628899
            ):
                return "Discover"
            elif 2221 <= first_four_digit <= 2720 or 51 <= first_two_digit <= 55:
                return "MasterCard"
            else:
                return "Unknown"
        except:
            return "Unknown"


if __name__ == "__main__":
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
