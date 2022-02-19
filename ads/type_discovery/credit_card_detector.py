#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
NOTE:

    There's an opportunity here to generate a new feature, credict card numbers are not preditive because they
    don't generalize, however, if the feature is replaced by the type of card that might be predictive.

    - Visa: ^4[0-9]{12}(?:[0-9]{3})?$ All Visa card numbers start with a 4. New cards have 16 digits. Old cards have 13.
    - MasterCard: ^(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}$ MasterCard numbers
        either start with the numbers 51 through 55 or with the numbers 2221 through 2720. All have 16 digits.
    - American Express: ^3[47][0-9]{13}$ American Express card numbers start with 34 or 37 and have 15 digits.
    - Diners Club: ^3(?:0[0-5]|[68][0-9])[0-9]{11}$ Diners Club card numbers begin with 300 through 305, 36 or 38.
        All have 14 digits. There are Diners Club cards that begin with 5 and have 16 digits. These are a joint
        venture between Diners Club and MasterCard, and should be processed like a MasterCard.
    - Discover: ^6(?:011|5[0-9]{2})[0-9]{12}$ Discover card numbers begin with 6011 or 65. All have 16 digits.
    - JCB: ^(?:2131|1800|35\d{3})\d{11}$ JCB cards beginning with 2131 or 1800 have 15 digits.
        JCB cards beginning with 35 have 16 digits.

"""

from __future__ import print_function, absolute_import, division

import re

import pandas as pd

from ads.type_discovery import logger
from ads.type_discovery.abstract_detector import AbstractTypeDiscoveryDetector
from ads.type_discovery.typed_feature import CreditCardTypedFeature


class CreditCardDetector(AbstractTypeDiscoveryDetector):

    _max_sample_size_to_luhn_check = 1000
    _pattern_string = r"""^(?:4[0-9]{12}(?:[0-9]{3})?            # Visa
                             |  (?:5[1-5][0-9]{2}                # MasterCard
                                 | 222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}
                             |  3[47][0-9]{13}                   # American Express
                             |  3(?:0[0-5]|[68][0-9])[0-9]{11}   # Diners Club
                             |  6(?:011|5[0-9]{2})[0-9]{12}      # Discover
                             |  (?:2131|1800|35\d{3})\d{11}      # JCB
                             |  (5018|5020|5038|5612|5893|6304|6759|6761|6762|6763|0604|6390)\d+$   # Maestro
                             |  ^(5[06789]|6)[0-9]{0,}$          # Maestro
                             |  ^4[0-9]{12}(?:[0-9]{6})?$        #Visa 19 digit
                            )$"""

    def luhn_checksum(self, card_number):
        def digits_of(n):
            return [int(d) for d in str(n)]

        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = 0
        checksum += sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        return checksum % 10

    def is_luhn_valid(self, card_number):
        return self.luhn_checksum(card_number) == 0

    def is_credit_card(self, name, values):
        cc = re.compile(CreditCardDetector._pattern_string, re.VERBOSE)
        # since the nulls have been previously filtered we can safely do "all"
        samp = (
            values
            if values.size <= CreditCardDetector._max_sample_size_to_luhn_check
            else values.sample(n=CreditCardDetector._max_sample_size_to_luhn_check)
        )

        if samp.dtype.name in ["float16", "float32", "float64"]:
            if samp.apply(float.is_integer).all():
                samp = samp.fillna(0.0).astype(int)

        if samp.dtype.name in ["int16", "int32", "int64"]:
            samp = samp.astype(str)

        if all([cc.match(str(x)) for x in samp]):
            #
            # iff the pattern matching succeeds do we try the luhn algorithm on a sample
            #
            return all([self.is_luhn_valid(x) for x in samp])

        return False

    def discover(self, name, series):
        candidates = series.loc[~series.isnull()]

        if self.is_credit_card(name, candidates.head(1000)):
            logger.debug("column [{}]/[{}] credit card".format(name, series.dtype))
            return CreditCardTypedFeature.build(name, series)

        return False
