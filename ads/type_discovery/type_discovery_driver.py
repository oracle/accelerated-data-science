#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import

import pandas as pd

from ads.common import utils
from ads.type_discovery.constant_detector import ConstantDetector
from ads.type_discovery.continuous_detector import ContinuousDetector
from ads.type_discovery.credit_card_detector import CreditCardDetector
from ads.type_discovery.datetime_detector import DateTimeDetector
from ads.type_discovery.discrete_detector import DiscreteDetector
from ads.type_discovery.document_detector import DocumentDetector
from ads.type_discovery.latlon_detector import LatLonDetector
from ads.type_discovery.phone_number_detector import PhoneNumberDetector
from ads.type_discovery.unknown_detector import UnknownDetector
from ads.type_discovery.zipcode_detector import ZipCodeDetector
from ads.type_discovery.typed_feature import ContinuousTypedFeature

#
# these should be in precendence order, from low to high
#
discovery_plugins = [
    ConstantDetector(),
    DocumentDetector(),
    ZipCodeDetector(),
    LatLonDetector(),
    CreditCardDetector(),
    PhoneNumberDetector(),
    DateTimeDetector(),
    DiscreteDetector(),
    ContinuousDetector(),
    UnknownDetector(),
]


class TypeDiscoveryDriver:

    #
    # takes a pandas series
    #
    def discover(self, name: str, s: pd.Series, is_target: bool = False):
        """return the type of series

        Parameters
        ----------
        name : type
            variable name to discover.
        s : type
            series of values to 'type'
        is_target : type
            when true the rules differ, any continuous is contunuous regardless of other rules

        Returns
        -------
        type
            one of:

                ConstantDetector,
                DocumentDetector,
                ZipCodeDetector,
                LatLonDetector,
                CreditCardDetector,
                PhoneNumberDetector,
                DateTimeDetector,
                DiscreteDetector,
                ContinuousDetector,

        """

        assert (
            type(s).__name__ == "Series"
        ), "Type discovery can only be performed on a pandas.Series"

        if is_target and ContinuousDetector._target_is_continuous(s):
            return ContinuousTypedFeature.build(name, s)

        #
        # to lazily evaluate the discover method we use a generator expression iterable
        #
        return utils.first_not_none(
            (plugin.discover(name, s) for plugin in discovery_plugins)
        )
