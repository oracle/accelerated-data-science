#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Address
    Type representing address.
Boolean
    Type representing binary values True/False.
Category
    Type representing discrete unordered values.
Constant
    Type representing constant values.
Continuous
    Type representing continuous values.
CreditCard
    Type representing credit card numbers.
DateTime
    Type representing date and/or time.
Document
    Type representing document values.
Discrete
    Type representing discrete values.
FeatureType
    Base class for all feature types.
GIS
    Type representing geographic information.
Integer
    Type representing integer values.
IpAddress
    Type representing IP Address.
IpAddressV4
    Type representing IP Address V4.
IpAddressV6
    Type representing IP Address V6.
LatLong
    Type representing longitude and latitute.
Object
    Type representing object.
Ordinal
    Type representing ordered values.
PhoneNumber
    Type representing phone numbers.
String
    Type representing string values.
Tag
    Free form tag.
Text
    Type representing text values.
ZipCode
    Type representing postal code.
Unknown
    Type representing third-party dtypes.
"""

from ads.feature_engineering.feature_type.address import Address
from ads.feature_engineering.feature_type.boolean import Boolean
from ads.feature_engineering.feature_type.category import Category
from ads.feature_engineering.feature_type.constant import Constant
from ads.feature_engineering.feature_type.continuous import Continuous
from ads.feature_engineering.feature_type.creditcard import CreditCard
from ads.feature_engineering.feature_type.datetime import DateTime
from ads.feature_engineering.feature_type.document import Document
from ads.feature_engineering.feature_type.discrete import Discrete
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.feature_type.gis import GIS
from ads.feature_engineering.feature_type.integer import Integer
from ads.feature_engineering.feature_type.ip_address import IpAddress
from ads.feature_engineering.feature_type.ip_address_v4 import IpAddressV4
from ads.feature_engineering.feature_type.ip_address_v6 import IpAddressV6
from ads.feature_engineering.feature_type.lat_long import LatLong
from ads.feature_engineering.feature_type.object import Object
from ads.feature_engineering.feature_type.ordinal import Ordinal
from ads.feature_engineering.feature_type.phone_number import PhoneNumber
from ads.feature_engineering.feature_type.string import String
from ads.feature_engineering.feature_type.base import Tag
from ads.feature_engineering.feature_type.text import Text
from ads.feature_engineering.feature_type.zip_code import ZipCode
from ads.feature_engineering.feature_type.unknown import Unknown
from ads.feature_engineering.feature_type.handler import warnings as w
from ads.feature_engineering.feature_type.adsstring.string import ADSString

Address.warning.register("missing_values", w.missing_values_handler)

Boolean.warning.register("missing_values", w.missing_values_handler)

Category.warning.register("missing_values", w.missing_values_handler)
Category.warning.register("high_cardinality", w.high_cardinality_handler)

Constant.warning.register("missing_values", w.missing_values_handler)

Continuous.warning.register("missing_values", w.missing_values_handler)
Continuous.warning.register("zeros", w.zeros_handler)
Continuous.warning.register("skew_handler", w.skew_handler)

CreditCard.warning.register("missing_values", w.missing_values_handler)
CreditCard.warning.register("high_cardinality", w.high_cardinality_handler)

DateTime.warning.register("missing_values", w.missing_values_handler)
DateTime.warning.register("high_cardinality", w.high_cardinality_handler)

Document.warning.register("missing_values", w.missing_values_handler)

GIS.warning.register("missing_values", w.missing_values_handler)

Integer.warning.register("missing_values", w.missing_values_handler)
Integer.warning.register("zeros", w.zeros_handler)

IpAddress.warning.register("missing_values", w.missing_values_handler)

IpAddressV4.warning.register("missing_values", w.missing_values_handler)

IpAddressV6.warning.register("missing_values", w.missing_values_handler)

LatLong.warning.register("missing_values", w.missing_values_handler)

Object.warning.register("missing_values", w.missing_values_handler)
Object.warning.register("high_cardinality", w.high_cardinality_handler)

Ordinal.warning.register("missing_values", w.missing_values_handler)

PhoneNumber.warning.register("missing_values", w.missing_values_handler)
PhoneNumber.warning.register("high_cardinality", w.high_cardinality_handler)

String.warning.register("missing_values", w.missing_values_handler)
String.warning.register("high_cardinality", w.high_cardinality_handler)

Text.warning.register("missing_values", w.missing_values_handler)

ZipCode.warning.register("missing_values", w.missing_values_handler)
ZipCode.warning.register("high_cardinality", w.high_cardinality_handler)
