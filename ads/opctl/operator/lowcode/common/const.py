#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from ads.common.extended_enum import ExtendedEnumMeta


class DataColumns:
    Series = "Series"
    Date = "Date"


class ImputationMethods(str, metaclass=ExtendedEnumMeta):
    MEAN = "mean"
    MEDIAN = "median"
    LINEAR_INTERPOLATION = "linear_interpolation"
    NONE = "None"


class OutlierTreatmentMethods(str, metaclass=ExtendedEnumMeta):
    ZSCORE_WITH_MEAN = "zscore_with_mean"
    NONE = "None"
