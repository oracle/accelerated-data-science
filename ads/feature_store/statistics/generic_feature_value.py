#!/usr/bin/env python
# -*- coding: utf-8; -*-
from ads.feature_store.statistics.abs_feature_value import AbsFeatureValue


# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
class GenericFeatureValue(AbsFeatureValue):
    CONST_VALUE = "value"

    def __init__(self, val: any):
        self.val = val
        super().__init__()

    def __validate__(self):
        pass

    @classmethod
    def __from_json_v2__(cls, json_dict: dict) -> "GenericFeatureValue":
        val = None
        if type(json_dict) == dict:
            metric_data = json_dict.get(AbsFeatureValue.CONST_METRIC_DATA)
            val = metric_data[0]
        return GenericFeatureValue(val=val)

    @classmethod
    def __from_json__(cls, json_dict: dict, version: int = 1) -> "GenericFeatureValue":
        if version == 2:
            return cls.__from_json_v2__(json_dict)
        val = None
        if type(json_dict) == dict:
            val = json_dict.get(cls.CONST_VALUE)
        return GenericFeatureValue(val=val)
