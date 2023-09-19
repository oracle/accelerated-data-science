#!/usr/bin/env python
# -*- coding: utf-8; -*-


# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
class GenericFeatureValue:
    CONST_VALUE = "value"

    def __init__(self, val: any):
        self.val = val

    @classmethod
    def from_json(cls, json_dict: dict) -> "GenericFeatureValue":
        if json_dict is not None:
            return GenericFeatureValue(
                val=json_dict.get(cls.CONST_VALUE),
            )
        else:
            return None
