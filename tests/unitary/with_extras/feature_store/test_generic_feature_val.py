#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json

from ads.feature_store.statistics.generic_feature_value import GenericFeatureValue


def test_from_empty_json():
    feature = GenericFeatureValue.from_json(None)
    assert feature.val is None


def test_valid_json():
    json_dict = '{"value": 0.0}'
    feature = GenericFeatureValue.from_json(json.loads(json_dict))
    assert feature.val == 0.0
