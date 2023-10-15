#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import json

import pytest

from ads.feature_store.statistics.charts.abstract_feature_plot import AbsFeaturePlot

from ads.feature_store.statistics.charts.top_k_frequent_elements import (
    TopKFrequentElements,
)


def test_failed_top_k_validation():
    with pytest.raises(AssertionError):
        assert TopKFrequentElements.TopKFrequentElement("", 3, 4, 3)


def test_successful_top_k_validation():
    TopKFrequentElements.TopKFrequentElement("str", 3, 4, 3)


def test_top_k_init_from_json():
    json_dict = '{"value": "AA", "estimate": 14, "lower_bound": 13, "upper_bound": 15}'
    plot = TopKFrequentElements.TopKFrequentElement.from_json(json.loads(json_dict))
    assert plot.value == "AA"
    assert plot.estimate == 14
    assert plot.lower_bound == 13
    assert plot.upper_bound == 15


def test_failed_top_k_elements_validation():
    with pytest.raises(AssertionError):
        TopKFrequentElements([None])


def test_successful_top_k_elements_validation():
    element = TopKFrequentElements.TopKFrequentElement("str", 3, 4, 3)
    TopKFrequentElements([element])


def test_top_k_elements_init_from_json():
    json_dict = '{"value": [{"value": "AA", "estimate": 14, "lower_bound": 14, "upper_bound": 14}, {"value": "B6", "estimate": 12, "lower_bound": 12, "upper_bound": 12}, {"value": "UA", "estimate": 11, "lower_bound": 11, "upper_bound": 11}, {"value": "AS", "estimate": 11, "lower_bound": 11, "upper_bound": 11}, {"value": "DL", "estimate": 11, "lower_bound": 11, "upper_bound": 11}, {"value": "NK", "estimate": 11, "lower_bound": 11, "upper_bound": 11}, {"value": "US", "estimate": 8, "lower_bound": 8, "upper_bound": 8}, {"value": "OO", "estimate": 8, "lower_bound": 8, "upper_bound": 8}, {"value": "EV", "estimate": 7, "lower_bound": 7, "upper_bound": 7}, {"value": "HA", "estimate": 5, "lower_bound": 5, "upper_bound": 5}]}'
    plot = TopKFrequentElements.from_json(json.loads(json_dict))
    assert len(plot.elements) == 10
