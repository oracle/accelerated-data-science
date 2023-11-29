#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json

import pytest
from ads.feature_store.statistics.charts.frequency_distribution import (
    FrequencyDistribution,
)

from ads.feature_store.statistics.charts.abstract_feature_plot import AbsFeaturePlot

from ads.feature_store.statistics.charts.frequency_distribution import (
    FrequencyDistribution,
)


def test_failed_freq_dist_validation():
    with pytest.raises(AssertionError):
        FrequencyDistribution(1, 2)
    with pytest.raises(AssertionError):
        FrequencyDistribution([0], [1, 2])


def test_successful_freq_dist_validation():
    FrequencyDistribution([0, 1], [1, 2])


def test_init_from_json():
    json_dict = '{"bins": [2015.0], "frequency": [100]}'
    plot: FrequencyDistribution = FrequencyDistribution.from_json(
        json.loads(json_dict), ignore_errors=False
    )
    assert plot.bins == [2015.0]
    assert plot.frequency == [100]
