#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json

import pytest

from ads.feature_store.statistics.charts.abstract_feature_plot import AbsFeaturePlot

from ads.feature_store.statistics.charts.probability_distribution import (
    ProbabilityDistribution,
)


def test_failed_prob_dist_validation():
    with pytest.raises(AssertionError):
        ProbabilityDistribution(1, 2)
    with pytest.raises(AssertionError):
        ProbabilityDistribution([0], [1, 2])


def test_successful_prob_dist_validation():
    ProbabilityDistribution([0, 1], [1, 2])


def test_init_from_json():
    json_dict = '{"bins": [2015.0], "density": [1.0]}'
    plot: ProbabilityDistribution = ProbabilityDistribution.from_json(
        json.loads(json_dict), ignore_errors=False
    )
    assert plot.bins == [2015.0]
    assert plot.density == [1.0]
