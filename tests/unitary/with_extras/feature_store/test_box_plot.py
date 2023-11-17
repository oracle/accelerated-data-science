#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json

import pytest
from ads.feature_store.statistics.charts.frequency_distribution import (
    FrequencyDistribution,
)

from ads.feature_store.statistics.charts.box_plot import BoxPlot


def test_failed_box_plot_validation():
    with pytest.raises(AssertionError):
        assert BoxPlot(2, None, 4, 3, 1, [1, 2])


def test_successful_box_plot_validation():
    BoxPlot(2, 3, 4, 1, 3, [1, 2])


def test_successful_box_points_from_frequency_distribution():
    bins = [3, 4, 5]
    frequency = [0, 1, 2]
    distribution = FrequencyDistribution(frequency=frequency, bins=bins)
    assert BoxPlot.get_box_points_from_frequency_distribution(distribution) == [4, 5]


def test_unsuccessful_box_points_from_frequency_distribution():
    assert BoxPlot.get_box_points_from_frequency_distribution(None) == []


def test_init_from_json():
    json_dict = '{"Skewness": {"value": null}, "StandardDeviation": {"value": 0.0}, "Min": {"value": 2015.0}, "IsConstantFeature": {"value": true}, "IQR": {"value": 0.0}, "Range": {"value": 0.0}, "ProbabilityDistribution": {"bins": [2015.0], "density": [1.0]}, "Variance": {"value": 0.0}, "TypeMetric": {"string_type_count": 0, "integral_type_count": 100, "fractional_type_count": 0, "boolean_type_count": 0}, "FrequencyDistribution": {"bins": [2015.0], "frequency": [100]}, "Count": {"total_count": 100, "missing_count": 0, "missing_count_percentage": 0.0}, "Max": {"value": 2015.0}, "DistinctCount": {"value": 1}, "Sum": {"value": 201500.0}, "IsQuasiConstantFeature": {"value": true}, "Quartiles": {"q1": 2015.0, "q2": 2015.0, "q3": 2016.0}, "Mean": {"value": 2017.0}, "Kurtosis": {"value": null}, "box_points":[0,1]}'
    plot: BoxPlot = BoxPlot.from_json(json.loads(json_dict), ignore_errors=False)
    assert plot.mean == 2017.0
    assert plot.median == 2015.0
    assert plot.q1 == 2015.0
    assert plot.q3 == 2016.0
    assert plot.sd == 0.0


def test_failed_quartile_validation():
    with pytest.raises(AssertionError):
        BoxPlot.Quartiles(2, 1, 0)


def test_successful_quartile_validation():
    BoxPlot.Quartiles(0, 0, 0)


def test_quartile_init_from_json():
    json_dict = '{"q1": 2015.0, "q2": 2015.0, "q3": 2016.0}'
    quartile = BoxPlot.Quartiles.from_json(json.loads(json_dict))
    assert quartile.q1 == 2015.0
    assert quartile.q2 == 2015.0
    assert quartile.q3 == 2016.0
