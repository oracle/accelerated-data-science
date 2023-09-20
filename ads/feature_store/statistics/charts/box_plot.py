#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from typing import List

from ads.common.decorator.runtime_dependency import OptionalDependency
from ads.feature_store.statistics.charts.abstract_feature_stat import AbsFeatureStat
from ads.feature_store.statistics.charts.frequency_distribution import (
    FrequencyDistribution,
)
from ads.feature_store.statistics.generic_feature_value import GenericFeatureValue

try:
    from plotly.graph_objs import Figure
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `plotly` module was not found. Please run `pip install "
        f"{OptionalDependency.FEATURE_STORE}`."
    )


class BoxPlot(AbsFeatureStat):
    CONST_MIN = "Min"
    CONST_MAX = "Max"
    CONST_QUARTILES = "Quartiles"
    CONST_SD = "StandardDeviation"
    CONST_MEAN = "Mean"
    CONST_BOX_PLOT_TITLE = "Box Plot"
    CONST_IQR = "IQR"
    CONST_FREQUENCY_DISTRIBUTION = "FrequencyDistribution"

    class Quartiles:
        CONST_Q1 = "q1"
        CONST_Q2 = "q2"
        CONST_Q3 = "q3"

        def __init__(self, q1: float, q2: float, q3: float):
            self.q1 = q1
            self.q2 = q2
            self.q3 = q3

        @classmethod
        def from_json(cls, json_dict: dict) -> "BoxPlot.Quartiles":
            if json_dict is not None:
                return cls(
                    json_dict.get(cls.CONST_Q1),
                    json_dict.get(cls.CONST_Q2),
                    json_dict.get(cls.CONST_Q3),
                )
            else:
                return None

    def __init__(
        self,
        mean: float,
        median: float,
        sd: float,
        q1: float,
        q3: float,
        boxpoints: List[float],
    ):
        self.mean = mean
        self.median = median
        self.q1 = q1
        self.q3 = q3
        self.sd = sd
        self.iqr = self.q3 - self.q1
        self.boxpoints = boxpoints

    def add_to_figure(self, fig: Figure, xaxis: int, yaxis: int):
        xaxis_str, yaxis_str, x_str, y_str = self.get_x_y_str_axes(xaxis, yaxis)
        fig.add_box(
            mean=[self.mean],
            median=[self.median],
            q1=[self.q1],
            q3=[self.q3],
            sd=[self.sd],
            y=[self.boxpoints],
            upperfence=[self.q3 + 1.5 * self.iqr],
            lowerfence=[self.q1 - 1.5 * self.iqr],
            xaxis=x_str,
            yaxis=y_str,
            name="",
            jitter=0,
        )
        fig.layout.annotations[xaxis].text = self.CONST_BOX_PLOT_TITLE
        fig.layout[yaxis_str]["title"] = "Values"

    @staticmethod
    def get_boxpoints_from_frequency_distribution(
        frequency_distribution: FrequencyDistribution,
    ) -> List[float]:
        boxpoints = []
        if frequency_distribution is not None:
            for frequency, bin in zip(
                frequency_distribution.frequency, frequency_distribution.bins
            ):
                boxpoints.extend([bin] * frequency)

        return boxpoints

    @classmethod
    def from_json(cls, json_dict: dict) -> "BoxPlot":
        if type(json_dict) is dict and json_dict.get(cls.CONST_QUARTILES) is not None:
            quartiles = cls.Quartiles.from_json(json_dict.get(cls.CONST_QUARTILES))
            return cls(
                mean=GenericFeatureValue.from_json(json_dict.get(cls.CONST_MEAN)).val,
                median=quartiles.q2,
                sd=GenericFeatureValue.from_json(json_dict.get(cls.CONST_SD)).val,
                q1=quartiles.q1,
                q3=quartiles.q3,
                boxpoints=cls.get_boxpoints_from_frequency_distribution(
                    FrequencyDistribution.from_json(
                        json_dict.get(cls.CONST_FREQUENCY_DISTRIBUTION)
                    )
                ),
            )
        else:
            return None
