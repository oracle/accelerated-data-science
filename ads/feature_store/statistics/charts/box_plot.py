#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from typing import List, Union

from ads.feature_store.statistics.abs_feature_value import AbsFeatureValue

from ads.common.decorator.runtime_dependency import OptionalDependency
from ads.feature_store.statistics.charts.abstract_feature_plot import AbsFeaturePlot
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


class BoxPlot(AbsFeaturePlot):
    CONST_MIN = "Min"
    CONST_MAX = "Max"
    CONST_QUARTILES = "Quartiles"
    CONST_SD = "StandardDeviation"
    CONST_MEAN = "Mean"
    CONST_BOX_PLOT_TITLE = "Box Plot"
    CONST_IQR = "IQR"
    CONST_BOX_POINTS = "box_points"

    class Quartiles(AbsFeatureValue):
        CONST_Q1 = "q1"
        CONST_Q2 = "q2"
        CONST_Q3 = "q3"

        def __init__(self, q1: float, q2: float, q3: float):
            self.q1 = q1
            self.q2 = q2
            self.q3 = q3
            super().__init__()

        @classmethod
        def __from_json__(
            cls, json_dict: dict, version: int = 1
        ) -> "BoxPlot.Quartiles":
            if version == 2:
                return cls.__from_json_v2__(json_dict)
            return cls(
                json_dict.get(cls.CONST_Q1),
                json_dict.get(cls.CONST_Q2),
                json_dict.get(cls.CONST_Q3),
            )

        @classmethod
        def __from_json_v2__(cls, json_dict: dict) -> "BoxPlot.Quartiles":
            metric_data = json_dict.get(AbsFeatureValue.CONST_METRIC_DATA)
            return cls(metric_data[0], metric_data[1], metric_data[2])

        def __validate__(self):
            assert type(self.q1) == type(self.q2) == type(self.q3) == int or float
            assert self.q3 >= self.q2 >= self.q1

    def __init__(
        self,
        mean: float,
        median: float,
        sd: float,
        q1: float,
        q3: float,
        box_points: List[float],
    ):
        self.mean = mean
        self.median = median
        self.q1 = q1
        self.q3 = q3
        self.sd = sd
        self.iqr = self.q3 - self.q1
        self.box_points = box_points
        super().__init__()

    def __validate__(self):
        assert self.q1 is not None
        assert self.q3 is not None
        assert self.iqr is not None
        assert self.q3 is not None
        assert self.median is not None
        assert self.mean is not None
        assert type(self.box_points) is list
        assert len(self.box_points) > 0

    def add_to_figure(self, fig: Figure, xaxis: int, yaxis: int):
        xaxis_str, yaxis_str, x_str, y_str = self.get_x_y_str_axes(xaxis, yaxis)
        fig.add_box(
            notched=False,
            boxmean=False,
            mean=[self.mean],
            median=[self.median],
            q1=[self.q1],
            q3=[self.q3],
            sd=[self.sd],
            y=[self.box_points],
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
    def get_box_points_from_frequency_distribution(
        frequency_distribution: FrequencyDistribution,
    ) -> List[float]:
        # box_points = []
        if (
            frequency_distribution is not None
            and frequency_distribution.frequency is not None
            and frequency_distribution.bins is not None
        ):
            return [
                bin_dist
                for frequency, bin_dist in zip(
                    frequency_distribution.frequency, frequency_distribution.bins
                )
                if frequency > 0
            ]
        else:
            return []

    @classmethod
    def __from_json__(cls, json_dict: dict, version: int = 1) -> "BoxPlot":
        quartiles = cls.Quartiles.from_json(json_dict.get(cls.CONST_QUARTILES), version)
        return cls(
            mean=GenericFeatureValue.from_json(
                json_dict.get(cls.CONST_MEAN), version
            ).val,
            median=quartiles.q2,
            sd=GenericFeatureValue.from_json(json_dict.get(cls.CONST_SD), version).val,
            q1=quartiles.q1,
            q3=quartiles.q3,
            box_points=json_dict.get(cls.CONST_BOX_POINTS),
        )
