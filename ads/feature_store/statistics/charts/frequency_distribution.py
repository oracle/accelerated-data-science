#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List
from ads.common.decorator.runtime_dependency import OptionalDependency
from ads.feature_store.statistics.charts.abstract_feature_stat import AbsFeatureStat

try:
    from plotly.graph_objs import Figure
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `plotly` module was not found. Please run `pip install "
        f"{OptionalDependency.FEATURE_STORE}`."
    )


class FrequencyDistribution(AbsFeatureStat):
    CONST_FREQUENCY = "frequency"
    CONST_BINS = "bins"
    CONST_FREQUENCY_DISTRIBUTION_TITLE = "Frequency Distribution"

    def __validate__(self):
        if not (
            type(self.frequency) == list
            and type(self.bins) == list
            and 0 < len(self.frequency) == len(self.bins) > 0
        ):
            raise self.ValidationFailedException()

    def __init__(self, frequency: List, bins: List):
        self.frequency = frequency
        self.bins = bins
        super().__init__()

    @classmethod
    def __from_json__(cls, json_dict: dict) -> "FrequencyDistribution":
        return FrequencyDistribution(
            frequency=json_dict.get(cls.CONST_FREQUENCY),
            bins=json_dict.get(cls.CONST_BINS),
        )

    def add_to_figure(self, fig: Figure, xaxis: int, yaxis: int):
        xaxis_str, yaxis_str, x_str, y_str = self.get_x_y_str_axes(xaxis, yaxis)
        if (
            type(self.frequency) == list
            and type(self.bins) == list
            and 0 < len(self.frequency) == len(self.bins) > 0
        ):
            fig.add_bar(
                x=self.bins, y=self.frequency, xaxis=x_str, yaxis=y_str, name=""
            )
            fig.layout.annotations[xaxis].text = self.CONST_FREQUENCY_DISTRIBUTION_TITLE
            fig.layout[xaxis_str]["title"] = "Bins"
            fig.layout[yaxis_str]["title"] = "Frequency"
