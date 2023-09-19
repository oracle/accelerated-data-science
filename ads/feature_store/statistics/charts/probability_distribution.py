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


class ProbabilityDistribution(AbsFeatureStat):
    CONST_DENSITY = "density"
    CONST_BINS = "bins"
    CONST_PROBABILITY_DISTRIBUTION_TITLE = "Probability Distribution"

    def __init__(self, density: List, bins: List):
        self.density = density
        self.bins = bins

    @classmethod
    def from_json(cls, json_dict: dict):
        if json_dict is not None:
            return cls(
                density=json_dict.get(ProbabilityDistribution.CONST_DENSITY),
                bins=json_dict.get(ProbabilityDistribution.CONST_BINS),
            )
        else:
            return None

    def add_to_figure(self, fig: Figure, xaxis: int, yaxis: int):
        xaxis_str, yaxis_str, x_str, y_str = self.get_x_y_str_axes(xaxis, yaxis)
        if (
            type(self.density) == list
            and type(self.bins) == list
            and 0 < len(self.density) == len(self.bins) > 0
        ):
            fig.add_bar(
                x=self.bins,
                y=self.density,
                xaxis=x_str,
                yaxis=y_str,
                name="",
            )
        fig.layout.annotations[xaxis].text = self.CONST_PROBABILITY_DISTRIBUTION_TITLE
        fig.layout[xaxis_str]["title"] = "Bins"
        fig.layout[yaxis_str]["title"] = "Density"
